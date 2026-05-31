import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import numpy as np
from src.utils import non_max_suppression, resize_image_keep_aspect, map_box_to_original

class ProductDetector:
    def __init__(self, model_name="is36e/detr-resnet-50-sku110k", device=None):
        """
        Initializes the DETR Product Detector.
        
        Args:
            model_name (str): Hugging Face model repository name
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing DETR Product Detector on {self.device}...")
        
        # Load processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Could not load processor from {model_name}, falling back to facebook/detr-resnet-50. Error: {e}")
            self.processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
            
        # Load model using custom patched config to prevent dilation/backbone_kwargs validation crashes on newer transformers
        from huggingface_hub import hf_hub_download
        import json
        from transformers import DetrConfig
        
        try:
            config_file = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_file, "r") as f:
                config_dict = json.load(f)
                
            # Clean None fields that crash newer transformers type checkers
            keys_to_delete = ["dilation", "backbone_kwargs"]
            for k in keys_to_delete:
                if k in config_dict and config_dict[k] is None:
                    del config_dict[k]
                    
            config = DetrConfig(**config_dict)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name, config=config).to(self.device)
        except Exception as e:
            print(f"Could not load custom patched config, falling back to standard loading. Error: {e}")
            self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device)
            
        self.model.eval()
        
    def detect(self, pil_image, threshold=0.5, iou_threshold=0.3, max_inference_size=1280):
        """
        Detects products in the image with box sanity filtering.
        
        Args:
            pil_image (PIL.Image.Image): Input image
            threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            max_inference_size (int): Max side dimension for inference speedup
            
        Returns:
            list: Bounding boxes in original coordinates [xmin, ymin, xmax, ymax]
            list: Detection confidence scores
        """
        # Resize image for inference
        resized_img, (orig_w, orig_h), (scale_x, scale_y) = resize_image_keep_aspect(pil_image, max_size=max_inference_size)
        
        # Prepare inputs
        inputs = self.processor(images=resized_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process
        target_sizes = torch.tensor([resized_img.size[::-1]]).to(self.device)
        
        # Use processor to get bounding boxes and scores in resized image scale
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        if len(boxes) == 0:
            return [], []
            
        # Apply box sanity checks in inference scale (discard extremely small or narrow boxes)
        valid_indices = []
        for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            w = xmax - xmin
            h = ymax - ymin
            if w >= 10 and h >= 10 and (w * h) >= 100:
                valid_indices.append(idx)
                
        if len(valid_indices) == 0:
            return [], []
            
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
            
        # Apply NMS to remove duplicate overlapping boxes
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=iou_threshold)
        
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]
        
        # Map boxes back to original coordinate system
        original_boxes = []
        for box in filtered_boxes:
            orig_box = map_box_to_original(box, scale_x, scale_y)
            # Ensure coordinates are within image boundaries
            orig_box[0] = max(0.0, min(orig_box[0], orig_w))
            orig_box[1] = max(0.0, min(orig_box[1], orig_h))
            orig_box[2] = max(0.0, min(orig_box[2], orig_w))
            orig_box[3] = max(0.0, min(orig_box[3], orig_h))
            original_boxes.append(orig_box)
            
        return original_boxes, filtered_scores.tolist()

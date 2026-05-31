import numpy as np
from PIL import Image

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Applies Non-Max Suppression (NMS) to bounding boxes.
    
    Args:
        boxes (np.ndarray): Shape (N, 4) in [xmin, ymin, xmax, ymax]
        scores (np.ndarray): Shape (N,) confidence scores
        iou_threshold (float): IoU overlap threshold
        
    Returns:
        list of int: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Cast to float
    boxes = boxes.astype(float)
    scores = scores.astype(float)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def resize_image_keep_aspect(pil_image, max_size=1024):
    """
    Resizes a PIL Image keeping the aspect ratio such that the largest side is at most max_size.
    
    Args:
        pil_image (PIL.Image.Image): Input image
        max_size (int): Max dimension size
        
    Returns:
        PIL.Image.Image: Resized image
        tuple: Original dimensions (width, height)
        tuple: Scale factors (scale_x, scale_y) to convert back to original scale
    """
    orig_w, orig_h = pil_image.size
    
    if max(orig_w, orig_h) <= max_size:
        return pil_image, (orig_w, orig_h), (1.0, 1.0)
        
    if orig_w > orig_h:
        new_w = max_size
        new_h = int(orig_h * (max_size / orig_w))
    else:
        new_h = max_size
        new_w = int(orig_w * (max_size / orig_h))
        
    resized_img = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h
    
    return resized_img, (orig_w, orig_h), (scale_x, scale_y)

def map_box_to_original(box, scale_x, scale_y):
    """
    Maps a bounding box in resized coordinates back to the original image coordinates.
    
    Args:
        box (list/tuple/np.ndarray): [xmin, ymin, xmax, ymax] in resized coordinates
        scale_x (float): Horizontal scale factor
        scale_y (float): Vertical scale factor
        
    Returns:
        list: [xmin, ymin, xmax, ymax] in original coordinates
    """
    xmin, ymin, xmax, ymax = box
    return [
        xmin * scale_x,
        ymin * scale_y,
        xmax * scale_x,
        ymax * scale_y
    ]

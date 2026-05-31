import cv2
import numpy as np
from PIL import Image

class ShelfVisualizer:
    def __init__(self):
        # Color mapping in BGR format for parent brands
        self.brand_colors = {
            "Coca-Cola": (0, 0, 255),      # Red
            "Pepsi": (255, 0, 0),          # Blue
            "Lay's": (0, 255, 255),        # Yellow
            "Doritos": (0, 100, 255),      # Orange
            "Other": (0, 200, 0)           # Green
        }
        
    def annotate_image(self, pil_image, boxes, skus, brands, scores, row_metrics, clean_ocr_tags):
        """
        Draws visual annotations on the shelf image.
        
        Args:
            pil_image (PIL.Image.Image): Original high-resolution image
            boxes (list): Bounding boxes [xmin, ymin, xmax, ymax]
            skus (list of str): Product SKU names (e.g. "Sprite", "Lay's Classic")
            brands (list of str): Mapped parent brand names (e.g. "Coca-Cola", "Other")
            scores (list of float): Detection confidence score for each box
            row_metrics (dict): Output of ShelfSegmenter.segment_shelves
            clean_ocr_tags (list of dict): Cleaned price tags with bounding boxes and text
            
        Returns:
            PIL.Image.Image: Professionally annotated PIL Image
        """
        # Convert PIL Image to OpenCV BGR format
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = img_cv.shape
        
        # 1. Draw Shelf Row Separators (Horizontal Lines)
        rows = row_metrics.get('rows', [])
        for r_idx, row in enumerate(rows):
            ymin = int(row['ymin'])
            ymax = int(row['ymax'])
            
            # Draw row boundaries
            cv2.line(img_cv, (0, ymin), (w_img, ymin), (200, 200, 200), 2, cv2.LINE_AA)
            cv2.line(img_cv, (0, ymax), (w_img, ymax), (120, 120, 120), 2, cv2.LINE_AA)
            
            # Label Row index
            label_text = f"Row {r_idx + 1}"
            cv2.putText(
                img_cv, label_text, (20, ymin + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA
            )
            
        # 2. Draw Product Bounding Boxes & Specific SKU Labels
        for box, sku, brand, score in zip(boxes, skus, brands, scores):
            xmin, ymin, xmax, ymax = map(int, box)
            color = self.brand_colors.get(brand, self.brand_colors["Other"])
            
            # Draw bounding box
            cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Draw SKU text label and confidence only (e.g. "Sprite (0.85)")
            label_str = f"{sku} ({score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
            
            # Draw background box for text
            text_bg_ymin = max(0, ymin - text_h - 10)
            cv2.rectangle(
                img_cv, (xmin, text_bg_ymin), (xmin + text_w + 10, ymin),
                color, -1
            )
            
            # Write text in white
            cv2.putText(
                img_cv, label_str, (xmin + 5, ymin - 5),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
            
        # 3. Draw OCR Price Overlays ONLY on price tags (magenta color)
        # Bypasses any packaging text OCR coordinates
        ocr_color = (255, 0, 255) # Magenta
        for tag in clean_ocr_tags:
            box_pts = tag['box']
            price_text = tag['text']
            
            # Draw bounding box polygon around the tag region
            pts = np.array(box_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_cv, [pts], True, ocr_color, 2)
            
            # Draw text label background
            top_left = tuple(np.min(pts, axis=0)[0])
            tx, ty = int(top_left[0]), int(top_left[1])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), bl = cv2.getTextSize(price_text, font, font_scale, thickness)
            
            cv2.rectangle(img_cv, (tx, ty - th - 6), (tx + tw + 6, ty), ocr_color, -1)
            cv2.putText(
                img_cv, price_text, (tx + 3, ty - 3),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
            
        # Convert annotated OpenCV BGR image back to RGB PIL format
        annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return annotated_pil

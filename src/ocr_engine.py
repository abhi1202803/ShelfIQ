import easyocr
import numpy as np
import re
import cv2
from PIL import Image

class OCREngine:
    def __init__(self, languages=['en'], gpu=True):
        """
        Initializes the EasyOCR reader.
        
        Args:
            languages (list): List of language codes
            gpu (bool): Whether to use GPU if available
        """
        print(f"Initializing EasyOCR Engine (GPU={gpu})...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
        # Whitelisted prices from the ParallelDots guidelines
        self.allowed_prices = {20, 25, 28, 30, 35, 40, 45, 50, 55, 60, 62, 70, 75, 85, 99, 105, 120, 125, 130, 140}
        
    def extract_price_strips_text(self, pil_image, rows, confidence_threshold=0.3):
        """
        Detects yellow tags inside row bands, crops them, and runs OCR on them.
        Falls back to reading the entire row strip if no yellow tag contours are found.
        
        Args:
            pil_image (PIL.Image.Image): Original high-resolution image
            rows (list of dict): Segmented shelf rows with vertical bounds
            confidence_threshold (float): Discard results with confidence below this
            
        Returns:
            list of dict: Mapped OCR results containing 'box', 'text', 'confidence', 'row_index'
        """
        orig_w, orig_h = pil_image.size
        img_np = np.array(pil_image.convert('RGB'))
        # Convert to BGR format for OpenCV color conversion
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        all_results = []
        
        for row in rows:
            ymin_row = row['ymin']
            ymax_row = row['ymax']
            row_height = ymax_row - ymin_row
            row_idx = row['row_index']
            
            # Define price strip search band directly below the product ymax boundary
            ymin_band = int(max(0, ymax_row - 0.10 * row_height))
            ymax_band = int(min(orig_h, ymax_row + 0.25 * row_height))
            
            # Crop search band
            band_crop = img_bgr[ymin_band:ymax_band, :, :]
            
            if band_crop.size == 0:
                continue
                
            # Convert band to HSV and threshold for yellow tags
            hsv = cv2.cvtColor(band_crop, cv2.COLOR_BGR2HSV)
            # HSV Yellow boundaries: Hue [10-45], Saturation [30-255], Value [80-255]
            mask = cv2.inRange(hsv, (10, 30, 80), (45, 255, 255))
            
            # Find contours of yellow price tags
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            yellow_tags = []
            for c in contours:
                x_tag, y_tag, w_tag, h_tag = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                
                # Check tag dimensions (in crop space) to filter noise
                if area > 150 and w_tag > 25 and h_tag > 10:
                    yellow_tags.append((x_tag, y_tag, w_tag, h_tag))
                    
            if len(yellow_tags) == 0:
                # Fallback: Run OCR on the entire strip band directly
                ocr_results = self.reader.readtext(band_crop)
                for bbox, text, confidence in ocr_results:
                    confidence = float(confidence)
                    if confidence < confidence_threshold:
                        continue
                        
                    # Map bbox points back to the original image coordinate system
                    mapped_box = []
                    for pt in bbox:
                        mapped_box.append([int(pt[0]), int(pt[1]) + ymin_band])
                        
                    all_results.append({
                        'box': mapped_box,
                        'text': text.strip(),
                        'confidence': confidence,
                        'row_index': row_idx
                    })
            else:
                # OCR on yellow tag crops inside the band
                for (x_tag, y_tag, w_tag, h_tag) in yellow_tags:
                    label_crop = band_crop[y_tag:y_tag+h_tag, x_tag:x_tag+w_tag]
                    if label_crop.size == 0:
                        continue
                        
                    ocr_results = self.reader.readtext(label_crop)
                    for bbox, text, confidence in ocr_results:
                        confidence = float(confidence)
                        if confidence < confidence_threshold:
                            continue
                            
                        # Map bbox back to original image
                        mapped_box = []
                        for pt in bbox:
                            mapped_box.append([int(pt[0]) + x_tag, int(pt[1]) + y_tag + ymin_band])
                            
                        all_results.append({
                            'box': mapped_box,
                            'text': text.strip(),
                            'confidence': confidence,
                            'row_index': row_idx
                        })
                        
        return all_results

    def filter_price_tags(self, ocr_results):
        """
        Cleans OCR text using regex and strictly filters against the target price list.
        Rejects noisy text (e.g. ₹0372, $9, weight labels).
        
        Args:
            ocr_results (list of dict): Output of extract_price_strips_text
            
        Returns:
            list of str: Normalized pricing labels (e.g. ["₹20", "₹99"])
        """
        cleaned_labels = []
        seen_prices = set()
        
        # Regex to extract any numeric digits from text
        num_pattern = re.compile(r'\d+')
        
        for res in ocr_results:
            text = res['text']
            
            # Find all numbers in the string
            numbers = num_pattern.findall(text)
            if not numbers:
                continue
                
            for num_str in numbers:
                try:
                    num_val = int(num_str)
                    if num_val in self.allowed_prices:
                        formatted_price = f"₹{num_val}"
                        if formatted_price not in seen_prices:
                            cleaned_labels.append(formatted_price)
                            seen_prices.add(formatted_price)
                except ValueError:
                    continue
                    
        return cleaned_labels

    def get_cleaned_price_tags_with_coords(self, ocr_results):
        """
        Filters price tags and returns their coordinates. Used for spatial association.
        
        Args:
            ocr_results (list of dict): Output of extract_price_strips_text
            
        Returns:
            list of dict: List of tags with 'box', 'text', 'confidence', 'row_index'
        """
        cleaned_tags = []
        seen_positions = set()
        
        num_pattern = re.compile(r'\d+')
        for res in ocr_results:
            text = res['text']
            box = res['box']
            conf = res['confidence']
            
            numbers = num_pattern.findall(text)
            if not numbers:
                continue
                
            for num_str in numbers:
                try:
                    num_val = int(num_str)
                    if num_val in self.allowed_prices:
                        formatted_price = f"₹{num_val}"
                        # Check duplicate tags at the same location to prevent duplicate associations
                        pos_key = (box[0][0], box[0][1])
                        if pos_key not in seen_positions:
                            cleaned_tags.append({
                                'box': box,
                                'text': formatted_price,
                                'confidence': conf,
                                'row_index': res.get('row_index', -1)
                            })
                            seen_positions.add(pos_key)
                        break
                except ValueError:
                    continue
        return cleaned_tags

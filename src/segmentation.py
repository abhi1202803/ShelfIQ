import numpy as np
from sklearn.cluster import DBSCAN

class ShelfSegmenter:
    def __init__(self, eps_ratio=0.05, min_samples=1):
        """
        Initializes the Shelf Segmenter.
        
        Args:
            eps_ratio (float): Ratio of image height to use as clustering epsilon
            min_samples (int): Min samples parameter for DBSCAN
        """
        self.eps_ratio = eps_ratio
        self.min_samples = min_samples

    def segment_shelves(self, boxes, skus, brands, image_height, image_width, ocr_tags=None, eps=None):
        """
        Segments shelf rows, calculates SOS, availability, business insights, and associates price tags.
        
        Args:
            boxes (list): Bounding boxes in original coordinates [xmin, ymin, xmax, ymax]
            skus (list of str): Product SKU names for each box
            brands (list of str): Mapped parent brand name for each box
            image_height (int): Height of original image
            image_width (int): Width of original image
            ocr_tags (list of dict, optional): Cleaned price tags with bounding boxes and text
            eps (float, optional): Epsilon distance for DBSCAN (overrides eps_ratio)
            
        Returns:
            dict: Comprehensive segmentation metrics containing:
                - 'rows': list of dicts representing each row
                - 'overall_share_of_shelf': dict of brand SOS percentages
                - 'box_row_assignments': list of row indices for each input box
                - 'availability': brand availability status
                - 'insights': business insights block
                - 'product_price_associations': list of mapped products and prices
        """
        if len(boxes) == 0:
            return {
                'rows': [],
                'overall_share_of_shelf': {},
                'box_row_assignments': [],
                'availability': {
                    "Coca-Cola": "Not Present",
                    "Pepsi": "Not Present",
                    "Lay's": "Not Present",
                    "Doritos": "Not Present"
                },
                'insights': {
                    "dominant_brand": "None",
                    "highest_share_of_shelf": "None",
                    "brands_present": []
                },
                'product_price_associations': []
            }
            
        boxes_arr = np.array(boxes)
        # Calculate vertical centers (yc) of each bounding box
        y_centers = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        y_centers_2d = y_centers.reshape(-1, 1)
        
        # Set clustering epsilon dynamically if not provided
        if eps is None:
            eps = image_height * self.eps_ratio
            
        # 1. Cluster y-centers using DBSCAN
        db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(y_centers_2d)
        labels = db.labels_
        unique_labels = np.unique(labels)
        
        # Calculate average y-center for each cluster to sort vertically (top to bottom)
        cluster_y_averages = []
        for lbl in unique_labels:
            indices = np.where(labels == lbl)[0]
            avg_y = np.mean(y_centers[indices])
            cluster_y_averages.append((lbl, avg_y))
            
        cluster_y_averages.sort(key=lambda x: x[1])
        sorted_labels = [lbl for lbl, _ in cluster_y_averages]
        
        # Map old labels to sorted row indices
        label_to_row = {old_lbl: row_idx for row_idx, old_lbl in enumerate(sorted_labels)}
        row_assignments = [label_to_row[lbl] for lbl in labels]
        
        # 2. Process each shelf row
        rows_data = []
        brand_widths = {}
        total_width_sum = 0.0
        
        brand_counts = {b: 0 for b in ["Coca-Cola", "Pepsi", "Lay's", "Doritos", "Other"]}
        for b in brands:
            brand_counts[b] = brand_counts.get(b, 0) + 1
            
        for row_idx in range(len(sorted_labels)):
            item_indices = [idx for idx, r_idx in enumerate(row_assignments) if r_idx == row_idx]
            item_indices.sort(key=lambda idx: boxes[idx][0])
            
            row_boxes = boxes_arr[item_indices]
            row_ymin = float(np.min(row_boxes[:, 1]))
            row_ymax = float(np.max(row_boxes[:, 3]))
            
            row_widths = row_boxes[:, 2] - row_boxes[:, 0]
            row_total_product_width = float(np.sum(row_widths))
            
            row_brand_counts = {}
            row_brand_widths = {}
            
            for idx in item_indices:
                brand = brands[idx]
                width = boxes[idx][2] - boxes[idx][0]
                
                row_brand_counts[brand] = row_brand_counts.get(brand, 0) + 1
                row_brand_widths[brand] = row_brand_widths.get(brand, 0.0) + float(width)
                brand_widths[brand] = brand_widths.get(brand, 0.0) + float(width)
                total_width_sum += float(width)
                
            row_sos = {}
            for brand, width in row_brand_widths.items():
                row_sos[brand] = f"{round((width / row_total_product_width) * 100)}%" if row_total_product_width > 0 else "0%"
                
            rows_data.append({
                'row_index': row_idx,
                'ymin': row_ymin,
                'ymax': row_ymax,
                'product_indices': item_indices,
                'total_products': len(item_indices),
                'brand_counts': row_brand_counts,
                'share_of_shelf': row_sos
            })
            
        # 3. Overall Share of Shelf (SOS)
        overall_sos = {}
        for brand, width in brand_widths.items():
            if total_width_sum > 0:
                overall_sos[brand] = f"{round((width / total_width_sum) * 100)}%"
            else:
                overall_sos[brand] = "0%"
                
        target_brands = ["Coca-Cola", "Pepsi", "Lay's", "Doritos", "Other"]
        for brand in target_brands:
            if brand not in overall_sos:
                overall_sos[brand] = "0%"
                
        # 4. Brand Availability Insights
        availability = {}
        for brand in ["Coca-Cola", "Pepsi", "Lay's", "Doritos"]:
            availability[brand] = "Available" if brand_counts.get(brand, 0) > 0 else "Not Present"
            
        # 5. Business Insights Block
        # Find dominant brand by count (excluding "Other" or including it? Let's check among all brands)
        dominant_brand = "Other"
        max_count = -1
        for b in target_brands:
            if brand_counts.get(b, 0) > max_count:
                max_count = brand_counts[b]
                dominant_brand = b
                
        # Find highest share of shelf
        highest_sos_brand = "Other"
        max_sos_pct = -1
        for b, pct_str in overall_sos.items():
            try:
                pct_val = int(pct_str.replace('%', ''))
                if pct_val > max_sos_pct:
                    max_sos_pct = pct_val
                    highest_sos_brand = b
            except ValueError:
                continue
                
        # Compile brands present
        brands_present = [b for b in ["Coca-Cola", "Pepsi", "Lay's", "Doritos"] if brand_counts.get(b, 0) > 0]
        
        insights = {
            "dominant_brand": dominant_brand,
            "highest_share_of_shelf": highest_sos_brand,
            "brands_present": brands_present
        }
        
        # 6. Product-Price Association Heuristic
        # Associate nearest shelf tag below product in the same row
        product_price_associations = []
        
        if ocr_tags is not None and len(ocr_tags) > 0:
            for tag in ocr_tags:
                tag_row = tag.get('row_index', -1)
                tag_box = np.array(tag['box'])
                tag_xmin = np.min(tag_box[:, 0])
                tag_ymin = np.min(tag_box[:, 1])
                tag_xmax = np.max(tag_box[:, 0])
                tag_ymax = np.max(tag_box[:, 1])
                
                tag_xc = (tag_xmin + tag_xmax) / 2.0
                price_text = tag['text']
                
                # Find all products above this tag in the same shelf row
                above_indices = []
                for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                    prod_row = row_assignments[idx]
                    if prod_row == tag_row and ymax <= tag_ymin + 30:
                        above_indices.append(idx)
                        
                # Fallback: search all products above tag if row-restricted matching is empty
                if len(above_indices) == 0:
                    for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                        if ymax <= tag_ymin + 30:
                            above_indices.append(idx)
                            
                if len(above_indices) == 0:
                    continue
                    
                # From those products above, find the one closest horizontally to the tag center
                best_idx = -1
                min_h_dist = float('inf')
                
                for idx in above_indices:
                    xmin, ymin, xmax, ymax = boxes[idx]
                    prod_xc = (xmin + xmax) / 2.0
                    h_dist = abs(prod_xc - tag_xc)
                    
                    # Ensure proximity limit (products shouldn't be horizontally too far from the price tag)
                    # Limit to 1.5 times the product width or a max of 150 pixels
                    prod_width = xmax - xmin
                    proximity_limit = max(150, prod_width * 1.5)
                    
                    if h_dist < min_h_dist and h_dist <= proximity_limit:
                        min_h_dist = h_dist
                        best_idx = idx
                        
                if best_idx != -1:
                    product_price_associations.append({
                        "product": skus[best_idx],
                        "price": price_text
                    })
            
            # Deduplicate product-price associations
            unique_associations = []
            seen_assoc = set()
            for assoc in product_price_associations:
                assoc_key = (assoc['product'], assoc['price'])
                if assoc_key not in seen_assoc:
                    seen_assoc.add(assoc_key)
                    unique_associations.append(assoc)
            product_price_associations = unique_associations
                    
        return {
            'rows': rows_data,
            'overall_share_of_shelf': overall_sos,
            'box_row_assignments': row_assignments,
            'availability': availability,
            'insights': insights,
            'product_price_associations': product_price_associations
        }

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class BrandClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initializes the CLIP Brand Classifier.
        
        Args:
            model_name (str): Hugging Face CLIP model repository
            device (str): Device to run inference on
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing CLIP Brand Classifier on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Comprehensive product vocabulary with descriptive prompt engineering
        self.sku_prompts = {
            # Beverages
            "Tropicana Orange Delight": "a supermarket shelf product image of Tropicana Orange Delight juice carton",
            "Tropicana Mixed Fruit": "a supermarket shelf product image of Tropicana Mixed Fruit juice carton",
            "Tropicana Apple Delight": "a supermarket shelf product image of Tropicana Apple Delight juice carton",
            "Tropicana Pineapple Delight": "a supermarket shelf product image of Tropicana Pineapple Delight juice carton",
            "Tropicana Guava Delight": "a supermarket shelf product image of Tropicana Guava Delight juice carton",
            "Tropicana Pomegranate Delight": "a supermarket shelf product image of Tropicana Pomegranate Delight juice carton",
            "Real Orange Juice": "a supermarket shelf product image of Real Orange Juice carton",
            "Real Mixed Fruit": "a supermarket shelf product image of Real Mixed Fruit juice carton",
            "Real Guava": "a supermarket shelf product image of Real Guava juice carton",
            "Real Pomegranate": "a supermarket shelf product image of Real Pomegranate juice carton",
            "Minute Maid Pulpy Orange": "a supermarket shelf product image of Minute Maid Pulpy Orange bottle",
            "Minute Maid Apple": "a supermarket shelf product image of Minute Maid Apple juice bottle",
            "Minute Maid Mixed Fruit": "a supermarket shelf product image of Minute Maid Mixed Fruit juice bottle",
            "Sprite": "a supermarket shelf product image of Sprite bottle or can",
            "Coca Cola": "a supermarket shelf product image of Coca Cola bottle or can",
            "Diet Coke": "a supermarket shelf product image of Diet Coke bottle or can",
            "Thums Up": "a supermarket shelf product image of Thums Up bottle or can",
            "Fanta": "a supermarket shelf product image of Fanta bottle or can",
            "Limca": "a supermarket shelf product image of Limca bottle or can",
            "Mountain Dew": "a supermarket shelf product image of Mountain Dew bottle or can",
            "7UP": "a supermarket shelf product image of 7UP bottle or can",
            "Mirinda": "a supermarket shelf product image of Mirinda bottle or can",
            "Pepsi": "a supermarket shelf product image of Pepsi bottle or can",
            "Nestea": "a supermarket shelf product image of Nestea bottle or package",
            "Lipton Ice Tea": "a supermarket shelf product image of Lipton Ice Tea bottle",
            "B Natural Mixed Fruit": "a supermarket shelf product image of B Natural Mixed Fruit juice bottle or carton",
            "B Natural Orange": "a supermarket shelf product image of B Natural Orange juice bottle or carton",
            "B Natural Pomegranate": "a supermarket shelf product image of B Natural Pomegranate juice bottle or carton",
            "Paper Boat Aam Panna": "a supermarket shelf product image of Paper Boat Aam Panna juice packet",
            "Paper Boat Jamun": "a supermarket shelf product image of Paper Boat Jamun juice packet",
            "Gatorade": "a supermarket shelf product image of Gatorade bottle",
            "Red Bull": "a supermarket shelf product image of Red Bull can",
            "Nescafe": "a supermarket shelf product image of Nescafe bottle or can",
            "Amul Kool": "a supermarket shelf product image of Amul Kool bottle",

            # Snacks & Biscuits
            "Lay's Classic": "a supermarket shelf product image of yellow Lay's Classic chips bag",
            "Lay's Magic Masala": "a supermarket shelf product image of blue Lay's Magic Masala chips bag",
            "Lay's Chile Limon": "a supermarket shelf product image of green Lay's Chile Limon chips bag",
            "Lay's Spanish Tomato Tango": "a supermarket shelf product image of red Lay's Spanish Tomato Tango chips bag",
            "Doritos Nacho Cheese": "a supermarket shelf product image of yellow-orange Doritos Nacho Cheese chips bag",
            "Doritos Cool Ranch": "a supermarket shelf product image of blue Doritos Cool Ranch chips bag",
            "Doritos Sweet Chili": "a supermarket shelf product image of red-black Doritos Sweet Chili chips bag",
            "Cheetos Crunchy": "a supermarket shelf product image of Cheetos Crunchy bag",
            "Cheetos Green": "a supermarket shelf product image of Cheetos green bag",
            "Kurkure": "a supermarket shelf product image of Kurkure snack bag",
            "Uncle Chips Plain Salted": "a supermarket shelf product image of Uncle Chips Plain Salted bag",
            "Uncle Chips Spicy Treat": "a supermarket shelf product image of Uncle Chips Spicy Treat bag",
            "Bingo Original": "a supermarket shelf product image of Bingo Original chips bag",
            "Bingo Cream and Onion": "a supermarket shelf product image of Bingo Cream and Onion chips bag",
            "Bingo Mad Angles": "a supermarket shelf product image of Bingo Mad Angles triangle chips bag",
            "Pringles Original": "a supermarket shelf product image of red Pringles Original canister can",
            "Pringles Sour Cream": "a supermarket shelf product image of green Pringles Sour Cream canister can",
            "Pringles Texas BBQ": "a supermarket shelf product image of purple Pringles Texas BBQ canister can",
            "Monaco": "a supermarket shelf product image of Monaco biscuit packet",
            "Parle G": "a supermarket shelf product image of Parle G biscuit packet",
            "Good Day Cashew": "a supermarket shelf product image of Good Day Cashew biscuit packet",
            "Good Day Butter": "a supermarket shelf product image of Good Day Butter biscuit packet",
            "Good Day Pista Almond": "a supermarket shelf product image of Britannia Good Day Pista Almond biscuit packet",
            "Marie Gold": "a supermarket shelf product image of Marie Gold biscuit packet",
            "Oreo Original": "a supermarket shelf product image of blue Oreo Original biscuit packet",
            "Oreo Choco Creme": "a supermarket shelf product image of Oreo Choco Creme biscuit packet",
            "Hide and Seek Fab": "a supermarket shelf product image of Hide and Seek Fab biscuit packet",
            "Hide and Seek Milano": "a supermarket shelf product image of Hide and Seek Milano biscuit packet",
            "Dark Fantasy": "a supermarket shelf product image of Dark Fantasy biscuit packet",
            "Malkist Cheese": "a supermarket shelf product image of Malkist Cheese biscuit packet",
            "Malkist Masala": "a supermarket shelf product image of Malkist Masala biscuit packet",
            "Tiger Krunch": "a supermarket shelf product image of Tiger Krunch biscuit packet",

            # Dairy
            "Nestle A+ Yogurt": "a supermarket shelf product image of Nestle A+ Yogurt cup",
            "Activia Yogurt": "a supermarket shelf product image of Activia Yogurt cup",
            "Actimel": "a supermarket shelf product image of Actimel bottle",
            "Yakult": "a supermarket shelf product image of Yakult bottle",
            "Amul Taaza Milk": "a supermarket shelf product image of Amul Taaza Milk carton or packet",
            "Mother Dairy Toned Milk": "a supermarket shelf product image of Mother Dairy Toned Milk packet or bottle",
            "Amul Shakti Milk": "a supermarket shelf product image of Amul Shakti Milk carton or packet",
            "Amul Gold Milk": "a supermarket shelf product image of Amul Gold Milk carton or packet",
            "Epigamia Milkshake": "a supermarket shelf product image of Epigamia Milkshake milkshake bottle",
            "Hersheys Milkshake": "a supermarket shelf product image of Hersheys Milkshake chocolate bottle",
            "Amul Royale Strawberry": "a supermarket shelf product image of Amul Royale Strawberry yogurt cup",
            "Amul Royale Mango": "a supermarket shelf product image of Amul Royale Mango yogurt cup",
            "Amul Pro Chocolate": "a supermarket shelf product image of Amul Pro Chocolate malt beverage container",
            "Amul Pro Vanilla": "a supermarket shelf product image of Amul Pro Vanilla malt beverage container",
            "Amul Butter": "a supermarket shelf product image of Amul Butter box",
            "Amul Cheese": "a supermarket shelf product image of Amul Cheese block or packet",
            "Amul Cheese Spread": "a supermarket shelf product image of Amul Cheese Spread tub",
            "Go Cheese Slices": "a supermarket shelf product image of Go Cheese Slices packet",
            "Milky Mist Butter": "a supermarket shelf product image of Milky Mist Butter block or packaging",
            "Milky Mist Mozzarella": "a supermarket shelf product image of Milky Mist Mozzarella cheese packet",
            "Milky Mist Pizza Cheese": "a supermarket shelf product image of Milky Mist Pizza Cheese block or packet",
            "Amul Pizza Cheese": "a supermarket shelf product image of Amul Pizza Cheese block or packet",

            # Other Category Fallback
            "Other": "a supermarket shelf product image of another retail packaging"
        }
        
        # Lists for matching
        self.sku_names = list(self.sku_prompts.keys())
        self.brand_prompts = list(self.sku_prompts.values())
        
        # Product SKU-to-brand mapping dict
        self.sku_to_brand = {
            # Coca-Cola owned/associated
            "Sprite": "Coca-Cola",
            "Coca Cola": "Coca-Cola",
            "Diet Coke": "Coca-Cola",
            "Thums Up": "Coca-Cola",
            "Fanta": "Coca-Cola",
            "Limca": "Coca-Cola",
            "Minute Maid Pulpy Orange": "Coca-Cola",
            "Minute Maid Apple": "Coca-Cola",
            "Minute Maid Mixed Fruit": "Coca-Cola",

            # PepsiCo owned/associated
            "Pepsi": "Pepsi",
            "7UP": "Pepsi",
            "Mountain Dew": "Pepsi",
            "Mirinda": "Pepsi",
            "Tropicana Orange Delight": "Pepsi",
            "Tropicana Mixed Fruit": "Pepsi",
            "Tropicana Apple Delight": "Pepsi",
            "Tropicana Pineapple Delight": "Pepsi",
            "Tropicana Guava Delight": "Pepsi",
            "Tropicana Pomegranate Delight": "Pepsi",
            "Gatorade": "Pepsi",
            "Cheetos Crunchy": "Pepsi",
            "Cheetos Green": "Pepsi",
            "Kurkure": "Pepsi",

            # Lay's (PepsiCo)
            "Lay's Classic": "Lay's",
            "Lay's Magic Masala": "Lay's",
            "Lay's Chile Limon": "Lay's",
            "Lay's Spanish Tomato Tango": "Lay's",

            # Doritos (PepsiCo)
            "Doritos Nacho Cheese": "Doritos",
            "Doritos Cool Ranch": "Doritos",
            "Doritos Sweet Chili": "Doritos"
        }
        
    def classify_crops(self, pil_image, boxes, batch_size=32, threshold=0.4):
        """
        Crops products from image and classifies them in batches using zero-shot CLIP.
        
        Args:
            pil_image (PIL.Image.Image): Original high-resolution image
            boxes (list): Bounding boxes in original coordinates [xmin, ymin, xmax, ymax]
            batch_size (int): Batch size for CLIP inference
            threshold (float): Confidence threshold. If max probability < threshold, returns "Other"
            
        Returns:
            list of str: Predicted product SKU for each bounding box
            list of str: Mapped brand name for each bounding box
        """
        if len(boxes) == 0:
            return [], []
            
        # 1. Extract crop images
        crops = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            # Ensure crop has positive size and boundaries are respected
            w, h = pil_image.size
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(xmin + 1, min(xmax, w))
            ymax = max(ymin + 1, min(ymax, h))
            
            crop = pil_image.crop((xmin, ymin, xmax, ymax))
            crops.append(crop)
            
        sku_predictions = []
        brand_predictions = []
        
        # 2. Tokenize prompt texts (same for all batches)
        text_inputs = self.processor(
            text=self.brand_prompts,
            return_tensors="pt",
            padding=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            # Normalize text features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        # 3. Process crops in batches
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            
            # Prepare image inputs
            inputs = self.processor(
                images=batch_crops,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get image features
                image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                if hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                # Normalize image features
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                logit_scale = self.model.logit_scale.exp()
                logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale
                
                # Calculate probabilities via Softmax
                probs = torch.softmax(logits_per_image, dim=-1).cpu().numpy()
                
            # Parse probabilities
            for prob_dist in probs:
                max_idx = np.argmax(prob_dist)
                max_prob = prob_dist[max_idx]
                
                # Check confidence threshold
                if max_prob < threshold:
                    predicted_sku = "Other"
                else:
                    predicted_sku = self.sku_names[max_idx]
                    
                # Map SKU to brand
                predicted_brand = self.sku_to_brand.get(predicted_sku, "Other")
                
                sku_predictions.append(predicted_sku)
                brand_predictions.append(predicted_brand)
                
        return sku_predictions, brand_predictions

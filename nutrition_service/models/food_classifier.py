"""
SMARTCARE+ Nutrition Service - Food Image Classifier

Owner: Kulasekara
CNN-based food recognition using MobileNetV2 pre-trained on Food-101 dataset.
Identifies foods from camera/gallery images and maps to nutrition database.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import asyncio
import logging

# Setup logger for food classifier
logger = logging.getLogger("smartcare.nutrition.classifier")
logger.setLevel(logging.DEBUG)

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Using mock food classification.")

# PIL for image processing
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available.")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOD-101 CLASS LABELS
# ═══════════════════════════════════════════════════════════════════════════════

# Standard Food-101 dataset classes (101 food categories)
FOOD_101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

# Mapping Food-101 classes to our local food database categories
FOOD_101_TO_LOCAL_MAPPING = {
    "chicken_curry": ["chicken_curry", "white_rice", "dhal_curry"],
    "fried_rice": ["fried_rice", "vegetable_fried_rice"],
    "rice": ["white_rice", "red_rice", "basmati_rice"],
    "dumplings": ["dumplings", "chinese_dumplings"],
    "spring_rolls": ["spring_rolls", "vegetable_rolls"],
    "salad": ["garden_salad", "gotu_kola_sambol"],
    "fish_and_chips": ["fish_curry", "fried_fish"],
    "grilled_salmon": ["grilled_fish", "fish_ambul_thiyal"],
    "omelette": ["egg_omelette", "plain_omelette"],
    "pancakes": ["hopper", "string_hopper", "pancakes"],
    "bread": ["bread", "roti", "pol_roti"],
    "soup": ["dhal_curry", "vegetable_soup"],
    "steak": ["beef_curry", "grilled_beef"],
    "pizza": ["pizza"],
    "hamburger": ["burger", "chicken_burger"],
    "hot_dog": ["sausage", "hot_dog"],
    "sushi": ["sushi", "sashimi"],
    "ice_cream": ["ice_cream", "watalappan"],
    "fruit": ["fruit_salad", "papaya", "banana", "mango"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGENET FOOD CLASSES MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
# ImageNet has 1000 classes, ~100 are food-related. Map them to display names.

IMAGENET_FOOD_CLASSES = {
    # Fruits
    "banana": {"display": "Banana", "foods": ["banana", "fruit"]},
    "pineapple": {"display": "Pineapple", "foods": ["pineapple", "fruit"]},
    "strawberry": {"display": "Strawberry", "foods": ["strawberry", "fruit_salad"]},
    "orange": {"display": "Orange", "foods": ["orange", "fruit"]},
    "lemon": {"display": "Lemon", "foods": ["lemon"]},
    "fig": {"display": "Fig", "foods": ["fig", "fruit"]},
    "pomegranate": {"display": "Pomegranate", "foods": ["pomegranate", "fruit"]},
    "jackfruit": {"display": "Jackfruit", "foods": ["jackfruit"]},
    "custard_apple": {"display": "Custard Apple", "foods": ["custard_apple", "fruit"]},
    "Granny_Smith": {"display": "Apple", "foods": ["apple", "fruit"]},
    "watermelon": {"display": "Watermelon", "foods": ["watermelon", "fruit"]},
    "mango": {"display": "Mango", "foods": ["mango", "fruit"]},
    "papaya": {"display": "Papaya", "foods": ["papaya", "fruit"]},
    "coconut": {"display": "Coconut", "foods": ["coconut", "fruit"]},
    "grape": {"display": "Grape", "foods": ["grape", "fruit"]},
    
    # Vegetables
    "head_cabbage": {"display": "Cabbage", "foods": ["cabbage", "vegetable"]},
    "broccoli": {"display": "Broccoli", "foods": ["broccoli", "vegetable"]},
    "cauliflower": {"display": "Cauliflower", "foods": ["cauliflower", "vegetable"]},
    "zucchini": {"display": "Zucchini", "foods": ["zucchini", "vegetable"]},
    "cucumber": {"display": "Cucumber", "foods": ["cucumber", "vegetable", "salad"]},
    "artichoke": {"display": "Artichoke", "foods": ["artichoke", "vegetable"]},
    "bell_pepper": {"display": "Bell Pepper", "foods": ["bell_pepper", "vegetable"]},
    "mushroom": {"display": "Mushroom", "foods": ["mushroom", "vegetable"]},
    "acorn_squash": {"display": "Squash", "foods": ["squash", "vegetable"]},
    "butternut_squash": {"display": "Butternut Squash", "foods": ["squash", "vegetable"]},
    "spaghetti_squash": {"display": "Spaghetti Squash", "foods": ["squash", "vegetable"]},
    
    # Prepared Foods & Dishes
    "pizza": {"display": "Pizza", "foods": ["pizza"]},
    "cheeseburger": {"display": "Cheeseburger", "foods": ["burger", "hamburger"]},
    "hotdog": {"display": "Hot Dog", "foods": ["hot_dog", "sausage"]},
    "French_loaf": {"display": "French Bread", "foods": ["bread", "french_bread"]},
    "bagel": {"display": "Bagel", "foods": ["bagel", "bread"]},
    "pretzel": {"display": "Pretzel", "foods": ["pretzel", "snack"]},
    "meat_loaf": {"display": "Meat Loaf", "foods": ["meatloaf", "beef"]},
    "burrito": {"display": "Burrito", "foods": ["burrito", "mexican"]},
    "potpie": {"display": "Pot Pie", "foods": ["pot_pie", "pie"]},
    "espresso": {"display": "Espresso Coffee", "foods": ["coffee", "espresso"]},
    "consomme": {"display": "Consommé Soup", "foods": ["soup", "broth"]},
    "trifle": {"display": "Trifle Dessert", "foods": ["dessert", "cake"]},
    "ice_cream": {"display": "Ice Cream", "foods": ["ice_cream", "dessert"]},
    "ice_lolly": {"display": "Ice Lolly/Popsicle", "foods": ["popsicle", "ice_cream"]},
    "carbonara": {"display": "Pasta Carbonara", "foods": ["pasta", "spaghetti_carbonara"]},
    "guacamole": {"display": "Guacamole", "foods": ["guacamole", "dip"]},
    
    # Baked Goods & Desserts
    "chocolate_sauce": {"display": "Chocolate Sauce", "foods": ["chocolate", "dessert"]},
    "dough": {"display": "Dough", "foods": ["bread", "dough"]},
    "plate": {"display": "Food Plate", "foods": ["meal"]},
    "menu": {"display": "Menu (Food)", "foods": ["meal"]},
    "eggnog": {"display": "Eggnog", "foods": ["eggnog", "drink"]},
    
    # Seafood
    "crayfish": {"display": "Crayfish", "foods": ["crayfish", "seafood"]},
    "lobster": {"display": "Lobster", "foods": ["lobster", "seafood"]},
    "American_lobster": {"display": "American Lobster", "foods": ["lobster", "seafood"]},
    "spiny_lobster": {"display": "Spiny Lobster", "foods": ["lobster", "seafood"]},
    "rock_crab": {"display": "Crab", "foods": ["crab", "seafood"]},
    "Dungeness_crab": {"display": "Dungeness Crab", "foods": ["crab", "seafood"]},
    "king_crab": {"display": "King Crab", "foods": ["king_crab", "seafood"]},
    "flatworm": {"display": "Seafood", "foods": ["seafood"]},
    
    # Meat
    "meat_loaf": {"display": "Meat Loaf", "foods": ["meatloaf", "beef"]},
    "pork_chop": {"display": "Pork Chop", "foods": ["pork", "pork_chop"]},
    
    # South Asian Foods (Sri Lankan, Indian)
    "curry": {"display": "Curry", "foods": ["curry", "dhal_curry", "chicken_curry"]},
    "dhal": {"display": "Dhal Curry", "foods": ["dhal_curry", "parippu"]},
    "lentil": {"display": "Dhal/Lentils", "foods": ["dhal_curry", "parippu"]},
    "naan": {"display": "Naan Bread", "foods": ["naan", "roti"]},
    "roti": {"display": "Roti", "foods": ["roti", "pol_roti"]},
    "samosa": {"display": "Samosa", "foods": ["samosa", "short_eats"]},
    "biryani": {"display": "Biryani", "foods": ["biryani", "fried_rice"]},
    "korma": {"display": "Korma Curry", "foods": ["curry", "chicken_curry"]},
    
    # Common items that appear in food photos (kitchen items - marked as indirect food indicators)
    "dining_table": {"display": "Meal on Table", "foods": ["meal"]},
    "soup_bowl": {"display": "Soup/Curry", "foods": ["soup", "dhal_curry", "curry"]},
    "mixing_bowl": {"display": "Food Bowl", "foods": ["meal", "curry"]},
    "ladle": {"display": "Soup/Curry", "foods": ["soup", "curry", "dhal_curry"]},
    "wok": {"display": "Stir Fry", "foods": ["stir_fry", "fried_rice"]},
    "frying_pan": {"display": "Cooked Dish", "foods": ["fried_food", "curry"]},
    "spatula": {"display": "Cooked Food", "foods": ["meal"]},
    "plate": {"display": "Food Plate", "foods": ["meal"]},
    "bowl": {"display": "Food Bowl", "foods": ["soup", "curry", "dhal_curry"]},
}


@dataclass
class FoodPrediction:
    """Single food prediction result."""
    class_name: str
    display_name: str
    confidence: float
    local_food_ids: List[str]  # Mapped to local database
    is_food: bool = True  # Whether this is a food-related class
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 3),
            "local_food_ids": self.local_food_ids,
            "is_food": self.is_food
        }


@dataclass
class ClassificationResult:
    """Complete food classification result."""
    success: bool
    predictions: List[FoodPrediction]
    image_size: Tuple[int, int]
    processing_time_ms: float
    model_version: str = "mobilenetv2_food101"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "predictions": [p.to_dict() for p in self.predictions],
            "image_size": list(self.image_size),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_version": self.model_version,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FOOD CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class FoodClassifier:
    """
    MobileNetV2-based food image classifier.
    
    Uses transfer learning from ImageNet with fine-tuning on Food-101 dataset.
    For demo purposes, uses ImageNet pretrained model with food-related class mapping.
    
    Features:
    - Classify food images from camera or gallery
    - Return top-k predictions with confidence scores
    - Map predictions to local food database
    - Support for batch classification
    """
    
    # Image input size for MobileNetV2
    INPUT_SIZE = (224, 224)
    
    def __init__(self, model_path: Optional[str] = None, use_food101: bool = True):
        """
        Initialize the food classifier.
        
        Args:
            model_path: Path to custom trained model weights (optional)
            use_food101: Whether to use Food-101 class mappings
        """
        self.model_path = model_path
        self.use_food101 = use_food101
        self.model: Optional[Model] = None
        self._initialized = False
        
        # Class labels
        self.class_labels = FOOD_101_CLASSES if use_food101 else []
        
    def initialize(self) -> bool:
        """
        Load and initialize the model.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True
            
        if not TF_AVAILABLE:
            print("TensorFlow not available - using mock classifier")
            self._initialized = True
            return True
        
        try:
            print("Loading MobileNetV2 food classifier...")
            
            # Check if we have a custom trained model
            if self.model_path and os.path.exists(self.model_path):
                # Load custom Food-101 trained model
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded custom model from {self.model_path}")
            else:
                # Use pre-trained MobileNetV2 on ImageNet
                # For demo, we'll use ImageNet and map food-related classes
                base_model = MobileNetV2(
                    weights='imagenet',
                    include_top=True,
                    input_shape=(*self.INPUT_SIZE, 3)
                )
                self.model = base_model
                print("Loaded MobileNetV2 with ImageNet weights")
                
                # Note: For production, you would fine-tune on Food-101 dataset
                # or load a pre-trained Food-101 model
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing food classifier: {e}")
            return False
    
    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """Preprocess image for MobileNetV2 input."""
        # Resize to model input size
        img = img.resize(self.INPUT_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def _format_class_name(self, class_name: str) -> str:
        """Convert class name to display format."""
        # First check if it's a known ImageNet food class
        if class_name in IMAGENET_FOOD_CLASSES:
            return IMAGENET_FOOD_CLASSES[class_name]["display"]
        return class_name.replace('_', ' ').title()
    
    def _is_food_class(self, class_name: str) -> bool:
        """Check if the class name is a food-related ImageNet class."""
        class_lower = class_name.lower().replace(' ', '_')
        
        # ═══════════════════════════════════════════════════════════════
        # EXPLICIT NON-FOOD EXCLUSIONS (ImageNet classes that look similar to food)
        # ═══════════════════════════════════════════════════════════════
        NON_FOOD_CLASSES = {
            # Machine/Tools
            'spindle', 'reel', 'sewing_machine', 'thimble', 'hook', 'corkscrew',
            'can_opener', 'plunger', 'screwdriver', 'hammer', 'nail',
            # Animals (not food items)
            'goldfish', 'tench', 'great_white_shark', 'tiger_shark', 'electric_ray',
            'stingray', 'rooster', 'hen', 'ostrich', 'brambling', 'goldfinch',
            'robin', 'jay', 'magpie', 'jellyfish', 'sea_anemone', 'coral',
            'starfish', 'sea_urchin', 'sea_cucumber', 'snail', 'slug', 'chiton',
            # Nature/Objects
            'cliff', 'valley', 'volcano', 'coral_reef', 'seashore', 'lakeside',
            'geyser', 'sandbar', 'promontory', 'dam', 'pier', 'suspension_bridge',
            # Containers/Objects (not food)
            'tennis_ball', 'basketball', 'volleyball', 'soccer_ball', 'ping_pong_ball',
            'bucket', 'barrel', 'crate', 'chest', 'safe', 'mailbox', 'garbage_truck',
            # Clothing/Fashion
            'jersey', 'kimono', 'bikini', 'miniskirt', 'abaya', 'academic_gown',
            # Furniture
            'bookcase', 'bookshelf', 'desk', 'file_cabinet', 'throne',
            # Electronics
            'laptop', 'desktop_computer', 'keyboard', 'mouse', 'monitor', 'screen',
            'television', 'cellular_telephone', 'iPod', 'loudspeaker',
        }
        
        if class_lower in NON_FOOD_CLASSES:
            return False
        
        # Check if it's in our ImageNet food mapping (high confidence)
        if class_name in IMAGENET_FOOD_CLASSES:
            return True
        
        # Check Food-101 mapping
        if class_lower in FOOD_101_TO_LOCAL_MAPPING:
            return True
        
        # Check for partial matches in known food terms
        food_keywords = ['food', 'fruit', 'vegetable', 'meat', 'dish', 'soup', 'bread', 
                         'cake', 'pie', 'salad', 'rice', 'pasta', 'cheese', 'egg', 'fish',
                         'pizza', 'burger', 'sandwich', 'curry', 'chicken', 'beef', 'pork',
                         'banana', 'apple', 'orange', 'lemon', 'grape', 'melon', 'mango',
                         'strawberry', 'pineapple', 'coconut', 'cabbage', 'broccoli',
                         'potato', 'tomato', 'onion', 'carrot', 'pepper', 'mushroom',
                         'bean', 'corn', 'pea', 'lentil', 'dhal', 'noodle', 'dumpling',
                         'sushi', 'ramen', 'curry', 'biryani', 'naan', 'roti', 'samosa']
        for keyword in food_keywords:
            if keyword in class_lower:
                return True
        
        return False
    
    def _map_to_local_foods(self, class_name: str) -> List[str]:
        """Map ImageNet or Food-101 class to local food database IDs."""
        # First check ImageNet food mapping
        if class_name in IMAGENET_FOOD_CLASSES:
            return IMAGENET_FOOD_CLASSES[class_name]["foods"]
        
        # Check direct Food-101 mapping
        class_lower = class_name.lower().replace(' ', '_')
        
        if class_lower in FOOD_101_TO_LOCAL_MAPPING:
            return FOOD_101_TO_LOCAL_MAPPING[class_lower]
        
        # Try partial matches
        for key, foods in FOOD_101_TO_LOCAL_MAPPING.items():
            if key in class_lower or class_lower in key:
                return foods
        
        # Return the class name itself as fallback
        return [class_lower]
    
    def _get_mock_predictions(self, image_bytes: bytes) -> List[FoodPrediction]:
        """Generate mock predictions when TensorFlow is not available."""
        import random
        
        # Select random foods for demo
        mock_foods = [
            ("chicken_curry", 0.85),
            ("fried_rice", 0.72),
            ("rice", 0.65),
            ("dhal_curry", 0.58),
            ("grilled_salmon", 0.45),
        ]
        
        # Shuffle and select top 3
        random.shuffle(mock_foods)
        selected = mock_foods[:3]
        
        predictions = []
        for class_name, base_conf in selected:
            conf = base_conf + random.uniform(-0.1, 0.1)
            conf = max(0.1, min(0.99, conf))
            
            predictions.append(FoodPrediction(
                class_name=class_name,
                display_name=self._format_class_name(class_name),
                confidence=conf,
                local_food_ids=self._map_to_local_foods(class_name)
            ))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions
    
    async def classify_image(
        self, 
        image_bytes: bytes, 
        top_k: int = 5,
        confidence_threshold: float = 0.1
    ) -> ClassificationResult:
        """
        Classify a food image.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            ClassificationResult with predictions
        """
        import time
        start_time = time.time()
        
        logger.info("=" * 50)
        logger.info("🍕 FOOD CLASSIFIER - classify_image() called")
        logger.info(f"📏 Image bytes: {len(image_bytes)}")
        logger.info(f"🎯 top_k={top_k}, threshold={confidence_threshold}")
        
        # Initialize if needed
        if not self._initialized:
            logger.info("⚙️ Initializing classifier...")
            self.initialize()
        
        try:
            # Load image
            if not PIL_AVAILABLE:
                logger.error("❌ PIL not available!")
                return ClassificationResult(
                    success=False,
                    predictions=[],
                    image_size=(0, 0),
                    processing_time_ms=0,
                    error="PIL not available for image processing"
                )
            
            img = Image.open(io.BytesIO(image_bytes))
            original_size = img.size
            logger.info(f"📷 Image loaded: {original_size[0]}x{original_size[1]}")
            
            # Use mock predictions if TensorFlow not available
            if not TF_AVAILABLE or self.model is None:
                logger.warning("⚠️ TensorFlow not available, using MOCK predictions!")
                predictions = self._get_mock_predictions(image_bytes)
                processing_time = (time.time() - start_time) * 1000
                
                return ClassificationResult(
                    success=True,
                    predictions=predictions[:top_k],
                    image_size=original_size,
                    processing_time_ms=processing_time,
                    model_version="mock_classifier"
                )
            
            logger.info("🧠 Running MobileNetV2 inference...")
            
            # Preprocess image
            img_array = self._preprocess_image(img)
            
            # Run inference
            predictions_raw = self.model.predict(img_array, verbose=0)
            
            # Decode predictions (ImageNet classes) - get more to filter food classes
            decoded = decode_predictions(predictions_raw, top=50)[0]
            
            logger.info("-" * 40)
            logger.info("🔍 RAW ImageNet Predictions (top 10):")
            for i, (imagenet_id, class_name, conf) in enumerate(decoded[:10]):
                logger.info(f"  #{i+1}: {class_name} ({conf:.2%})")
            logger.info("-" * 40)
            
            # Filter and map to food-related predictions
            predictions = []
            food_predictions = []
            
            for _, class_name, confidence in decoded:
                if confidence < confidence_threshold:
                    continue
                
                # Check if this is a food-related class
                is_food = self._is_food_class(class_name)
                
                # Map ImageNet class to food if applicable
                local_foods = self._map_to_local_foods(class_name)
                
                pred = FoodPrediction(
                    class_name=class_name,
                    display_name=self._format_class_name(class_name),
                    confidence=float(confidence),
                    local_food_ids=local_foods,
                    is_food=is_food
                )
                
                if is_food:
                    food_predictions.append(pred)
                    logger.info(f"✅ FOOD: {class_name} -> {self._format_class_name(class_name)} ({confidence:.2%})")
                predictions.append(pred)
            
            # Prioritize food predictions, but fallback to all predictions
            final_predictions = food_predictions if food_predictions else predictions
            logger.info(f"📊 Found {len(food_predictions)} food classes, {len(predictions)} total")
            
            # Sort by confidence and limit
            final_predictions.sort(key=lambda x: x.confidence, reverse=True)
            final_predictions = final_predictions[:top_k]
            
            # If still no predictions, return the top predictions anyway
            if not final_predictions and predictions:
                final_predictions = predictions[:top_k]
            
            processing_time = (time.time() - start_time) * 1000
            
            return ClassificationResult(
                success=True,
                predictions=final_predictions,
                image_size=original_size,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return ClassificationResult(
                success=False,
                predictions=[],
                image_size=(0, 0),
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    async def classify_image_file(
        self,
        file_path: str,
        top_k: int = 5,
        confidence_threshold: float = 0.1
    ) -> ClassificationResult:
        """
        Classify a food image from file path.
        
        Args:
            file_path: Path to image file
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            ClassificationResult with predictions
        """
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            return await self.classify_image(image_bytes, top_k, confidence_threshold)
        except Exception as e:
            return ClassificationResult(
                success=False,
                predictions=[],
                image_size=(0, 0),
                processing_time_ms=0,
                error=f"Failed to read file: {e}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_food_classifier: Optional[FoodClassifier] = None


def get_food_classifier() -> FoodClassifier:
    """Get or create singleton food classifier instance."""
    global _food_classifier
    if _food_classifier is None:
        # Check for custom model in ml_models directory
        model_path = Path(__file__).parent.parent.parent / "ml_models" / "nutrition" / "food_classifier.h5"
        
        _food_classifier = FoodClassifier(
            model_path=str(model_path) if model_path.exists() else None
        )
        _food_classifier.initialize()
    
    return _food_classifier

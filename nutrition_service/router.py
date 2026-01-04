"""
SMARTCARE+ Nutrition Service Router

Owner: Kulasekara
Endpoints for food recognition, meal logging, RDA-based meal planning, and hydration.
Uses USDA FoodData Central API and local food database.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime, date
import random
import logging

# Setup logger for nutrition service
logger = logging.getLogger("smartcare.nutrition")
logger.setLevel(logging.DEBUG)

from .models import (
    USDAClient,
    LocalFoodDatabase,
    FoodItem,
    MealPlanGenerator,
    MealLogger,
    MealType,
    RDAProfile,
    DietaryRestriction,
    get_usda_client,
    get_local_food_db,
    get_meal_plan_generator,
    get_meal_logger,
    # Food Classifier
    FoodClassifier,
    FoodPrediction,
    ClassificationResult,
    get_food_classifier
)

router = APIRouter()


# Service instances (singleton pattern)
_usda_client: Optional[USDAClient] = None
_local_food_db: Optional[LocalFoodDatabase] = None
_meal_planner: Optional[MealPlanGenerator] = None
_meal_logger: Optional[MealLogger] = None
_food_classifier: Optional[FoodClassifier] = None


def get_services():
    """Get or initialize service instances."""
    global _usda_client, _local_food_db, _meal_planner, _meal_logger, _food_classifier
    if _usda_client is None:
        _usda_client = get_usda_client()
    if _local_food_db is None:
        _local_food_db = get_local_food_db()
    if _meal_planner is None:
        _meal_planner = get_meal_plan_generator()
    if _meal_logger is None:
        _meal_logger = get_meal_logger()
    if _food_classifier is None:
        _food_classifier = get_food_classifier()
    return _usda_client, _local_food_db, _meal_planner, _meal_logger, _food_classifier


# Estimated nutrition data for common foods (when not in local DB)
ESTIMATED_NUTRITION = {
    # Main dishes
    "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "fiber": 2},
    "hamburger": {"calories": 295, "protein": 17, "carbs": 24, "fat": 14, "fiber": 1},
    "cheeseburger": {"calories": 303, "protein": 15, "carbs": 27, "fat": 15, "fiber": 1},
    "hot_dog": {"calories": 290, "protein": 11, "carbs": 22, "fat": 18, "fiber": 1},
    "hotdog": {"calories": 290, "protein": 11, "carbs": 22, "fat": 18, "fiber": 1},
    "burrito": {"calories": 340, "protein": 15, "carbs": 42, "fat": 13, "fiber": 4},
    "pasta": {"calories": 220, "protein": 8, "carbs": 43, "fat": 1, "fiber": 2},
    "carbonara": {"calories": 360, "protein": 14, "carbs": 38, "fat": 17, "fiber": 2},
    "spaghetti": {"calories": 220, "protein": 8, "carbs": 43, "fat": 1, "fiber": 2},
    
    # South Asian Foods
    "curry": {"calories": 180, "protein": 8, "carbs": 12, "fat": 10, "fiber": 3},
    "dhal": {"calories": 116, "protein": 9, "carbs": 20, "fat": 1, "fiber": 8},
    "dhal_curry": {"calories": 116, "protein": 9, "carbs": 20, "fat": 1, "fiber": 8},
    "dhal/lentils": {"calories": 116, "protein": 9, "carbs": 20, "fat": 1, "fiber": 8},
    "lentil": {"calories": 116, "protein": 9, "carbs": 20, "fat": 1, "fiber": 8},
    "parippu": {"calories": 116, "protein": 9, "carbs": 20, "fat": 1, "fiber": 8},
    "chicken_curry": {"calories": 240, "protein": 22, "carbs": 8, "fat": 14, "fiber": 2},
    "biryani": {"calories": 290, "protein": 12, "carbs": 38, "fat": 10, "fiber": 2},
    "samosa": {"calories": 262, "protein": 5, "carbs": 28, "fat": 15, "fiber": 3},
    "naan": {"calories": 262, "protein": 9, "carbs": 45, "fat": 5, "fiber": 2},
    "roti": {"calories": 120, "protein": 3, "carbs": 20, "fat": 3, "fiber": 2},
    "soup/curry": {"calories": 150, "protein": 8, "carbs": 15, "fat": 6, "fiber": 3},
    "food_bowl": {"calories": 180, "protein": 8, "carbs": 20, "fat": 8, "fiber": 3},
    "cooked_dish": {"calories": 200, "protein": 10, "carbs": 22, "fat": 9, "fiber": 3},
    
    # Seafood
    "lobster": {"calories": 89, "protein": 19, "carbs": 0, "fat": 1, "fiber": 0},
    "crab": {"calories": 97, "protein": 19, "carbs": 0, "fat": 2, "fiber": 0},
    "shrimp": {"calories": 99, "protein": 24, "carbs": 0, "fat": 0, "fiber": 0},
    
    # Fruits
    "banana": {"calories": 89, "protein": 1, "carbs": 23, "fat": 0, "fiber": 3},
    "apple": {"calories": 52, "protein": 0, "carbs": 14, "fat": 0, "fiber": 2},
    "orange": {"calories": 47, "protein": 1, "carbs": 12, "fat": 0, "fiber": 2},
    "strawberry": {"calories": 32, "protein": 1, "carbs": 8, "fat": 0, "fiber": 2},
    "pineapple": {"calories": 50, "protein": 0, "carbs": 13, "fat": 0, "fiber": 1},
    "watermelon": {"calories": 30, "protein": 1, "carbs": 8, "fat": 0, "fiber": 0},
    "mango": {"calories": 60, "protein": 1, "carbs": 15, "fat": 0, "fiber": 2},
    "papaya": {"calories": 43, "protein": 0, "carbs": 11, "fat": 0, "fiber": 2},
    "coconut": {"calories": 354, "protein": 3, "carbs": 15, "fat": 33, "fiber": 9},
    "grape": {"calories": 67, "protein": 1, "carbs": 17, "fat": 0, "fiber": 1},
    
    # Vegetables
    "broccoli": {"calories": 34, "protein": 3, "carbs": 7, "fat": 0, "fiber": 3},
    "cabbage": {"calories": 25, "protein": 1, "carbs": 6, "fat": 0, "fiber": 2},
    "cauliflower": {"calories": 25, "protein": 2, "carbs": 5, "fat": 0, "fiber": 2},
    "cucumber": {"calories": 16, "protein": 1, "carbs": 4, "fat": 0, "fiber": 1},
    "mushroom": {"calories": 22, "protein": 3, "carbs": 3, "fat": 0, "fiber": 1},
    
    # Desserts
    "ice_cream": {"calories": 207, "protein": 4, "carbs": 24, "fat": 11, "fiber": 1},
    "cake": {"calories": 257, "protein": 3, "carbs": 38, "fat": 11, "fiber": 0},
    "chocolate": {"calories": 546, "protein": 5, "carbs": 60, "fat": 31, "fiber": 7},
    
    # Bread & Bakery
    "bread": {"calories": 265, "protein": 9, "carbs": 49, "fat": 3, "fiber": 2},
    "bagel": {"calories": 250, "protein": 10, "carbs": 50, "fat": 1, "fiber": 2},
    "pretzel": {"calories": 380, "protein": 9, "carbs": 80, "fat": 3, "fiber": 3},
    
    # Drinks
    "coffee": {"calories": 2, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0},
    "espresso": {"calories": 2, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0},
    
    # Default for unknown foods
    "default": {"calories": 200, "protein": 8, "carbs": 25, "fat": 8, "fiber": 2},
}


def _create_estimated_food(display_name: str, class_name: str) -> dict:
    """
    Create estimated nutrition data for foods not in local database.
    
    Uses common nutrition values or estimates based on food type.
    """
    # Normalize class name for lookup
    lookup_key = class_name.lower().replace(' ', '_').replace('-', '_')
    
    # Try to find matching nutrition data
    nutrition = None
    for key in [lookup_key, display_name.lower().replace(' ', '_')]:
        if key in ESTIMATED_NUTRITION:
            nutrition = ESTIMATED_NUTRITION[key]
            break
        # Try partial match
        for food_key, food_nutrition in ESTIMATED_NUTRITION.items():
            if food_key in key or key in food_key:
                nutrition = food_nutrition
                break
        if nutrition:
            break
    
    # Use default if no match
    if not nutrition:
        nutrition = ESTIMATED_NUTRITION["default"]
    
    return {
        "name": display_name,
        "fdc_id": None,
        "category": "Detected Food",
        "serving_size": 100,
        "serving_unit": "g",
        "calories": nutrition["calories"],
        "protein": nutrition["protein"],
        "carbohydrates": nutrition["carbs"],
        "fat": nutrition["fat"],
        "fiber": nutrition["fiber"],
        "sugar": 0,
        "sodium": 0,
        "estimated": True,  # Flag to indicate this is estimated data
        "note": f"Nutritional values are estimates for {display_name}"
    }


# ============= Pydantic Models =============

class MealLogRequest(BaseModel):
    user_id: str
    foods: List[dict]  # List of {fdc_id or name, portion_multiplier}
    meal_type: str  # breakfast, lunch, dinner, snack
    notes: Optional[str] = None


class GeneratePlanRequest(BaseModel):
    user_id: str
    duration_days: int = 7
    restrictions: Optional[List[str]] = None  # List of DietaryRestriction values
    calorie_target: Optional[int] = None
    budget_level: Optional[str] = "medium"  # low, medium, high
    food_dislikes: Optional[List[str]] = None


class HydrationLogRequest(BaseModel):
    user_id: str
    amount_ml: int
    beverage_type: str = "water"


class UserProfileRequest(BaseModel):
    user_id: str
    age: int = 70
    weight_kg: float = 70.0
    height_cm: float = 165.0
    activity_level: str = "sedentary"  # sedentary, light, moderate
    restrictions: Optional[List[str]] = None


# Note: Mock data removed - using actual food classification and local database


# ============= REST Endpoints =============

@router.post("/analyze-food")
async def analyze_food(
    image: UploadFile = File(...),
    top_k: int = 5,
    confidence_threshold: float = 0.1
):
    """
    Recognize food from image using MobileNetV2 CNN (Food-101 pre-trained).
    
    Accepts images from camera or gallery.
    Returns detected foods with nutrition data from local database.
    
    Args:
        image: Image file (JPEG/PNG) from camera or gallery
        top_k: Number of top predictions to return (default 5)
        confidence_threshold: Minimum confidence score (default 0.1)
    
    Returns:
        Classification results with nutrition data for detected foods
    """
    logger.info("=" * 60)
    logger.info("🍕 FOOD ANALYSIS REQUEST RECEIVED")
    logger.info("=" * 60)
    
    _, local_db, _, _, food_classifier = get_services()
    logger.info(f"📁 Image filename: {image.filename}")
    logger.info(f"📄 Content-Type: {image.content_type}")
    
    # Read image file
    content = await image.read()
    logger.info(f"📏 Image size: {len(content)} bytes")
    
    # Validate image
    if not content:
        logger.error("❌ Empty image file received!")
        raise HTTPException(status_code=400, detail="Empty image file")
    
    # Validate file type - check content-type OR file extension OR magic bytes
    content_type = image.content_type or ""
    filename = image.filename or ""
    
    # Check if it's a valid image
    is_valid_image = False
    
    # Method 1: Check content-type header
    if content_type.startswith("image/"):
        is_valid_image = True
        logger.info(f"✅ Valid content-type: {content_type}")
    
    # Method 2: Check file extension
    if not is_valid_image:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
        if filename.lower().endswith(valid_extensions):
            is_valid_image = True
            logger.info(f"✅ Valid file extension: {filename}")
    
    # Method 3: Check magic bytes (file signature)
    if not is_valid_image and len(content) > 8:
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        # GIF: 47 49 46 38
        if content[:3] == b'\xff\xd8\xff':
            is_valid_image = True
            logger.info("✅ Valid JPEG magic bytes detected")
        elif content[:8] == b'\x89PNG\r\n\x1a\n':
            is_valid_image = True
            logger.info("✅ Valid PNG magic bytes detected")
        elif content[:4] == b'GIF8':
            is_valid_image = True
            logger.info("✅ Valid GIF magic bytes detected")
    
    if not is_valid_image:
        logger.error(f"❌ Invalid image file - content_type: {content_type}, filename: {filename}")
        raise HTTPException(status_code=400, detail="File must be a valid image (JPEG/PNG/GIF)")
    
    logger.info("🔍 Starting MobileNetV2 CNN classification...")
    
    # Run CNN food classification
    result = await food_classifier.classify_image(
        image_bytes=content,
        top_k=top_k,
        confidence_threshold=confidence_threshold
    )
    
    logger.info(f"✅ Classification completed - Success: {result.success}")
    
    if not result.success:
        logger.error(f"❌ Classification failed: {result.error}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {result.error}")
    
    # Log CNN predictions
    logger.info("-" * 40)
    logger.info("🧠 MobileNetV2 CNN PREDICTIONS:")
    for i, pred in enumerate(result.predictions):
        logger.info(f"  #{i+1}: {pred.display_name} (class: {pred.class_name})")
        logger.info(f"       Confidence: {pred.confidence:.2%}")
        logger.info(f"       Is Food: {pred.is_food}")
    logger.info("-" * 40)
    
    # Map predictions to local food database with full nutrition
    detected_foods = []
    for pred in result.predictions:
        logger.info(f"🔎 Searching local DB for: {pred.display_name}")
        
        # Try to find matching food in local database
        food_matches = []
        for local_id in pred.local_food_ids:
            # Search by food ID or name
            matches = local_db.search(local_id) if hasattr(local_db, 'search') else []
            food_matches.extend(matches)
        
        # If no matches, search by prediction display name
        if not food_matches:
            food_matches = local_db.search(pred.display_name) if hasattr(local_db, 'search') else []
        
        logger.info(f"   Found {len(food_matches)} matches in local DB")
        
        detected_foods.append({
            "prediction": pred.to_dict(),
            "matched_foods": [f.to_dict() if hasattr(f, 'to_dict') else f for f in food_matches[:3]]
        })
    
    # Get the top prediction with full nutrition data
    primary_food = None
    if detected_foods and detected_foods[0]["matched_foods"]:
        primary_food = detected_foods[0]["matched_foods"][0]
        logger.info(f"✅ PRIMARY FOOD (from local DB): {primary_food.get('name', 'Unknown')}")
    elif detected_foods and detected_foods[0]["prediction"]:
        # No local match found - create estimated food data from prediction
        pred = detected_foods[0]["prediction"]
        primary_food = _create_estimated_food(pred["display_name"], pred["class_name"])
        logger.info(f"⚠️ PRIMARY FOOD (estimated): {pred['display_name']} (no local DB match)")
    else:
        logger.warning("⚠️ No primary food detected!")
    
    logger.info("=" * 60)
    logger.info(f"📤 RETURNING: primary_food = {primary_food.get('name') if primary_food else None}")
    logger.info("=" * 60)
    
    return {
        "status": "completed",
        "classification": result.to_dict(),
        "detected_foods": detected_foods,
        "primary_food": primary_food,
        "source_type": "camera_or_gallery",
        "tips": [
            "Tap on a food item to see full nutrition facts",
            "Use 'Add to Meal' to log this food",
            "Adjust portion size for accurate tracking"
        ]
    }


@router.get("/food/{food_id}")
async def get_food(food_id: str):
    """Get detailed nutrition data for a food item from USDA or local DB."""
    usda_client, local_db, _, _, _ = get_services()
    
    # Try to get from local DB first
    food = local_db.get_by_id(int(food_id)) if food_id.isdigit() else None
    
    if not food:
        # Search local DB by name
        results = local_db.search(food_id)
        food = results[0] if results else None
    
    if not food:
        # Try USDA API
        try:
            food = await usda_client.get_food(int(food_id))
        except:
            raise HTTPException(status_code=404, detail=f"Food '{food_id}' not found")
    
    return {
        "food_id": food_id,
        "nutrition": food.to_dict(),
        "rda_percent": {
            "calories": round((food.calories / 1800) * 100, 1),
            "protein": round((food.protein / 60) * 100, 1),
            "fiber": round((food.fiber / 25) * 100, 1) if food.fiber else 0,
        }
    }


@router.get("/search")
async def search_food(query: str, limit: int = 10, use_usda: bool = False):
    """
    Search food database.
    
    By default searches local database for speed.
    Set use_usda=True to search USDA FoodData Central API.
    """
    usda_client, local_db, _, _, _ = get_services()
    
    # Search local DB
    local_results = local_db.search(query)
    
    results = [f.to_dict() for f in local_results[:limit]]
    
    # Optionally search USDA API for more results
    if use_usda and len(results) < limit:
        try:
            usda_results = await usda_client.search(query, limit - len(results))
            for food in usda_results:
                results.append(food.to_dict())
        except Exception as e:
            # USDA API error - continue with local results
            pass
    
    return {
        "query": query,
        "results": results[:limit],
        "total": len(results),
        "source": "usda" if use_usda else "local"
    }


@router.get("/recent-foods")
async def get_recent_foods(user_id: str = "current_user", limit: int = 10):
    """
    Get recently logged foods for a user.
    
    Returns common foods if no recent foods are available.
    """
    _, local_db, _, meal_logger, _ = get_services()
    
    # Try to get from meal logger's daily log
    try:
        daily_log = meal_logger.get_daily_log(user_id)
        recent_foods = []
        
        # Extract foods from recent meals
        for meal in daily_log.get("meals", []):
            for food in meal.get("foods", []):
                recent_foods.append({
                    "name": food.get("name", "Unknown"),
                    "calories": food.get("calories", 0),
                    "serving": food.get("serving_size", "1 serving"),
                    "protein": food.get("protein", 0),
                    "carbs": food.get("carbs", 0),
                    "fat": food.get("fat", 0),
                })
        
        if recent_foods:
            # Return unique foods (deduplicated by name)
            seen = set()
            unique_foods = []
            for food in recent_foods:
                if food["name"] not in seen:
                    seen.add(food["name"])
                    unique_foods.append(food)
            return {"foods": unique_foods[:limit]}
    except Exception:
        pass
    
    # Fall back to popular/common foods from local database
    common_searches = ["rice", "chicken", "bread", "milk", "egg", "apple", "banana", "fish"]
    popular_foods = []
    
    for search_term in common_searches:
        results = local_db.search(search_term)
        if results:
            food = results[0]
            popular_foods.append({
                "name": food.description,
                "calories": food.calories,
                "serving": f"{food.serving_size} {food.serving_unit}",
                "protein": food.protein,
                "carbs": food.carbs,
                "fat": food.fat,
            })
    
    return {"foods": popular_foods[:limit]}


@router.post("/log-meal")
async def log_meal(request: MealLogRequest):
    """
    Log a meal entry with detailed nutrition tracking.
    
    Stores meal data and updates daily totals.
    """
    _, local_db, _, meal_logger, _ = get_services()
    
    # Parse meal type
    try:
        meal_type = MealType(request.meal_type.lower())
    except ValueError:
        meal_type = MealType.LUNCH  # Default
    
    # Calculate totals from logged foods
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    total_fiber = 0
    logged_foods = []
    food_items_for_log = []
    
    for food_entry in request.foods:
        food_name = food_entry.get("name", "")
        portion = food_entry.get("portion", 1.0)
        
        # Look up food in database
        results = local_db.search(food_name)
        if results:
            food = results[0]
            total_calories += food.calories * portion
            total_protein += food.protein * portion
            total_carbs += food.carbohydrates * portion
            total_fat += food.fat * portion
            total_fiber += food.fiber * portion
            logged_foods.append({
                "name": food.name,
                "portion": portion,
                "calories": round(food.calories * portion, 1)
            })
            food_items_for_log.append((food, portion))
        else:
            # Use provided values if food not found
            calories = food_entry.get("calories", 0)
            protein = food_entry.get("protein", 0)
            carbs = food_entry.get("carbs", 0)
            fat = food_entry.get("fat", 0)
            
            total_calories += calories * portion
            total_protein += protein * portion
            total_carbs += carbs * portion
            total_fat += fat * portion
            
            logged_foods.append({
                "name": food_name,
                "portion": portion,
                "calories": calories * portion
            })
            
            # Create a temporary FoodItem for logging
            from .models import FoodItem
            temp_food = FoodItem(
                fdc_id=0,
                name=food_name,
                category="Logged",
                serving_size=100,
                serving_unit="g",
                calories=calories,
                protein=protein,
                carbohydrates=carbs,
                fat=fat,
                fiber=0,
                sugar=0,
                sodium=0,
            )
            food_items_for_log.append((temp_food, portion))
    
    # Create MealItem objects and log the meal
    from .models.meal_planner import MealItem
    meal_items = []
    for food, portion in food_items_for_log:
        meal_items.append(MealItem(
            food=food,
            serving_size=100 * portion,
            serving_unit="g",
        ))
    
    # Log the meal
    log = meal_logger.log_meal(
        user_id=request.user_id,
        meal_type=meal_type,
        items=meal_items,
        notes=request.notes
    )
    
    logger.info(f"✅ Meal logged: {len(logged_foods)} foods, {total_calories:.0f} cal for user {request.user_id}")
    
    return {
        "status": "logged",
        "log_id": log.log_id,
        "user_id": request.user_id,
        "meal_type": request.meal_type,
        "foods_logged": len(logged_foods),
        "totals": {
            "calories": round(total_calories, 1),
            "protein": round(total_protein, 1),
            "carbohydrates": round(total_carbs, 1),
            "fat": round(total_fat, 1)
        },
        "logged_at": datetime.now().isoformat()
    }


@router.get("/daily-summary/{user_id}")
async def get_daily_summary(user_id: str, date: Optional[str] = None):
    """
    Get daily nutrient intake summary with RDA comparison.
    
    Shows consumed vs. recommended amounts for elderly-specific RDA.
    """
    _, _, meal_planner, meal_logger, _ = get_services()
    
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    
    # Get logged meals for the day
    daily_log = meal_logger.get_daily_log(user_id, target_date)
    
    # Get user's RDA profile (would come from user settings in production)
    rda = RDAProfile()
    
    # Calculate consumed totals
    consumed = daily_log.get("totals", {
        "calories": 0,
        "protein": 0,
        "carbohydrates": 0,
        "fat": 0,
        "fiber": 0
    })
    
    return {
        "user_id": user_id,
        "date": target_date,
        "total_calories": consumed.get("calories", 0),
        "target_calories": rda.calories,
        "protein": consumed.get("protein", 0),
        "protein_target": rda.protein,
        "carbs": consumed.get("carbohydrates", 0),
        "carbs_target": rda.carbohydrates,
        "fat": consumed.get("fat", 0),
        "fat_target": rda.fat,
        "fiber": consumed.get("fiber", 0),
        "fiber_target": rda.fiber,
        "calories": {
            "consumed": consumed.get("calories", 0),
            "goal": rda.calories,
            "percent": round((consumed.get("calories", 0) / rda.calories) * 100, 1)
        },
        "protein": {
            "consumed": consumed.get("protein", 0),
            "goal": rda.protein,
            "percent": round((consumed.get("protein", 0) / rda.protein) * 100, 1)
        },
        "carbohydrates": {
            "consumed": consumed.get("carbohydrates", 0),
            "goal": rda.carbohydrates,
            "percent": round((consumed.get("carbohydrates", 0) / rda.carbohydrates) * 100, 1)
        },
        "fat": {
            "consumed": consumed.get("fat", 0),
            "goal": rda.fat,
            "percent": round((consumed.get("fat", 0) / rda.fat) * 100, 1)
        },
        "fiber": {
            "consumed": consumed.get("fiber", 0),
            "goal": rda.fiber,
            "percent": round((consumed.get("fiber", 0) / rda.fiber) * 100, 1)
        },
        "meals": [
            {
                "meal_type": m.get("type", "").lower(),
                "name": m.get("name", ""),
                "calories": m.get("calories", 0),
                "time": m.get("time", ""),
                "foods": [{"name": name} for name in m.get("name", "").split(", ") if name]
            }
            for m in daily_log.get("meals", [])
        ],
        "recommendations": _get_nutrition_recommendations(consumed, rda)
    }


def _get_nutrition_recommendations(consumed: dict, rda: RDAProfile) -> List[str]:
    """Generate personalized nutrition recommendations."""
    recommendations = []
    
    cal_percent = consumed.get("calories", 0) / rda.calories
    protein_percent = consumed.get("protein", 0) / rda.protein
    fiber_percent = consumed.get("fiber", 0) / rda.fiber
    
    if cal_percent < 0.5:
        recommendations.append("Consider having a nutritious snack to meet your calorie goals")
    
    if protein_percent < 0.6:
        recommendations.append("Include more protein-rich foods like fish, eggs, or legumes")
    
    if fiber_percent < 0.5:
        recommendations.append("Add more fiber through whole grains, fruits, and vegetables")
    
    if not recommendations:
        recommendations.append("Great job! You're on track with your nutrition goals")
    
    return recommendations


@router.get("/meal-plan/{user_id}")
async def get_meal_plan(user_id: str, date: Optional[str] = None):
    """Get current meal plan for the user."""
    _, _, meal_planner, _, _ = get_services()
    
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    
    # Generate meal plan for requested days
    plans = meal_planner.generate_plan(
        user_id=user_id,
        duration_days=7,
        restrictions=None,
        preferences=None
    )
    
    if plans:
        # Convert plans to response format
        meal_plan = {}
        for i, daily_plan in enumerate(plans):
            meal_plan[f"day_{i}"] = [
                {
                    "meal_type": meal.meal_type.value if hasattr(meal.meal_type, 'value') else str(meal.meal_type),
                    "name": meal.meal_type.value.capitalize() if hasattr(meal.meal_type, 'value') else str(meal.meal_type).capitalize(),
                    "time": meal.time,
                    "calories": meal.total_calories,
                    "foods": [item.to_dict() for item in meal.items] if hasattr(meal, 'items') else [],
                    "description": ", ".join([item.food.description for item in meal.items]) if hasattr(meal, 'items') else "",
                    "serving_size": "1 serving"
                }
                for meal in daily_plan.meals
            ]
        
        return {
            "user_id": user_id,
            "date": target_date,
            "meal_plan": meal_plan,
            "total_calories": sum(p.total_calories for p in plans) / len(plans) if plans else 0,
        }
    else:
        return {
            "user_id": user_id,
            "date": target_date,
            "meal_plan": {},
            "total_calories": 0,
        }


@router.post("/generate-plan")
async def generate_plan(request: GeneratePlanRequest):
    """
    Generate personalized meal plan based on RDA and dietary restrictions.
    
    Supports restrictions like diabetic, low-sodium, vegetarian, etc.
    Budget levels: low, medium, high (affects food selection by cost)
    """
    _, _, meal_planner, _, _ = get_services()
    
    # Parse dietary restrictions
    restrictions = []
    if request.restrictions:
        for r in request.restrictions:
            try:
                restrictions.append(DietaryRestriction(r.lower()))
            except ValueError:
                pass  # Skip invalid restrictions
    
    # Create adjusted RDA profile
    rda = RDAProfile()
    if request.calorie_target:
        rda.calories = request.calorie_target
    
    if restrictions:
        rda = rda.adjust_for_restrictions(restrictions)
    
    # Build preferences dict with budget and dislikes
    preferences = {
        "calorie_goal": request.calorie_target,
        "budget_level": request.budget_level or "medium",
        "dislikes": request.food_dislikes or [],
    }
    
    # Generate meal plan
    plans = meal_planner.generate_plan(
        user_id=request.user_id,
        duration_days=request.duration_days,
        restrictions=restrictions,
        preferences=preferences,
        base_rda=rda
    )
    
    # Convert plans to response format
    meal_plan = {}
    for i, daily_plan in enumerate(plans):
        meal_plan[f"day_{i}"] = [
            {
                "meal_type": meal.meal_type.value if hasattr(meal.meal_type, 'value') else str(meal.meal_type),
                "name": meal.meal_type.value.capitalize() if hasattr(meal.meal_type, 'value') else str(meal.meal_type).capitalize(),
                "time": meal.time,
                "calories": meal.total_calories,
                "foods": [item.to_dict() for item in meal.items] if hasattr(meal, 'items') else [],
            }
            for meal in daily_plan.meals
        ]
    
    return {
        "status": "generated",
        "user_id": request.user_id,
        "duration_days": request.duration_days,
        "restrictions_applied": [r.value for r in restrictions],
        "budget_level": request.budget_level or "medium",
        "daily_targets": {
            "calories": rda.calories,
            "protein": rda.protein,
            "carbohydrates": rda.carbohydrates,
            "fat": rda.fat
        },
        "meal_plan": meal_plan,
        "generated_at": datetime.now().isoformat()
    }


@router.get("/restrictions")
async def get_available_restrictions():
    """Get list of available dietary restrictions."""
    return {
        "restrictions": [
            {"id": r.value, "name": r.value.replace("_", " ").title(), "description": _get_restriction_description(r)}
            for r in DietaryRestriction
        ]
    }


def _get_restriction_description(restriction: DietaryRestriction) -> str:
    """Get description for dietary restriction."""
    descriptions = {
        DietaryRestriction.DIABETIC: "Low glycemic index foods, controlled carbohydrates",
        DietaryRestriction.LOW_SODIUM: "Reduced sodium intake for heart health",
        DietaryRestriction.LOW_FAT: "Limited fat content for weight management",
        DietaryRestriction.HIGH_FIBER: "Increased fiber for digestive health",
        DietaryRestriction.LOW_CHOLESTEROL: "Reduced cholesterol for heart health",
        DietaryRestriction.LACTOSE_FREE: "No dairy products",
        DietaryRestriction.GLUTEN_FREE: "No wheat, barley, or rye products",
        DietaryRestriction.VEGETARIAN: "No meat or fish",
        DietaryRestriction.SOFT_FOODS: "Easy to chew foods for dental issues",
        DietaryRestriction.KIDNEY_FRIENDLY: "Controlled protein, phosphorus, and potassium",
        DietaryRestriction.HEART_HEALTHY: "Low sodium, low saturated fat, high fiber"
    }
    return descriptions.get(restriction, "")


@router.post("/hydration/log")
async def log_hydration(request: HydrationLogRequest):
    """Log water/fluid intake."""
    _, _, _, meal_logger, _ = get_services()
    
    # Log hydration
    meal_logger.log_hydration(
        user_id=request.user_id,
        amount_ml=request.amount_ml,
        beverage_type=request.beverage_type
    )
    
    return {
        "status": "logged",
        "user_id": request.user_id,
        "amount_ml": request.amount_ml,
        "beverage_type": request.beverage_type,
        "logged_at": datetime.now().isoformat()
    }


@router.get("/hydration/{user_id}")
async def get_hydration(user_id: str, date: Optional[str] = None):
    """Get daily hydration data with recommendations for elderly."""
    _, _, _, meal_logger, _ = get_services()
    
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    
    # Get hydration log
    hydration = meal_logger.get_hydration_log(user_id, target_date)
    
    total_ml = hydration.get("total_ml", 0)
    goal_ml = 2000  # Elderly recommended daily intake
    
    # Calculate glasses (250ml per glass)
    glasses = total_ml // 250
    goal_glasses = goal_ml // 250
    
    # Hydration status and recommendations
    if total_ml >= goal_ml:
        status = "excellent"
        recommendation = "Great job staying hydrated!"
    elif total_ml >= goal_ml * 0.75:
        status = "good"
        recommendation = "You're doing well, try to drink a bit more"
    elif total_ml >= goal_ml * 0.5:
        status = "fair"
        recommendation = "Consider drinking more water throughout the day"
    else:
        status = "low"
        recommendation = "Please drink more fluids. Dehydration is a health risk."
    
    return {
        "user_id": user_id,
        "date": target_date,
        "glasses": glasses,
        "goal_glasses": goal_glasses,
        "total_ml": total_ml,
        "goal_ml": goal_ml,
        "percent": round((total_ml / goal_ml) * 100, 1),
        "status": status,
        "recommendation": recommendation,
        "log": hydration.get("log", [])
    }


@router.get("/rda-profile/{user_id}")
async def get_rda_profile(user_id: str):
    """Get the RDA (Recommended Daily Allowance) profile for a user."""
    # In production, this would come from user settings/database
    rda = RDAProfile()
    
    return {
        "user_id": user_id,
        "profile": {
            "calories": rda.calories,
            "protein": rda.protein,
            "carbohydrates": rda.carbohydrates,
            "fat": rda.fat,
            "fiber": rda.fiber,
            "sodium": rda.sodium,
            "sugar": rda.sugar,
            "vitamins": {
                "vitamin_d": rda.vitamin_d,
                "vitamin_b12": rda.vitamin_b12,
                "vitamin_c": rda.vitamin_c,
                "vitamin_a": rda.vitamin_a
            },
            "minerals": {
                "calcium": rda.calcium,
                "iron": rda.iron,
                "potassium": rda.potassium,
                "magnesium": rda.magnesium
            }
        },
        "age_group": "65+",
        "note": "Values based on dietary guidelines for adults 65 and older"
    }


# In-memory storage for nutrition preferences (would use database in production)
_nutrition_preferences: Dict[str, dict] = {}


class NutritionPreferencesRequest(BaseModel):
    user_id: str
    dietary_restrictions: Optional[List[str]] = []
    food_allergies: Optional[List[str]] = []
    food_dislikes: Optional[List[str]] = []
    budget_level: Optional[str] = "medium"  # low, medium, high
    calorie_target: Optional[int] = None
    preferred_cuisines: Optional[List[str]] = []


@router.get("/nutrition-preferences/{user_id}")
async def get_nutrition_preferences(user_id: str):
    """
    Get user's nutrition preferences for meal planning.
    
    Returns dietary restrictions, budget level, dislikes, etc.
    """
    prefs = _nutrition_preferences.get(user_id, {
        "dietary_restrictions": [],
        "food_allergies": [],
        "food_dislikes": [],
        "budget_level": "medium",
        "calorie_target": None,
        "preferred_cuisines": []
    })
    
    return {
        "user_id": user_id,
        "preferences": prefs
    }


@router.post("/nutrition-preferences")
async def update_nutrition_preferences(request: NutritionPreferencesRequest):
    """
    Update user's nutrition preferences for meal planning.
    
    Saves dietary restrictions, budget level, food dislikes, etc.
    """
    _nutrition_preferences[request.user_id] = {
        "dietary_restrictions": request.dietary_restrictions or [],
        "food_allergies": request.food_allergies or [],
        "food_dislikes": request.food_dislikes or [],
        "budget_level": request.budget_level or "medium",
        "calorie_target": request.calorie_target,
        "preferred_cuisines": request.preferred_cuisines or []
    }
    
    return {
        "status": "saved",
        "user_id": request.user_id,
        "preferences": _nutrition_preferences[request.user_id]
    }

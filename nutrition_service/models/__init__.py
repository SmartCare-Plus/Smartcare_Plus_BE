"""
SMARTCARE+ Nutrition Service Models

Food recognition, USDA API integration, and meal planning.
"""

from .usda_client import (
    USDAClient,
    LocalFoodDatabase,
    FoodItem,
    NutrientInfo,
    get_usda_client,
    get_local_food_db
)

from .meal_planner import (
    MealPlanGenerator,
    MealLogger,
    MealType,
    MealItem,
    Meal,
    DailyMealPlan,
    MealLog,
    RDAProfile,
    DietaryRestriction,
    get_meal_plan_generator,
    get_meal_logger
)

from .food_classifier import (
    FoodClassifier,
    FoodPrediction,
    ClassificationResult,
    FOOD_101_CLASSES,
    get_food_classifier
)

__all__ = [
    # USDA Client
    "USDAClient",
    "LocalFoodDatabase", 
    "FoodItem",
    "NutrientInfo",
    "get_usda_client",
    "get_local_food_db",
    # Meal Planner
    "MealPlanGenerator",
    "MealLogger",
    "MealType",
    "MealItem",
    "Meal",
    "DailyMealPlan",
    "MealLog",
    "RDAProfile",
    "DietaryRestriction",
    "get_meal_plan_generator",
    "get_meal_logger",
    # Food Classifier
    "FoodClassifier",
    "FoodPrediction",
    "ClassificationResult",
    "FOOD_101_CLASSES",
    "get_food_classifier",
]

"""
SMARTCARE+ Nutrition Service - Meal Plan Generator

Owner: Kulasekara
RDA-based meal planning with dietary restrictions support for elderly users.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, date, timedelta
from enum import Enum
import random

from .usda_client import FoodItem, get_local_food_db, LocalFoodDatabase


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class MealType(Enum):
    """Meal types throughout the day."""
    BREAKFAST = "breakfast"
    MORNING_SNACK = "morning_snack"
    LUNCH = "lunch"
    AFTERNOON_SNACK = "afternoon_snack"
    DINNER = "dinner"
    EVENING_SNACK = "evening_snack"


class DietaryRestriction(Enum):
    """Common dietary restrictions for elderly."""
    DIABETIC = "diabetic"
    LOW_SODIUM = "low_sodium"
    LOW_FAT = "low_fat"
    HIGH_FIBER = "high_fiber"
    LOW_CHOLESTEROL = "low_cholesterol"
    LACTOSE_FREE = "lactose_free"
    GLUTEN_FREE = "gluten_free"
    VEGETARIAN = "vegetarian"
    SOFT_FOODS = "soft_foods"  # For dental issues
    KIDNEY_FRIENDLY = "kidney_friendly"
    HEART_HEALTHY = "heart_healthy"


@dataclass
class RDAProfile:
    """
    Recommended Daily Allowance profile for elderly users.
    
    Based on dietary guidelines for adults 65+.
    """
    # Macronutrients
    calories: float = 1800.0  # kcal (lower for sedentary elderly)
    protein: float = 60.0     # g (0.8-1.0g per kg body weight)
    carbohydrates: float = 225.0  # g (45-65% of calories)
    fat: float = 60.0         # g (20-35% of calories)
    fiber: float = 25.0       # g (21-30g recommended)
    
    # Limits
    sodium: float = 1500.0    # mg (lower for hypertension risk)
    sugar: float = 25.0       # g (limit added sugars)
    saturated_fat: float = 15.0  # g (<10% of calories)
    cholesterol: float = 300.0   # mg
    
    # Vitamins (mcg or mg)
    vitamin_d: float = 20.0   # mcg (higher for elderly)
    vitamin_b12: float = 2.4  # mcg
    vitamin_c: float = 90.0   # mg
    vitamin_a: float = 900.0  # mcg RAE
    
    # Minerals
    calcium: float = 1200.0   # mg (higher for bone health)
    iron: float = 8.0         # mg
    potassium: float = 2600.0 # mg
    magnesium: float = 420.0  # mg
    
    def adjust_for_restrictions(self, restrictions: List[DietaryRestriction]) -> "RDAProfile":
        """Create adjusted RDA based on dietary restrictions."""
        adjusted = RDAProfile(
            calories=self.calories,
            protein=self.protein,
            carbohydrates=self.carbohydrates,
            fat=self.fat,
            fiber=self.fiber,
            sodium=self.sodium,
            sugar=self.sugar,
            saturated_fat=self.saturated_fat,
            cholesterol=self.cholesterol,
            vitamin_d=self.vitamin_d,
            vitamin_b12=self.vitamin_b12,
            vitamin_c=self.vitamin_c,
            vitamin_a=self.vitamin_a,
            calcium=self.calcium,
            iron=self.iron,
            potassium=self.potassium,
            magnesium=self.magnesium,
        )
        
        for restriction in restrictions:
            if restriction == DietaryRestriction.DIABETIC:
                adjusted.sugar = 15.0
                adjusted.carbohydrates = 180.0
                adjusted.fiber = 30.0  # Higher fiber for blood sugar control
            
            elif restriction == DietaryRestriction.LOW_SODIUM:
                adjusted.sodium = 1000.0
            
            elif restriction == DietaryRestriction.LOW_FAT:
                adjusted.fat = 45.0
                adjusted.saturated_fat = 10.0
            
            elif restriction == DietaryRestriction.HIGH_FIBER:
                adjusted.fiber = 35.0
            
            elif restriction == DietaryRestriction.LOW_CHOLESTEROL:
                adjusted.cholesterol = 200.0
                adjusted.saturated_fat = 10.0
            
            elif restriction == DietaryRestriction.HEART_HEALTHY:
                adjusted.sodium = 1200.0
                adjusted.saturated_fat = 10.0
                adjusted.cholesterol = 200.0
                adjusted.fiber = 30.0
            
            elif restriction == DietaryRestriction.KIDNEY_FRIENDLY:
                adjusted.potassium = 2000.0
                adjusted.sodium = 1500.0
                adjusted.protein = 50.0  # Moderate protein
        
        return adjusted


@dataclass
class MealItem:
    """Single item in a meal."""
    food: FoodItem
    serving_size: float
    serving_unit: str = "g"
    
    @property
    def scaled_nutrients(self) -> Dict[str, float]:
        """Get nutrients scaled to serving size."""
        scale = self.serving_size / self.food.serving_size if self.food.serving_size > 0 else 1.0
        return {
            "calories": self.food.calories * scale,
            "protein": self.food.protein * scale,
            "carbohydrates": self.food.carbohydrates * scale,
            "fat": self.food.fat * scale,
            "fiber": self.food.fiber * scale,
            "sugar": self.food.sugar * scale,
            "sodium": self.food.sodium * scale,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.food.name,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
            **self.scaled_nutrients
        }


@dataclass
class Meal:
    """A complete meal with multiple items."""
    meal_type: MealType
    time: str
    items: List[MealItem] = field(default_factory=list)
    
    @property
    def total_calories(self) -> float:
        return sum(item.scaled_nutrients["calories"] for item in self.items)
    
    @property
    def total_protein(self) -> float:
        return sum(item.scaled_nutrients["protein"] for item in self.items)
    
    @property
    def total_carbs(self) -> float:
        return sum(item.scaled_nutrients["carbohydrates"] for item in self.items)
    
    @property
    def total_fat(self) -> float:
        return sum(item.scaled_nutrients["fat"] for item in self.items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meal_type": self.meal_type.value,
            "time": self.time,
            "items": [item.to_dict() for item in self.items],
            "totals": {
                "calories": round(self.total_calories, 1),
                "protein": round(self.total_protein, 1),
                "carbohydrates": round(self.total_carbs, 1),
                "fat": round(self.total_fat, 1),
            }
        }


@dataclass
class DailyMealPlan:
    """Complete meal plan for one day."""
    date: date
    meals: List[Meal] = field(default_factory=list)
    target_rda: Optional[RDAProfile] = None
    
    @property
    def total_calories(self) -> float:
        return sum(meal.total_calories for meal in self.meals)
    
    @property
    def total_protein(self) -> float:
        return sum(meal.total_protein for meal in self.meals)
    
    @property
    def macro_breakdown(self) -> Dict[str, float]:
        total_cal = self.total_calories or 1
        return {
            "protein_pct": (self.total_protein * 4 / total_cal) * 100,
            "carbs_pct": (sum(m.total_carbs for m in self.meals) * 4 / total_cal) * 100,
            "fat_pct": (sum(m.total_fat for m in self.meals) * 9 / total_cal) * 100,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        macros = self.macro_breakdown
        return {
            "date": self.date.isoformat(),
            "meals": [meal.to_dict() for meal in self.meals],
            "daily_totals": {
                "calories": round(self.total_calories, 1),
                "protein": round(self.total_protein, 1),
                "protein_pct": round(macros["protein_pct"], 1),
                "carbs_pct": round(macros["carbs_pct"], 1),
                "fat_pct": round(macros["fat_pct"], 1),
            },
            "rda_completion": self._calculate_rda_completion() if self.target_rda else None
        }
    
    def _calculate_rda_completion(self) -> Dict[str, float]:
        """Calculate percentage of RDA achieved."""
        if not self.target_rda:
            return {}
        
        return {
            "calories": round((self.total_calories / self.target_rda.calories) * 100, 1),
            "protein": round((self.total_protein / self.target_rda.protein) * 100, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MEAL PLAN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MealPlanGenerator:
    """
    RDA-based meal plan generator for elderly users.
    
    Features:
    - Personalized calorie and macro targets
    - Dietary restriction handling
    - Balanced meal distribution
    - Variety optimization
    """
    
    # Meal calorie distribution (percentage of daily calories)
    MEAL_DISTRIBUTION = {
        MealType.BREAKFAST: 0.25,
        MealType.MORNING_SNACK: 0.05,
        MealType.LUNCH: 0.30,
        MealType.AFTERNOON_SNACK: 0.10,
        MealType.DINNER: 0.25,
        MealType.EVENING_SNACK: 0.05,
    }
    
    # Default meal times
    MEAL_TIMES = {
        MealType.BREAKFAST: "8:00 AM",
        MealType.MORNING_SNACK: "10:30 AM",
        MealType.LUNCH: "12:30 PM",
        MealType.AFTERNOON_SNACK: "3:30 PM",
        MealType.DINNER: "6:30 PM",
        MealType.EVENING_SNACK: "8:30 PM",
    }
    
    # Food categories for meal types
    MEAL_CATEGORIES = {
        MealType.BREAKFAST: ["grains", "dairy", "fruits", "protein"],
        MealType.MORNING_SNACK: ["fruits", "dairy", "nuts"],
        MealType.LUNCH: ["protein", "vegetables", "grains"],
        MealType.AFTERNOON_SNACK: ["fruits", "dairy", "nuts"],
        MealType.DINNER: ["protein", "vegetables", "grains"],
        MealType.EVENING_SNACK: ["dairy", "fruits"],
    }
    
    # Foods to avoid for each restriction
    RESTRICTION_AVOID = {
        DietaryRestriction.DIABETIC: {"sweets", "candy", "cake", "cookie", "soda", "juice"},
        DietaryRestriction.LACTOSE_FREE: {"milk", "cheese", "yogurt", "cream", "butter"},
        DietaryRestriction.GLUTEN_FREE: {"bread", "pasta", "cereal", "wheat", "flour"},
        DietaryRestriction.VEGETARIAN: {"chicken", "beef", "pork", "fish", "meat", "bacon"},
        DietaryRestriction.LOW_SODIUM: {"bacon", "ham", "sausage", "chips", "pickles"},
    }
    
    def __init__(self, food_db: Optional[LocalFoodDatabase] = None):
        """
        Initialize meal plan generator.
        
        Args:
            food_db: Local food database (uses global if None)
        """
        self.food_db = food_db or get_local_food_db()
        self._used_foods: Set[str] = set()  # Track used foods for variety
    
    def generate_plan(
        self,
        user_id: str,
        duration_days: int = 7,
        restrictions: Optional[List[DietaryRestriction]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        base_rda: Optional[RDAProfile] = None
    ) -> List[DailyMealPlan]:
        """
        Generate a multi-day meal plan.
        
        Args:
            user_id: User ID for personalization
            duration_days: Number of days to plan
            restrictions: Dietary restrictions
            preferences: User food preferences
            base_rda: Base RDA profile (uses default if None)
        
        Returns:
            List of DailyMealPlan for each day
        """
        restrictions = restrictions or []
        preferences = preferences or {}
        
        # Get adjusted RDA
        rda = base_rda or RDAProfile()
        if restrictions:
            rda = rda.adjust_for_restrictions(restrictions)
        
        # Adjust calories based on preferences
        if "calorie_goal" in preferences:
            rda.calories = preferences["calorie_goal"]
        
        plans = []
        start_date = date.today()
        
        for day_offset in range(duration_days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Generate daily plan
            daily_plan = self._generate_daily_plan(
                current_date, rda, restrictions, preferences
            )
            plans.append(daily_plan)
            
            # Reset used foods every 3 days for variety
            if day_offset % 3 == 2:
                self._used_foods.clear()
        
        return plans
    
    def _generate_daily_plan(
        self,
        plan_date: date,
        rda: RDAProfile,
        restrictions: List[DietaryRestriction],
        preferences: Dict[str, Any]
    ) -> DailyMealPlan:
        """Generate a single day's meal plan."""
        daily_plan = DailyMealPlan(date=plan_date, target_rda=rda)
        
        # Include snacks based on preference
        include_snacks = preferences.get("include_snacks", True)
        
        for meal_type in MealType:
            # Skip snacks if not wanted
            if not include_snacks and "snack" in meal_type.value.lower():
                continue
            
            target_calories = rda.calories * self.MEAL_DISTRIBUTION[meal_type]
            
            meal = self._generate_meal(
                meal_type, target_calories, restrictions, preferences
            )
            daily_plan.meals.append(meal)
        
        return daily_plan
    
    def _generate_meal(
        self,
        meal_type: MealType,
        target_calories: float,
        restrictions: List[DietaryRestriction],
        preferences: Dict[str, Any]
    ) -> Meal:
        """Generate a single meal."""
        meal = Meal(
            meal_type=meal_type,
            time=self.MEAL_TIMES[meal_type]
        )
        
        # Get suitable food categories for this meal
        categories = self.MEAL_CATEGORIES.get(meal_type, ["protein", "vegetables"])
        
        # Build set of foods to avoid
        avoid_foods: Set[str] = set()
        for restriction in restrictions:
            avoid_foods.update(self.RESTRICTION_AVOID.get(restriction, set()))
        
        # Add user dislikes
        if "dislikes" in preferences:
            avoid_foods.update(set(preferences["dislikes"]))
        
        # Get budget level - convert string to int: low=1, medium=2, high=3
        budget_str = preferences.get("budget_level", "medium")
        budget_map = {"low": 1, "medium": 2, "high": 3}
        max_cost_level = budget_map.get(budget_str.lower() if isinstance(budget_str, str) else "medium", 2)
        
        current_calories = 0.0
        max_items = 3 if "snack" in meal_type.value else 4
        
        for category in categories:
            if len(meal.items) >= max_items:
                break
            
            if current_calories >= target_calories * 0.9:
                break
            
            # Find suitable food with budget filter
            food = self._find_food_for_meal(
                category, avoid_foods, target_calories - current_calories, max_cost_level
            )
            
            if food:
                # Calculate appropriate serving size
                remaining_cal = target_calories - current_calories
                if food.calories > 0:
                    serving_ratio = min(1.5, remaining_cal / food.calories)
                    serving_size = food.serving_size * serving_ratio
                else:
                    serving_size = food.serving_size
                
                meal_item = MealItem(
                    food=food,
                    serving_size=round(serving_size, 1),
                    serving_unit=food.serving_unit
                )
                meal.items.append(meal_item)
                current_calories += meal_item.scaled_nutrients["calories"]
                
                # Track used food
                self._used_foods.add(food.name.lower())
        
        return meal
    
    def _find_food_for_meal(
        self,
        category: str,
        avoid: Set[str],
        remaining_calories: float,
        max_cost_level: int = 3
    ) -> Optional[FoodItem]:
        """
        Find a suitable food item for the meal.
        
        Args:
            category: Food category to search in
            avoid: Set of words to avoid in food names
            remaining_calories: Remaining calories for the meal
            max_cost_level: Maximum cost level (1=low, 2=medium, 3=high)
        
        Returns:
            Suitable FoodItem or None
        """
        # Get foods from category
        category_foods = self.food_db.get_by_category(category)
        
        if not category_foods:
            # Fallback: search by category name
            category_foods = self.food_db.search(category, limit=20)
        
        # Filter out avoided and recently used foods
        suitable = []
        for food in category_foods:
            food_name_lower = food.name.lower()
            
            # Check if food should be avoided
            should_avoid = any(avoid_word in food_name_lower for avoid_word in avoid)
            if should_avoid:
                continue
            
            # Filter by cost level if applicable
            food_cost = getattr(food, 'cost_level', 2)
            if food_cost > max_cost_level:
                continue
            
            # Prefer foods not recently used
            if food_name_lower not in self._used_foods:
                suitable.append(food)
        
        if not suitable:
            # If all filtered out, use any from category
            suitable = [f for f in category_foods 
                       if not any(a in f.name.lower() for a in avoid)]
        
        if not suitable:
            return None
        
        # Prefer foods with appropriate calorie content
        # Sort by how close to ideal calories (1/3 of remaining)
        target = remaining_calories / 3
        suitable.sort(key=lambda f: abs(f.calories - target))
        
        # Add some randomness from top choices
        top_choices = suitable[:min(5, len(suitable))]
        return random.choice(top_choices)
    
    def adjust_plan_for_preferences(
        self,
        plan: DailyMealPlan,
        likes: List[str],
        dislikes: List[str]
    ) -> DailyMealPlan:
        """
        Adjust a meal plan based on user feedback.
        
        Args:
            plan: Existing meal plan
            likes: Foods user likes
            dislikes: Foods user dislikes
        
        Returns:
            Adjusted meal plan
        """
        # Replace disliked foods
        for meal in plan.meals:
            new_items = []
            for item in meal.items:
                if any(dislike.lower() in item.food.name.lower() for dislike in dislikes):
                    # Find replacement
                    replacement = self._find_food_for_meal(
                        item.food.category or "protein",
                        set(d.lower() for d in dislikes),
                        item.scaled_nutrients["calories"] * 1.2
                    )
                    if replacement:
                        new_items.append(MealItem(
                            food=replacement,
                            serving_size=item.serving_size,
                            serving_unit=item.serving_unit
                        ))
                    else:
                        new_items.append(item)
                else:
                    new_items.append(item)
            
            meal.items = new_items
        
        return plan


# ═══════════════════════════════════════════════════════════════════════════════
# MEAL LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MealLog:
    """Logged meal entry."""
    log_id: str
    user_id: str
    meal_type: MealType
    timestamp: datetime
    items: List[MealItem] = field(default_factory=list)
    notes: Optional[str] = None
    
    @property
    def total_calories(self) -> float:
        return sum(item.scaled_nutrients["calories"] for item in self.items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "user_id": self.user_id,
            "meal_type": self.meal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "items": [item.to_dict() for item in self.items],
            "total_calories": round(self.total_calories, 1),
            "notes": self.notes,
        }


class MealLogger:
    """
    Logs and tracks user meals.
    
    Features:
    - Log meals with nutrition calculation
    - Daily/weekly summaries
    - Progress tracking vs RDA goals
    """
    
    def __init__(self):
        """Initialize meal logger."""
        self.logs: Dict[str, List[MealLog]] = {}  # user_id -> logs
    
    def log_meal(
        self,
        user_id: str,
        meal_type: MealType,
        items: List[MealItem],
        notes: Optional[str] = None
    ) -> MealLog:
        """
        Log a meal.
        
        Args:
            user_id: User ID
            meal_type: Type of meal
            items: List of food items
            notes: Optional notes
        
        Returns:
            Created MealLog
        """
        import uuid
        
        log = MealLog(
            log_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            meal_type=meal_type,
            timestamp=datetime.now(),
            items=items,
            notes=notes
        )
        
        if user_id not in self.logs:
            self.logs[user_id] = []
        
        self.logs[user_id].append(log)
        
        return log
    
    def get_daily_summary(
        self,
        user_id: str,
        target_date: Optional[date] = None,
        rda: Optional[RDAProfile] = None
    ) -> Dict[str, Any]:
        """
        Get nutrition summary for a day.
        
        Args:
            user_id: User ID
            target_date: Date to summarize (default today)
            rda: RDA profile for percentage calculation
        
        Returns:
            Daily summary dict
        """
        target_date = target_date or date.today()
        rda = rda or RDAProfile()
        
        user_logs = self.logs.get(user_id, [])
        
        # Filter to target date
        day_logs = [
            log for log in user_logs
            if log.timestamp.date() == target_date
        ]
        
        # Calculate totals
        totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbohydrates": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
        }
        
        meals_by_type = {}
        
        for log in day_logs:
            meal_type = log.meal_type.value
            if meal_type not in meals_by_type:
                meals_by_type[meal_type] = {
                    "time": log.timestamp.strftime("%I:%M %p"),
                    "calories": 0.0,
                    "items": []
                }
            
            for item in log.items:
                nutrients = item.scaled_nutrients
                totals["calories"] += nutrients["calories"]
                totals["protein"] += nutrients["protein"]
                totals["carbohydrates"] += nutrients["carbohydrates"]
                totals["fat"] += nutrients["fat"]
                totals["fiber"] += nutrients.get("fiber", 0)
                
                meals_by_type[meal_type]["calories"] += nutrients["calories"]
                meals_by_type[meal_type]["items"].append(item.food.name)
        
        return {
            "user_id": user_id,
            "date": target_date.isoformat(),
            "totals": {k: round(v, 1) for k, v in totals.items()},
            "goals": {
                "calories": rda.calories,
                "protein": rda.protein,
                "carbohydrates": rda.carbohydrates,
                "fat": rda.fat,
            },
            "completion": {
                "calories": round((totals["calories"] / rda.calories) * 100, 1),
                "protein": round((totals["protein"] / rda.protein) * 100, 1),
            },
            "meals": meals_by_type,
            "meal_count": len(day_logs),
        }
    
    def get_weekly_summary(
        self,
        user_id: str,
        rda: Optional[RDAProfile] = None
    ) -> Dict[str, Any]:
        """Get weekly nutrition summary."""
        rda = rda or RDAProfile()
        
        daily_summaries = []
        for i in range(7):
            day = date.today() - timedelta(days=i)
            summary = self.get_daily_summary(user_id, day, rda)
            daily_summaries.append(summary)
        
        # Calculate averages
        avg_calories = sum(s["totals"]["calories"] for s in daily_summaries) / 7
        avg_protein = sum(s["totals"]["protein"] for s in daily_summaries) / 7
        
        return {
            "user_id": user_id,
            "week_ending": date.today().isoformat(),
            "daily_summaries": daily_summaries,
            "averages": {
                "calories": round(avg_calories, 1),
                "protein": round(avg_protein, 1),
            },
            "goals_met_days": sum(
                1 for s in daily_summaries 
                if s["completion"]["calories"] >= 90
            ),
        }
    
    def get_daily_log(self, user_id: str, target_date: str) -> Dict[str, Any]:
        """
        Get all meals logged for a specific day.
        
        Args:
            user_id: User ID
            target_date: Date string (YYYY-MM-DD)
        
        Returns:
            Dict with totals and meals list
        """
        from datetime import datetime
        
        try:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            date_obj = date.today()
        
        user_logs = self.logs.get(user_id, [])
        
        # Filter to target date
        day_logs = [
            log for log in user_logs
            if log.timestamp.date() == date_obj
        ]
        
        # Calculate totals
        totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbohydrates": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
        }
        
        meals = []
        
        for log in day_logs:
            meal_data = {
                "type": log.meal_type.value.capitalize(),
                "name": "",
                "calories": 0,
                "time": log.timestamp.strftime("%I:%M %p"),
            }
            
            item_names = []
            for item in log.items:
                nutrients = item.scaled_nutrients
                totals["calories"] += nutrients["calories"]
                totals["protein"] += nutrients["protein"]
                totals["carbohydrates"] += nutrients["carbohydrates"]
                totals["fat"] += nutrients["fat"]
                totals["fiber"] += nutrients.get("fiber", 0)
                meal_data["calories"] += int(nutrients["calories"])
                item_names.append(item.food.name)
            
            meal_data["name"] = ", ".join(item_names) if item_names else "Logged meal"
            meals.append(meal_data)
        
        return {
            "totals": {k: round(v, 1) for k, v in totals.items()},
            "meals": meals,
        }
    
    def log_hydration(
        self,
        user_id: str,
        amount_ml: int,
        beverage_type: str = "water"
    ) -> None:
        """
        Log hydration/fluid intake.
        
        Args:
            user_id: User ID
            amount_ml: Amount in milliliters
            beverage_type: Type of beverage
        """
        if not hasattr(self, 'hydration_logs'):
            self.hydration_logs: Dict[str, List[Dict]] = {}
        
        if user_id not in self.hydration_logs:
            self.hydration_logs[user_id] = []
        
        self.hydration_logs[user_id].append({
            "timestamp": datetime.now(),
            "amount_ml": amount_ml,
            "beverage_type": beverage_type,
        })
    
    def get_hydration_log(self, user_id: str, target_date: str) -> Dict[str, Any]:
        """
        Get hydration log for a specific day.
        
        Args:
            user_id: User ID
            target_date: Date string (YYYY-MM-DD)
        
        Returns:
            Dict with total_ml and log entries
        """
        if not hasattr(self, 'hydration_logs'):
            self.hydration_logs = {}
        
        try:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            date_obj = date.today()
        
        user_logs = self.hydration_logs.get(user_id, [])
        
        # Filter to target date
        day_logs = [
            log for log in user_logs
            if log["timestamp"].date() == date_obj
        ]
        
        total_ml = sum(log["amount_ml"] for log in day_logs)
        
        return {
            "total_ml": total_ml,
            "log": [
                {
                    "time": log["timestamp"].strftime("%I:%M %p"),
                    "amount_ml": log["amount_ml"],
                    "beverage_type": log["beverage_type"],
                }
                for log in day_logs
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

_generator: Optional[MealPlanGenerator] = None
_logger: Optional[MealLogger] = None

def get_meal_plan_generator() -> MealPlanGenerator:
    """Get or create meal plan generator singleton."""
    global _generator
    if _generator is None:
        _generator = MealPlanGenerator()
    return _generator

def get_meal_logger() -> MealLogger:
    """Get or create meal logger singleton."""
    global _logger
    if _logger is None:
        _logger = MealLogger()
    return _logger

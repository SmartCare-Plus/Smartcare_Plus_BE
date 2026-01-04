"""
SMARTCARE+ Nutrition Service - USDA FoodData Central Client

Owner: Kulasekara
Integration with USDA FoodData Central API for nutrition data.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NutrientInfo:
    """Single nutrient information."""
    name: str
    amount: float
    unit: str
    daily_value_percent: Optional[float] = None


@dataclass
class FoodItem:
    """Complete food item with nutrition data."""
    fdc_id: int
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    serving_size: float = 100.0
    serving_unit: str = "g"
    
    # Macronutrients
    calories: float = 0.0
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    
    # Additional nutrients
    sodium: float = 0.0
    cholesterol: float = 0.0
    saturated_fat: float = 0.0
    
    # Vitamins (as % daily value or mg)
    vitamin_a: float = 0.0
    vitamin_c: float = 0.0
    vitamin_d: float = 0.0
    vitamin_b12: float = 0.0
    
    # Minerals
    calcium: float = 0.0
    iron: float = 0.0
    potassium: float = 0.0
    magnesium: float = 0.0
    
    # Cost and budget - 1=low, 2=medium, 3=high
    cost_level: int = 2
    
    # All nutrients
    nutrients: List[NutrientInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "fdc_id": self.fdc_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
            "calories": self.calories,
            "protein": self.protein,
            "carbohydrates": self.carbohydrates,
            "fat": self.fat,
            "fiber": self.fiber,
            "sugar": self.sugar,
            "sodium": self.sodium,
            "cholesterol": self.cholesterol,
            "saturated_fat": self.saturated_fat,
            "vitamins": {
                "a": self.vitamin_a,
                "c": self.vitamin_c,
                "d": self.vitamin_d,
                "b12": self.vitamin_b12,
            },
            "minerals": {
                "calcium": self.calcium,
                "iron": self.iron,
                "potassium": self.potassium,
                "magnesium": self.magnesium,
            }
        }
    
    def scale_to_serving(self, serving_grams: float) -> "FoodItem":
        """Return a new FoodItem scaled to the specified serving size."""
        if self.serving_size == 0:
            return self
        
        scale = serving_grams / self.serving_size
        
        return FoodItem(
            fdc_id=self.fdc_id,
            name=self.name,
            brand=self.brand,
            category=self.category,
            serving_size=serving_grams,
            serving_unit=self.serving_unit,
            calories=self.calories * scale,
            protein=self.protein * scale,
            carbohydrates=self.carbohydrates * scale,
            fat=self.fat * scale,
            fiber=self.fiber * scale,
            sugar=self.sugar * scale,
            sodium=self.sodium * scale,
            cholesterol=self.cholesterol * scale,
            saturated_fat=self.saturated_fat * scale,
            vitamin_a=self.vitamin_a * scale,
            vitamin_c=self.vitamin_c * scale,
            vitamin_d=self.vitamin_d * scale,
            vitamin_b12=self.vitamin_b12 * scale,
            calcium=self.calcium * scale,
            iron=self.iron * scale,
            potassium=self.potassium * scale,
            magnesium=self.magnesium * scale,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# USDA API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class USDAClient:
    """
    Client for USDA FoodData Central API.
    
    API Documentation: https://fdc.nal.usda.gov/api-guide.html
    
    Features:
    - Food search by name
    - Get food details by FDC ID
    - Local caching to reduce API calls
    - Rate limiting handling
    """
    
    BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    
    # Nutrient ID mappings from USDA
    NUTRIENT_IDS = {
        "calories": 1008,      # Energy (kcal)
        "protein": 1003,       # Protein
        "carbohydrates": 1005, # Carbohydrate, by difference
        "fat": 1004,           # Total lipid (fat)
        "fiber": 1079,         # Fiber, total dietary
        "sugar": 2000,         # Sugars, total including NLEA
        "sodium": 1093,        # Sodium, Na
        "cholesterol": 1253,   # Cholesterol
        "saturated_fat": 1258, # Fatty acids, total saturated
        "vitamin_a": 1106,     # Vitamin A, RAE
        "vitamin_c": 1162,     # Vitamin C, total ascorbic acid
        "vitamin_d": 1114,     # Vitamin D (D2 + D3)
        "vitamin_b12": 1178,   # Vitamin B-12
        "calcium": 1087,       # Calcium, Ca
        "iron": 1089,          # Iron, Fe
        "potassium": 1092,     # Potassium, K
        "magnesium": 1090,     # Magnesium, Mg
    }
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize USDA client.
        
        Args:
            api_key: USDA API key (uses env var USDA_API_KEY if not provided)
            cache_dir: Directory for caching API responses
        """
        self.api_key = api_key or os.getenv("USDA_API_KEY", "DEMO_KEY")
        
        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "data" / "usda_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequent lookups
        self._memory_cache: Dict[int, FoodItem] = {}
        self._search_cache: Dict[str, List[Dict]] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests
    
    async def _make_request(
        self, 
        endpoint: str, 
        method: str = "GET",
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to USDA API with rate limiting."""
        # Rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Add API key to params
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        
        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, params=params) as response:
                        self._last_request_time = asyncio.get_event_loop().time()
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:
                            # Rate limited - wait and retry
                            await asyncio.sleep(1.0)
                            return await self._make_request(endpoint, method, params, json_data)
                        else:
                            return {"error": f"API error: {response.status}"}
                
                elif method == "POST":
                    async with session.post(url, params=params, json=json_data) as response:
                        self._last_request_time = asyncio.get_event_loop().time()
                        
                        if response.status == 200:
                            return await response.json()
                        else:
                            return {"error": f"API error: {response.status}"}
        
        except aiohttp.ClientError as e:
            return {"error": f"Network error: {str(e)}"}
    
    async def search_foods(
        self, 
        query: str, 
        page_size: int = 25,
        data_type: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for foods by name.
        
        Args:
            query: Search query string
            page_size: Number of results per page (max 50)
            data_type: Filter by data type (e.g., ["Survey (FNDDS)", "Branded"])
        
        Returns:
            List of search result dicts
        """
        # Check cache first
        cache_key = f"{query}_{page_size}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        
        params = {
            "query": query,
            "pageSize": min(page_size, 50),
        }
        
        if data_type:
            params["dataType"] = ",".join(data_type)
        
        result = await self._make_request("foods/search", params=params)
        
        if "error" in result:
            return []
        
        foods = result.get("foods", [])
        
        # Simplify results
        simplified = []
        for food in foods:
            simplified.append({
                "fdc_id": food.get("fdcId"),
                "name": food.get("description", "Unknown"),
                "brand": food.get("brandOwner", None),
                "category": food.get("foodCategory", None),
                "data_type": food.get("dataType", None),
            })
        
        # Cache results
        self._search_cache[cache_key] = simplified
        
        return simplified
    
    async def get_food(self, fdc_id: int) -> Optional[FoodItem]:
        """
        Get detailed food information by FDC ID.
        
        Args:
            fdc_id: USDA FoodData Central ID
        
        Returns:
            FoodItem with complete nutrition data or None
        """
        # Check memory cache
        if fdc_id in self._memory_cache:
            return self._memory_cache[fdc_id]
        
        # Check file cache
        cache_file = self.cache_dir / f"{fdc_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    food = self._parse_food_data(data)
                    self._memory_cache[fdc_id] = food
                    return food
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Fetch from API
        result = await self._make_request(f"food/{fdc_id}")
        
        if "error" in result:
            return None
        
        # Parse and cache
        food = self._parse_food_data(result)
        
        # Save to file cache
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except IOError:
            pass
        
        self._memory_cache[fdc_id] = food
        return food
    
    def _parse_food_data(self, data: Dict) -> FoodItem:
        """Parse USDA API response into FoodItem."""
        food = FoodItem(
            fdc_id=data.get("fdcId", 0),
            name=data.get("description", "Unknown"),
            brand=data.get("brandOwner"),
            category=data.get("foodCategory", {}).get("description") if isinstance(data.get("foodCategory"), dict) else data.get("foodCategory"),
        )
        
        # Extract serving size if available
        if "servingSize" in data:
            food.serving_size = data["servingSize"]
            food.serving_unit = data.get("servingSizeUnit", "g")
        
        # Extract nutrients
        nutrients = data.get("foodNutrients", [])
        
        for nutrient in nutrients:
            # Handle different API response formats
            if isinstance(nutrient.get("nutrient"), dict):
                nutrient_id = nutrient["nutrient"].get("id")
                nutrient_name = nutrient["nutrient"].get("name")
                nutrient_unit = nutrient["nutrient"].get("unitName", "")
            else:
                nutrient_id = nutrient.get("nutrientId")
                nutrient_name = nutrient.get("nutrientName", "")
                nutrient_unit = nutrient.get("unitName", "")
            
            amount = nutrient.get("amount", 0) or 0
            
            # Map to food item fields
            if nutrient_id == self.NUTRIENT_IDS["calories"]:
                food.calories = amount
            elif nutrient_id == self.NUTRIENT_IDS["protein"]:
                food.protein = amount
            elif nutrient_id == self.NUTRIENT_IDS["carbohydrates"]:
                food.carbohydrates = amount
            elif nutrient_id == self.NUTRIENT_IDS["fat"]:
                food.fat = amount
            elif nutrient_id == self.NUTRIENT_IDS["fiber"]:
                food.fiber = amount
            elif nutrient_id == self.NUTRIENT_IDS["sugar"]:
                food.sugar = amount
            elif nutrient_id == self.NUTRIENT_IDS["sodium"]:
                food.sodium = amount
            elif nutrient_id == self.NUTRIENT_IDS["cholesterol"]:
                food.cholesterol = amount
            elif nutrient_id == self.NUTRIENT_IDS["saturated_fat"]:
                food.saturated_fat = amount
            elif nutrient_id == self.NUTRIENT_IDS["vitamin_a"]:
                food.vitamin_a = amount
            elif nutrient_id == self.NUTRIENT_IDS["vitamin_c"]:
                food.vitamin_c = amount
            elif nutrient_id == self.NUTRIENT_IDS["vitamin_d"]:
                food.vitamin_d = amount
            elif nutrient_id == self.NUTRIENT_IDS["vitamin_b12"]:
                food.vitamin_b12 = amount
            elif nutrient_id == self.NUTRIENT_IDS["calcium"]:
                food.calcium = amount
            elif nutrient_id == self.NUTRIENT_IDS["iron"]:
                food.iron = amount
            elif nutrient_id == self.NUTRIENT_IDS["potassium"]:
                food.potassium = amount
            elif nutrient_id == self.NUTRIENT_IDS["magnesium"]:
                food.magnesium = amount
            
            # Store all nutrients
            food.nutrients.append(NutrientInfo(
                name=nutrient_name,
                amount=amount,
                unit=nutrient_unit
            ))
        
        return food
    
    async def get_multiple_foods(self, fdc_ids: List[int]) -> List[FoodItem]:
        """
        Get multiple foods by their FDC IDs.
        
        Args:
            fdc_ids: List of USDA FoodData Central IDs
        
        Returns:
            List of FoodItems
        """
        # Batch API call
        result = await self._make_request(
            "foods",
            method="POST",
            json_data={"fdcIds": fdc_ids}
        )
        
        if "error" in result or not isinstance(result, list):
            # Fallback to individual requests
            foods = []
            for fdc_id in fdc_ids:
                food = await self.get_food(fdc_id)
                if food:
                    foods.append(food)
            return foods
        
        return [self._parse_food_data(data) for data in result]
    
    def clear_cache(self):
        """Clear all caches."""
        self._memory_cache.clear()
        self._search_cache.clear()
        
        # Optionally clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except IOError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL FOOD DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class LocalFoodDatabase:
    """
    Local food database for offline operation and faster lookups.
    
    Uses JSON file for storage with common elderly-friendly foods.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize local database.
        
        Args:
            data_path: Path to foods.json file
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).parent.parent / "data" / "foods.json"
        
        self.foods: Dict[str, FoodItem] = {}
        self._load_database()
    
    def _load_database(self):
        """Load foods from JSON file."""
        if not self.data_path.exists():
            print(f"Warning: Food database not found at {self.data_path}")
            return
        
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both list format and dict format
            if isinstance(data, list):
                foods_list = data
            elif isinstance(data, dict):
                foods_list = data.get("foods", [])
            else:
                print(f"Warning: Unexpected data format in {self.data_path}")
                return
            
            for item in foods_list:
                # Handle Sri Lankan food format from foods.json
                food_id = item.get("food_id", item.get("id", 0))
                food_name = item.get("name", "Unknown")
                
                food = FoodItem(
                    fdc_id=hash(food_id) if isinstance(food_id, str) else food_id,
                    name=food_name,
                    category=item.get("cuisine", item.get("category")),
                    serving_size=100,  # Default serving size
                    serving_unit="g",
                    calories=item.get("calories_per_serving", item.get("calories", 0)),
                    protein=item.get("protein_g", item.get("protein", 0)),
                    carbohydrates=item.get("carbs_g", item.get("carbs", 0)),
                    fat=item.get("fat_g", item.get("fat", 0)),
                    fiber=item.get("fiber_g", item.get("fiber", 0)),
                    sugar=item.get("sugar", 0),
                    sodium=item.get("sodium_mg", item.get("sodium", 0)),
                    cost_level=item.get("cost_level", 2),  # 1=low, 2=medium, 3=high
                )
                
                # Store additional metadata
                food.food_id = food_id
                food.description = item.get("description", "")
                food.meal_types = item.get("meal_type", [])
                food.is_vegetarian = item.get("is_vegetarian", False)
                food.suitable_for_diabetes = item.get("suitable_for_diabetes", False)
                food.suitable_for_hypertension = item.get("suitable_for_hypertension", False)
                food.elderly_benefits = item.get("elderly_benefits", [])
                
                # Index by name (lowercase for searching)
                self.foods[food_name.lower()] = food
                
                # Also index by food_id if it's a string
                if isinstance(food_id, str):
                    self.foods[food_id.lower()] = food
            
            print(f"✅ Loaded {len(self.foods)} foods from local database")
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading food database: {e}")
    
    def search(self, query: str, limit: int = 10) -> List[FoodItem]:
        """
        Search local database by food name.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            List of matching FoodItems
        """
        query_lower = query.lower().strip()
        matches = []
        seen_names = set()  # Avoid duplicates
        
        for name, food in self.foods.items():
            if query_lower in name and food.name.lower() not in seen_names:
                matches.append(food)
                seen_names.add(food.name.lower())
        
        # Sort by relevance:
        # Priority: exact match > starts with > complete word match > partial match
        # Secondary: word position (earlier is better)
        # Tertiary: shorter names are better
        def relevance(food: FoodItem) -> tuple:
            name = food.name.lower()
            words = name.split()
            
            # Check if query is a complete word in the name
            is_complete_word = query_lower in words
            
            # Find position of word containing query
            word_position = 10  # Default high if not found
            for i, word in enumerate(words):
                if query_lower in word:
                    word_position = i
                    break
            
            # Exact match gets highest priority
            if name == query_lower:
                return (0, 0, 0, len(name))
            
            # Name starts with query (e.g., "Apple Pie" for "apple")
            if name.startswith(query_lower):
                return (1, 0, 0, len(name))
            
            # Complete word match vs partial (prefer "Apple" over "Pineapple")
            if is_complete_word:
                return (2, word_position, 0, len(name))
            
            # Partial match (query is part of a word)
            return (3, word_position, 1, len(name))
        
        matches.sort(key=relevance)
        
        return matches[:limit]
    
    def get_by_name(self, name: str) -> Optional[FoodItem]:
        """Get food by exact name match."""
        return self.foods.get(name.lower())
    
    def get_by_category(self, category: str) -> List[FoodItem]:
        """
        Get foods by category - maps abstract categories to food properties.
        
        Categories like 'protein', 'vegetables', 'grains', etc. are mapped
        to food properties like is_vegetarian, protein content, meal_type, etc.
        """
        category_lower = category.lower()
        results = []
        
        # Category to property mapping
        category_keywords = {
            'protein': ['fish', 'egg', 'chicken', 'beef', 'pork', 'dhal', 'lentil', 'bean', 'tofu'],
            'vegetables': ['vegetable', 'salad', 'sambol', 'curry', 'gotukola', 'mukunuwenna', 'kankun'],
            'grains': ['rice', 'bread', 'roti', 'string hoppers', 'hoppers', 'pittu', 'kiribath'],
            'fruits': ['fruit', 'banana', 'papaya', 'mango', 'pineapple', 'woodapple'],
            'dairy': ['milk', 'curd', 'yogurt', 'cheese'],
            'nuts': ['nuts', 'cashew', 'peanut', 'coconut sambol'],
            'breakfast': [],  # Will check meal_type
            'lunch': [],
            'dinner': [],
            'snack': [],
        }
        
        keywords = category_keywords.get(category_lower, [])
        
        for food in self.foods.values():
            food_name_lower = food.name.lower()
            food_desc = getattr(food, 'description', '').lower()
            meal_types = getattr(food, 'meal_types', [])
            
            # Check meal_type for meal categories
            if category_lower in ['breakfast', 'lunch', 'dinner']:
                if meal_types and category_lower in [m.lower() for m in meal_types]:
                    results.append(food)
                    continue
            
            if category_lower == 'snack':
                # Snacks are typically lower calorie or marked as snack
                if food.calories < 200 or 'snack' in food_name_lower:
                    results.append(food)
                    continue
            
            # Check keywords
            for keyword in keywords:
                if keyword in food_name_lower or keyword in food_desc:
                    results.append(food)
                    break
            
            # Special checks
            if category_lower == 'protein' and food.protein >= 10:
                if food not in results:
                    results.append(food)
            elif category_lower == 'vegetables':
                if getattr(food, 'is_vegetarian', False) and food.protein < 10:
                    if food not in results:
                        results.append(food)
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

_usda_client: Optional[USDAClient] = None
_local_db: Optional[LocalFoodDatabase] = None

def get_usda_client() -> USDAClient:
    """Get or create USDA API client singleton."""
    global _usda_client
    if _usda_client is None:
        _usda_client = USDAClient()
    return _usda_client

def get_local_food_db() -> LocalFoodDatabase:
    """Get or create local food database singleton."""
    global _local_db
    if _local_db is None:
        _local_db = LocalFoodDatabase()
    return _local_db

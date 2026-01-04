"""
SMARTCARE+ Database Seeder

Populates Firestore with demo data for the 3 services:
- Physio: Users with conditions, exercises
- Nutrition: Food items, meal data
- Guardian: Elderly profiles, geofences

Run with: python -m scripts.seed_database
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import our modules
from core.database import init_firebase, get_db, is_mock_mode


def seed_physio_data(db):
    """
    Seed physiotherapy data.
    Owner: Neelaka
    """
    print("\n🏃 Seeding Physio Service Data...")
    
    # ============================================
    # 1. Create sample user with condition
    # ============================================
    users_ref = db.collection('physio_users')
    
    user_data = {
        'user_id': 'user_neelaka_demo',
        'name': 'Mr. Perera',
        'age': 72,
        'condition': 'Arthritis',
        'severity': 'moderate',
        'mobility_score': 65,
        'fall_risk_level': 'medium',
        'last_assessment': datetime.utcnow().isoformat(),
        'created_at': datetime.utcnow().isoformat(),
        'notes': 'Mild knee arthritis, requires gentle exercises'
    }
    
    users_ref.document('user_neelaka_demo').set(user_data)
    print(f"  ✅ Created user: {user_data['name']} (Condition: {user_data['condition']})")
    
    # ============================================
    # 2. Create exercise library
    # ============================================
    exercises_ref = db.collection('exercises')
    
    exercises = [
        {
            'exercise_id': 'ex_leg_lifts',
            'name': 'Seated Leg Lifts',
            'category': 'lower_body',
            'difficulty': 'easy',
            'duration_seconds': 60,
            'repetitions': 10,
            'sets': 3,
            'description': 'While seated, slowly lift one leg straight out in front of you. Hold for 2 seconds, then lower.',
            'benefits': ['Strengthens quadriceps', 'Improves knee stability', 'Safe for arthritis'],
            'target_conditions': ['arthritis', 'knee_pain', 'general_mobility'],
            'instructions': [
                'Sit upright in a sturdy chair',
                'Keep your back straight against the chair',
                'Slowly lift your right leg until straight',
                'Hold for 2 seconds',
                'Lower slowly and repeat with left leg'
            ],
            'video_url': None,
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        },
        {
            'exercise_id': 'ex_sit_to_stand',
            'name': 'Sit-to-Stand',
            'category': 'functional',
            'difficulty': 'medium',
            'duration_seconds': 90,
            'repetitions': 8,
            'sets': 2,
            'description': 'Practice standing up from a seated position without using hands, then sitting back down slowly.',
            'benefits': ['Builds leg strength', 'Improves balance', 'Mimics daily activities'],
            'target_conditions': ['general_mobility', 'fall_prevention', 'arthritis'],
            'instructions': [
                'Sit at the edge of a sturdy chair',
                'Place feet flat on floor, hip-width apart',
                'Lean slightly forward',
                'Stand up using only your legs (use armrests if needed)',
                'Stand fully upright, pause',
                'Slowly lower yourself back to seated position'
            ],
            'video_url': None,
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        },
        {
            'exercise_id': 'ex_ankle_circles',
            'name': 'Ankle Circles',
            'category': 'flexibility',
            'difficulty': 'easy',
            'duration_seconds': 45,
            'repetitions': 10,
            'sets': 2,
            'description': 'Rotate your ankles in circles to improve flexibility and circulation.',
            'benefits': ['Improves ankle mobility', 'Enhances circulation', 'Reduces stiffness'],
            'target_conditions': ['arthritis', 'circulation_issues', 'general_mobility'],
            'instructions': [
                'Sit comfortably with feet slightly off the ground',
                'Rotate your right ankle clockwise 10 times',
                'Rotate your right ankle counter-clockwise 10 times',
                'Repeat with left ankle'
            ],
            'video_url': None,
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        }
    ]
    
    for exercise in exercises:
        exercises_ref.document(exercise['exercise_id']).set(exercise)
        print(f"  ✅ Created exercise: {exercise['name']}")
    
    # ============================================
    # 3. Create sample assessment history
    # ============================================
    assessments_ref = db.collection('physio_assessments')
    
    assessment_data = {
        'assessment_id': 'assess_001',
        'user_id': 'user_neelaka_demo',
        'type': 'tug_test',
        'score': 12.5,  # seconds
        'interpretation': 'moderate_risk',
        'notes': 'Slight hesitation when turning',
        'recorded_at': datetime.utcnow().isoformat(),
        'video_file_id': None
    }
    
    assessments_ref.document('assess_001').set(assessment_data)
    print(f"  ✅ Created TUG assessment: {assessment_data['score']}s")
    
    print("  ✅ Physio data seeding complete!")


def seed_nutrition_data(db):
    """
    Seed nutrition data.
    Owner: Kulasekara
    """
    print("\n🥗 Seeding Nutrition Service Data...")
    
    # ============================================
    # 1. Create food items database
    # ============================================
    foods_ref = db.collection('foods')
    
    foods = [
        {
            'food_id': 'food_salmon',
            'name': 'Grilled Salmon',
            'category': 'protein',
            'calories_per_100g': 208,
            'protein_g': 20,
            'carbs_g': 0,
            'fat_g': 13,
            'fiber_g': 0,
            'sodium_mg': 59,
            'vitamin_d_iu': 526,
            'omega3_g': 2.3,
            'elderly_benefits': ['Heart health', 'Brain function', 'Anti-inflammatory'],
            'allergens': ['fish'],
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        },
        {
            'food_id': 'food_spinach',
            'name': 'Fresh Spinach',
            'category': 'vegetable',
            'calories_per_100g': 23,
            'protein_g': 2.9,
            'carbs_g': 3.6,
            'fat_g': 0.4,
            'fiber_g': 2.2,
            'sodium_mg': 79,
            'vitamin_k_mcg': 483,
            'iron_mg': 2.7,
            'elderly_benefits': ['Bone health', 'Eye health', 'Iron for energy'],
            'allergens': [],
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        },
        {
            'food_id': 'food_oatmeal',
            'name': 'Oatmeal',
            'category': 'grain',
            'calories_per_100g': 68,
            'protein_g': 2.4,
            'carbs_g': 12,
            'fat_g': 1.4,
            'fiber_g': 1.7,
            'sodium_mg': 49,
            'elderly_benefits': ['Heart health', 'Digestive health', 'Steady energy'],
            'allergens': ['gluten'],
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        },
        {
            'food_id': 'food_greek_yogurt',
            'name': 'Greek Yogurt',
            'category': 'dairy',
            'calories_per_100g': 97,
            'protein_g': 9,
            'carbs_g': 3.6,
            'fat_g': 5,
            'fiber_g': 0,
            'calcium_mg': 100,
            'elderly_benefits': ['Bone strength', 'Gut health', 'Protein for muscles'],
            'allergens': ['dairy'],
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        },
        {
            'food_id': 'food_banana',
            'name': 'Banana',
            'category': 'fruit',
            'calories_per_100g': 89,
            'protein_g': 1.1,
            'carbs_g': 23,
            'fat_g': 0.3,
            'fiber_g': 2.6,
            'potassium_mg': 358,
            'elderly_benefits': ['Heart health', 'Muscle function', 'Easy to digest'],
            'allergens': [],
            'image_url': None,
            'created_at': datetime.utcnow().isoformat()
        }
    ]
    
    for food in foods:
        foods_ref.document(food['food_id']).set(food)
        print(f"  ✅ Created food: {food['name']}")
    
    # ============================================
    # 2. Create sample meal plan
    # ============================================
    meal_plans_ref = db.collection('meal_plans')
    
    meal_plan = {
        'plan_id': 'plan_demo_001',
        'user_id': 'user_kulasekara_demo',
        'name': 'Heart-Healthy Weekly Plan',
        'goal': 'cardiovascular_health',
        'daily_calories_target': 1800,
        'created_at': datetime.utcnow().isoformat(),
        'meals': {
            'breakfast': {
                'foods': ['food_oatmeal', 'food_banana'],
                'time': '08:00'
            },
            'lunch': {
                'foods': ['food_salmon', 'food_spinach'],
                'time': '12:30'
            },
            'dinner': {
                'foods': ['food_greek_yogurt'],
                'time': '18:00'
            }
        }
    }
    
    meal_plans_ref.document('plan_demo_001').set(meal_plan)
    print(f"  ✅ Created meal plan: {meal_plan['name']}")
    
    # ============================================
    # 3. Create hydration tracking
    # ============================================
    hydration_ref = db.collection('hydration_logs')
    
    hydration_log = {
        'log_id': 'hydration_demo_001',
        'user_id': 'user_kulasekara_demo',
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'goal_ml': 2000,
        'consumed_ml': 1200,
        'entries': [
            {'time': '08:00', 'amount_ml': 250, 'type': 'water'},
            {'time': '10:30', 'amount_ml': 200, 'type': 'tea'},
            {'time': '12:30', 'amount_ml': 300, 'type': 'water'},
            {'time': '15:00', 'amount_ml': 250, 'type': 'water'},
            {'time': '17:00', 'amount_ml': 200, 'type': 'juice'}
        ]
    }
    
    hydration_ref.document('hydration_demo_001').set(hydration_log)
    print(f"  ✅ Created hydration log: {hydration_log['consumed_ml']}ml / {hydration_log['goal_ml']}ml")
    
    print("  ✅ Nutrition data seeding complete!")


def seed_guardian_data(db):
    """
    Seed guardian/monitoring data.
    Owner: Madhushani
    """
    print("\n🛡️ Seeding Guardian Service Data...")
    
    # ============================================
    # 1. Create elderly profile
    # ============================================
    elderly_ref = db.collection('elderly_profiles')
    
    elderly_data = {
        'elderly_id': 'elder_madhushani_demo',
        'name': 'Mrs. Silva',
        'age': 78,
        'status': 'safe',
        'last_seen': datetime.utcnow().isoformat(),
        'emergency_contacts': [
            {
                'name': 'Daughter - Nimali',
                'phone': '+94771234567',
                'relationship': 'daughter',
                'is_primary': True
            }
        ],
        'health_conditions': ['hypertension', 'diabetes_type2'],
        'mobility_level': 'walks_with_cane',
        'created_at': datetime.utcnow().isoformat(),
        'location': {
            'latitude': 6.9271,  # Colombo coordinates
            'longitude': 79.8612,
            'last_updated': datetime.utcnow().isoformat()
        }
    }
    
    elderly_ref.document('elder_madhushani_demo').set(elderly_data)
    print(f"  ✅ Created elderly profile: {elderly_data['name']} (Status: {elderly_data['status']})")
    
    # ============================================
    # 2. Create geofence zones
    # ============================================
    geofences_ref = db.collection('geofences')
    
    geofence_data = {
        'geofence_id': 'geo_home_001',
        'elderly_id': 'elder_madhushani_demo',
        'name': 'Home Zone',
        'type': 'safe_zone',
        'center': {
            'latitude': 6.9271,
            'longitude': 79.8612
        },
        'radius_meters': 100,
        'is_active': True,
        'alert_on_exit': True,
        'alert_on_enter': False,
        'created_at': datetime.utcnow().isoformat()
    }
    
    geofences_ref.document('geo_home_001').set(geofence_data)
    print(f"  ✅ Created geofence: {geofence_data['name']} (Radius: {geofence_data['radius_meters']}m)")
    
    # ============================================
    # 3. Create guardian account
    # ============================================
    guardians_ref = db.collection('guardians')
    
    guardian_data = {
        'guardian_id': 'guardian_demo_001',
        'name': 'Nimali Silva',
        'email': 'nimali@example.com',
        'phone': '+94771234567',
        'linked_elderly': ['elder_madhushani_demo'],
        'notification_preferences': {
            'push_enabled': True,
            'sms_enabled': True,
            'email_enabled': True,
            'alert_types': ['fall', 'sos', 'geofence_breach', 'inactivity']
        },
        'created_at': datetime.utcnow().isoformat()
    }
    
    guardians_ref.document('guardian_demo_001').set(guardian_data)
    print(f"  ✅ Created guardian: {guardian_data['name']}")
    
    # ============================================
    # 4. Create sample alert history
    # ============================================
    alerts_ref = db.collection('alerts')
    
    # Create a resolved alert from yesterday
    alert_data = {
        'alert_id': 'alert_demo_001',
        'elderly_id': 'elder_madhushani_demo',
        'guardian_id': 'guardian_demo_001',
        'type': 'inactivity',
        'severity': 'low',
        'status': 'resolved',
        'message': 'No movement detected for 2 hours',
        'created_at': (datetime.utcnow() - timedelta(days=1)).isoformat(),
        'resolved_at': (datetime.utcnow() - timedelta(days=1, hours=-1)).isoformat(),
        'resolution_notes': 'Elderly was napping, confirmed safe'
    }
    
    alerts_ref.document('alert_demo_001').set(alert_data)
    print(f"  ✅ Created sample alert: {alert_data['type']} ({alert_data['status']})")
    
    print("  ✅ Guardian data seeding complete!")


def main():
    """Main seeder function."""
    print("=" * 60)
    print("🌱 SMARTCARE+ Database Seeder")
    print("=" * 60)
    
    # Initialize Firebase
    print("\n📡 Connecting to Firebase...")
    connected = init_firebase()
    
    if not connected:
        if is_mock_mode():
            print("⚠️ Running in MOCK MODE - data will not be persisted!")
            print("   To connect to Firestore, add your service-account.json")
            return
        else:
            print("❌ Failed to connect to Firebase")
            return
    
    # Get database client
    db = get_db()
    
    if db is None:
        print("❌ Database client is None")
        return
    
    print("✅ Connected to Firestore!")
    
    # Seed all services
    try:
        seed_physio_data(db)
        seed_nutrition_data(db)
        seed_guardian_data(db)
        
        print("\n" + "=" * 60)
        print("🎉 DATABASE SEEDING COMPLETE!")
        print("=" * 60)
        print("\nCreated data summary:")
        print("  • Physio: 1 user, 3 exercises, 1 assessment")
        print("  • Nutrition: 5 foods, 1 meal plan, 1 hydration log")
        print("  • Guardian: 1 elderly profile, 1 geofence, 1 guardian, 1 alert")
        print("\nYou can now view this data in Firebase Console!")
        print("https://console.firebase.google.com/project/smartcare-plus-c9617/firestore")
        
    except Exception as e:
        print(f"\n❌ Seeding failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

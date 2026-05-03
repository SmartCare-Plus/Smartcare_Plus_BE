"""
SMARTCARE+ Physio Service - Patient Profile & BMI Module

Owner: Neelaka
Comprehensive patient data collection for personalized exercise plans.

Collects:
- Personal info (age, weight, height) → BMI calculation
- Medical history (arthritis type, severity, affected joints)
- Mobility assessment
- Pain tolerance baseline
- Daily activity level
- Lifestyle factors
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum
import logging
import sys
import math


def _setup_logger(name: str) -> logging.Logger:
    """Configure logger at DEBUG level."""
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        _logger.addHandler(handler)
    return _logger

logger = _setup_logger("smartcare.physio.patient")


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS FOR PATIENT DATA
# ══════════════════════════════════════════════════════════════════════════════

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ArthritisType(Enum):
    """Types of arthritis affecting elderly patients."""
    OSTEOARTHRITIS = "osteoarthritis"           # Most common, wear-and-tear
    RHEUMATOID = "rheumatoid_arthritis"         # Autoimmune
    PSORIATIC = "psoriatic_arthritis"           # Related to psoriasis
    GOUT = "gout"                                # Uric acid crystals
    NONE = "none"                                # No arthritis diagnosed
    OTHER = "other"


class ArthritisSeverity(Enum):
    """OARSI-based severity levels."""
    NONE = "none"
    MILD = "mild"           # Grade 1-2: Minimal symptoms
    MODERATE = "moderate"   # Grade 3: Significant symptoms
    SEVERE = "severe"       # Grade 4: Severe symptoms, may need surgery


class JointLocation(Enum):
    """Body joints affected by arthritis."""
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LOWER_BACK = "lower_back"
    NECK = "neck"
    FINGERS = "fingers"


class MobilityLevel(Enum):
    """Patient's general mobility status."""
    INDEPENDENT = "independent"       # Can move freely without assistance
    MINIMAL_ASSIST = "minimal_assist" # Occasional support needed
    MODERATE_ASSIST = "moderate_assist"  # Regular assistance needed
    DEPENDENT = "dependent"           # Requires constant assistance
    WHEELCHAIR = "wheelchair"         # Uses wheelchair


class ActivityLevel(Enum):
    """Daily physical activity level."""
    SEDENTARY = "sedentary"           # Mostly sitting, minimal movement
    LIGHTLY_ACTIVE = "lightly_active" # Light household tasks
    MODERATELY_ACTIVE = "moderately_active"  # Regular walking, some exercise
    ACTIVE = "active"                 # Regular exercise routine


class PainToleranceLevel(Enum):
    """Self-reported pain tolerance baseline."""
    LOW = "low"           # Highly sensitive to pain
    MODERATE = "moderate" # Average pain tolerance
    HIGH = "high"         # High pain tolerance


class BMICategory(Enum):
    """WHO BMI categories."""
    UNDERWEIGHT = "underweight"       # < 18.5
    NORMAL = "normal"                 # 18.5 - 24.9
    OVERWEIGHT = "overweight"         # 25.0 - 29.9
    OBESE_CLASS_1 = "obese_class_1"   # 30.0 - 34.9
    OBESE_CLASS_2 = "obese_class_2"   # 35.0 - 39.9
    OBESE_CLASS_3 = "obese_class_3"   # >= 40


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BMIResult:
    """BMI calculation result with health implications."""
    bmi: float
    category: BMICategory
    health_risk: str
    weight_status: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bmi": round(self.bmi, 1),
            "category": self.category.value,
            "health_risk": self.health_risk,
            "weight_status": self.weight_status,
            "recommendations": self.recommendations,
        }


@dataclass
class AffectedJoint:
    """Details about an affected joint."""
    location: JointLocation
    severity: ArthritisSeverity
    pain_level: int  # 0-10 scale
    stiffness_level: int  # 0-10 scale
    range_of_motion_percent: int  # 0-100% of normal
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location.value,
            "severity": self.severity.value,
            "pain_level": self.pain_level,
            "stiffness_level": self.stiffness_level,
            "range_of_motion_percent": self.range_of_motion_percent,
            "notes": self.notes,
        }


@dataclass
class MedicalHistory:
    """Patient's medical history relevant to physiotherapy."""
    arthritis_type: ArthritisType = ArthritisType.NONE
    arthritis_severity: ArthritisSeverity = ArthritisSeverity.NONE
    affected_joints: List[AffectedJoint] = field(default_factory=list)
    
    # Other conditions
    has_osteoporosis: bool = False
    has_cardiovascular_issues: bool = False
    has_balance_issues: bool = False
    has_vision_problems: bool = False
    has_hearing_problems: bool = False
    
    # Fall history
    falls_last_year: int = 0
    fear_of_falling: bool = False
    
    # Medications affecting exercise
    on_blood_thinners: bool = False
    on_pain_medication: bool = False
    on_steroids: bool = False
    
    # Previous physiotherapy
    previous_physiotherapy: bool = False
    physiotherapy_notes: str = ""
    
    # Surgery history
    joint_replacements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "arthritis_type": self.arthritis_type.value,
            "arthritis_severity": self.arthritis_severity.value,
            "affected_joints": [j.to_dict() for j in self.affected_joints],
            "has_osteoporosis": self.has_osteoporosis,
            "has_cardiovascular_issues": self.has_cardiovascular_issues,
            "has_balance_issues": self.has_balance_issues,
            "has_vision_problems": self.has_vision_problems,
            "has_hearing_problems": self.has_hearing_problems,
            "falls_last_year": self.falls_last_year,
            "fear_of_falling": self.fear_of_falling,
            "on_blood_thinners": self.on_blood_thinners,
            "on_pain_medication": self.on_pain_medication,
            "on_steroids": self.on_steroids,
            "previous_physiotherapy": self.previous_physiotherapy,
            "physiotherapy_notes": self.physiotherapy_notes,
            "joint_replacements": self.joint_replacements,
        }


@dataclass
class LifestyleFactors:
    """Patient lifestyle factors affecting exercise prescription."""
    activity_level: ActivityLevel = ActivityLevel.SEDENTARY
    mobility_level: MobilityLevel = MobilityLevel.INDEPENDENT
    
    # Living situation
    lives_alone: bool = False
    has_caregiver: bool = False
    caregiver_available_during_exercise: bool = False
    
    # Exercise environment
    has_exercise_space: bool = True
    has_chair_for_support: bool = True
    has_wall_for_support: bool = True
    
    # Sleep and energy
    avg_sleep_hours: float = 7.0
    best_exercise_time: str = "morning"  # morning, afternoon, evening
    
    # Motivation
    exercise_motivation: int = 5  # 1-10 scale
    adherence_confidence: int = 5  # 1-10 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_level": self.activity_level.value,
            "mobility_level": self.mobility_level.value,
            "lives_alone": self.lives_alone,
            "has_caregiver": self.has_caregiver,
            "caregiver_available_during_exercise": self.caregiver_available_during_exercise,
            "has_exercise_space": self.has_exercise_space,
            "has_chair_for_support": self.has_chair_for_support,
            "has_wall_for_support": self.has_wall_for_support,
            "avg_sleep_hours": self.avg_sleep_hours,
            "best_exercise_time": self.best_exercise_time,
            "exercise_motivation": self.exercise_motivation,
            "adherence_confidence": self.adherence_confidence,
        }


@dataclass
class PatientProfile:
    """Complete patient profile for personalized exercise planning."""
    
    # Identity
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Personal info
    first_name: str = ""
    last_name: str = ""
    date_of_birth: Optional[date] = None
    gender: Gender = Gender.OTHER
    
    # Physical measurements
    height_cm: float = 0.0
    weight_kg: float = 0.0
    
    # Contact
    emergency_contact_name: str = ""
    emergency_contact_phone: str = ""
    
    # Health data
    medical_history: MedicalHistory = field(default_factory=MedicalHistory)
    lifestyle: LifestyleFactors = field(default_factory=LifestyleFactors)
    pain_tolerance: PainToleranceLevel = PainToleranceLevel.MODERATE
    
    # Baseline assessments (0-10)
    baseline_pain_level: int = 0
    baseline_fatigue_level: int = 0
    baseline_mobility_score: int = 5
    
    # Goals
    primary_goal: str = "Maintain mobility and reduce pain"
    secondary_goals: List[str] = field(default_factory=list)
    
    # Profile completion
    profile_complete: bool = False
    
    @property
    def age(self) -> int:
        """Calculate age from date of birth."""
        if not self.date_of_birth:
            return 0
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def bmi(self) -> float:
        """Calculate BMI."""
        if self.height_cm <= 0 or self.weight_kg <= 0:
            return 0.0
        height_m = self.height_cm / 100
        return self.weight_kg / (height_m ** 2)
    
    def calculate_bmi_result(self) -> BMIResult:
        """Calculate BMI with category and recommendations."""
        return calculate_bmi(self.weight_kg, self.height_cm, self.age)
    
    def get_exercise_risk_level(self) -> str:
        """
        Determine exercise risk level based on profile.
        
        Returns: "low", "moderate", "high", or "very_high"
        """
        risk_score = 0
        
        # Age factor
        if self.age >= 80:
            risk_score += 3
        elif self.age >= 70:
            risk_score += 2
        elif self.age >= 60:
            risk_score += 1
        
        # BMI factor
        bmi = self.bmi
        if bmi >= 35:
            risk_score += 3
        elif bmi >= 30:
            risk_score += 2
        elif bmi >= 25 or bmi < 18.5:
            risk_score += 1
        
        # Medical conditions
        if self.medical_history.has_cardiovascular_issues:
            risk_score += 3
        if self.medical_history.has_osteoporosis:
            risk_score += 2
        if self.medical_history.has_balance_issues:
            risk_score += 2
        if self.medical_history.falls_last_year > 0:
            risk_score += self.medical_history.falls_last_year
        
        # Arthritis severity
        if self.medical_history.arthritis_severity == ArthritisSeverity.SEVERE:
            risk_score += 3
        elif self.medical_history.arthritis_severity == ArthritisSeverity.MODERATE:
            risk_score += 2
        elif self.medical_history.arthritis_severity == ArthritisSeverity.MILD:
            risk_score += 1
        
        # Mobility
        if self.lifestyle.mobility_level == MobilityLevel.DEPENDENT:
            risk_score += 4
        elif self.lifestyle.mobility_level == MobilityLevel.MODERATE_ASSIST:
            risk_score += 3
        elif self.lifestyle.mobility_level == MobilityLevel.MINIMAL_ASSIST:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 10:
            return "very_high"
        elif risk_score >= 6:
            return "high"
        elif risk_score >= 3:
            return "moderate"
        else:
            return "low"
    
    def get_affected_joint_locations(self) -> List[str]:
        """Get list of affected joint location values."""
        return [j.location.value for j in self.medical_history.affected_joints]
    
    def check_profile_completeness(self) -> Dict[str, bool]:
        """Check which sections are complete."""
        return {
            "personal_info": bool(self.first_name and self.date_of_birth),
            "physical_measurements": self.height_cm > 0 and self.weight_kg > 0,
            "medical_history": self.medical_history.arthritis_type != ArthritisType.NONE or len(self.medical_history.affected_joints) > 0,
            "lifestyle": self.lifestyle.activity_level is not None,
            "emergency_contact": bool(self.emergency_contact_name and self.emergency_contact_phone),
        }
    
    def is_physio_ready(self) -> bool:
        """Check if profile has enough info for exercise plan generation.
        
        Only requires physical measurements and lifestyle - not emergency contact.
        """
        return (
            self.height_cm > 0 and 
            self.weight_kg > 0 and 
            self.lifestyle.activity_level is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "first_name": self.first_name,
            "last_name": self.last_name,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "age": self.age,
            "gender": self.gender.value,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "bmi": round(self.bmi, 1),
            "bmi_result": self.calculate_bmi_result().to_dict() if self.height_cm > 0 and self.weight_kg > 0 else None,
            "emergency_contact_name": self.emergency_contact_name,
            "emergency_contact_phone": self.emergency_contact_phone,
            "medical_history": self.medical_history.to_dict(),
            "lifestyle": self.lifestyle.to_dict(),
            "pain_tolerance": self.pain_tolerance.value,
            "baseline_pain_level": self.baseline_pain_level,
            "baseline_fatigue_level": self.baseline_fatigue_level,
            "baseline_mobility_score": self.baseline_mobility_score,
            "primary_goal": self.primary_goal,
            "secondary_goals": self.secondary_goals,
            "exercise_risk_level": self.get_exercise_risk_level(),
            "profile_complete": self.profile_complete,
            "physio_ready": self.is_physio_ready(),
            "completeness": self.check_profile_completeness(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BMI CALCULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_bmi(weight_kg: float, height_cm: float, age: int = 0) -> BMIResult:
    """
    Calculate BMI with category and health recommendations.
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years (for age-specific recommendations)
    
    Returns:
        BMIResult with category and recommendations
    """
    if height_cm <= 0 or weight_kg <= 0:
        return BMIResult(
            bmi=0.0,
            category=BMICategory.NORMAL,
            health_risk="Unable to calculate",
            weight_status="Invalid measurements",
            recommendations=["Please provide valid height and weight"]
        )
    
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # Determine category (WHO standards)
    if bmi < 18.5:
        category = BMICategory.UNDERWEIGHT
        health_risk = "Increased"
        weight_status = "Underweight"
        recommendations = [
            "Consult healthcare provider about healthy weight gain",
            "Focus on nutrient-dense foods",
            "Strength training exercises recommended",
        ]
    elif bmi < 25:
        category = BMICategory.NORMAL
        health_risk = "Average"
        weight_status = "Normal weight"
        recommendations = [
            "Maintain current healthy weight",
            "Continue regular physical activity",
            "Balanced diet with adequate protein",
        ]
    elif bmi < 30:
        category = BMICategory.OVERWEIGHT
        health_risk = "Increased"
        weight_status = "Overweight"
        recommendations = [
            "Gradual weight loss of 0.5-1 kg per week recommended",
            "Low-impact exercises to protect joints",
            "Reduce processed foods and sugary drinks",
            "Monitor joint pain - excess weight stresses joints",
        ]
    elif bmi < 35:
        category = BMICategory.OBESE_CLASS_1
        health_risk = "High"
        weight_status = "Obese (Class I)"
        recommendations = [
            "Weight management crucial for joint health",
            "Water-based exercises recommended to reduce joint stress",
            "Consult dietitian for meal planning",
            "Start with seated exercises if mobility is limited",
        ]
    elif bmi < 40:
        category = BMICategory.OBESE_CLASS_2
        health_risk = "Very High"
        weight_status = "Obese (Class II)"
        recommendations = [
            "Medical supervision recommended during exercise",
            "Focus on chair-based and water exercises",
            "Prioritize weight loss for joint protection",
            "Consider weight management program",
        ]
    else:
        category = BMICategory.OBESE_CLASS_3
        health_risk = "Extremely High"
        weight_status = "Obese (Class III)"
        recommendations = [
            "Medical clearance required before exercise program",
            "Chair-based exercises only initially",
            "Close monitoring during physical activity",
            "Comprehensive weight management program recommended",
        ]
    
    # Age-specific adjustments for elderly (65+)
    if age >= 65:
        # For elderly, slightly higher BMI may be protective
        if 23 <= bmi <= 27:
            recommendations.insert(0, "BMI in healthy range for your age")
        recommendations.append("Balance exercises important to prevent falls")
        recommendations.append("Maintain muscle mass through protein intake")
    
    return BMIResult(
        bmi=bmi,
        category=category,
        health_risk=health_risk,
        weight_status=weight_status,
        recommendations=recommendations
    )


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT PROFILE STORE (Firestore-backed with in-memory cache)
# ══════════════════════════════════════════════════════════════════════════════

class PatientProfileStore:
    """Firestore-backed storage for patient profiles with in-memory cache."""
    
    COLLECTION_NAME = 'physio_profiles'
    
    def __init__(self):
        self._cache: Dict[str, PatientProfile] = {}
        self._db = None
        self._init_firestore()
    
    def _init_firestore(self):
        """Initialize Firestore connection."""
        try:
            from core.database import get_db
            self._db = get_db()
            if self._db:
                logger.info("📊 PatientProfileStore connected to Firestore")
            else:
                logger.warning("⚠️ Firestore not available, using in-memory only")
        except Exception as e:
            logger.warning(f"⚠️ Firestore init failed: {e}, using in-memory only")
    
    def _profile_to_dict(self, profile: PatientProfile) -> Dict[str, Any]:
        """Convert PatientProfile to Firestore-compatible dict."""
        data = asdict(profile)
        # Convert enums to strings
        if data.get('gender'):
            data['gender'] = data['gender'].value if hasattr(data['gender'], 'value') else str(data['gender'])
        if data.get('pain_tolerance'):
            data['pain_tolerance'] = data['pain_tolerance'].value if hasattr(data['pain_tolerance'], 'value') else str(data['pain_tolerance'])
        if data.get('date_of_birth'):
            data['date_of_birth'] = data['date_of_birth'].isoformat() if hasattr(data['date_of_birth'], 'isoformat') else str(data['date_of_birth'])
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat() if hasattr(data['created_at'], 'isoformat') else str(data['created_at'])
        if data.get('updated_at'):
            data['updated_at'] = data['updated_at'].isoformat() if hasattr(data['updated_at'], 'isoformat') else str(data['updated_at'])
        
        # Handle nested objects
        mh = data.get('medical_history', {})
        if mh:
            if mh.get('arthritis_type'):
                mh['arthritis_type'] = mh['arthritis_type'].value if hasattr(mh['arthritis_type'], 'value') else str(mh['arthritis_type'])
            if mh.get('arthritis_severity'):
                mh['arthritis_severity'] = mh['arthritis_severity'].value if hasattr(mh['arthritis_severity'], 'value') else str(mh['arthritis_severity'])
            if mh.get('affected_joints'):
                joints = []
                for joint in mh['affected_joints']:
                    j = dict(joint) if not isinstance(joint, dict) else joint
                    if j.get('location'):
                        j['location'] = j['location'].value if hasattr(j['location'], 'value') else str(j['location'])
                    if j.get('severity'):
                        j['severity'] = j['severity'].value if hasattr(j['severity'], 'value') else str(j['severity'])
                    joints.append(j)
                mh['affected_joints'] = joints
        
        ls = data.get('lifestyle', {})
        if ls:
            if ls.get('activity_level'):
                ls['activity_level'] = ls['activity_level'].value if hasattr(ls['activity_level'], 'value') else str(ls['activity_level'])
            if ls.get('mobility_level'):
                ls['mobility_level'] = ls['mobility_level'].value if hasattr(ls['mobility_level'], 'value') else str(ls['mobility_level'])
        
        return data
    
    def _dict_to_profile(self, user_id: str, data: Dict[str, Any]) -> PatientProfile:
        """Convert Firestore dict to PatientProfile."""
        profile = PatientProfile(user_id=user_id)
        
        # Simple fields
        for field in ['first_name', 'last_name', 'height_cm', 'weight_kg',
                      'emergency_contact_name', 'emergency_contact_phone',
                      'baseline_pain_level', 'baseline_fatigue_level', 'baseline_mobility_score',
                      'primary_goal', 'profile_complete']:
            if field in data and data[field] is not None:
                setattr(profile, field, data[field])
        
        # Enum fields
        if data.get('gender'):
            try:
                profile.gender = Gender(data['gender'])
            except: pass
        if data.get('pain_tolerance'):
            try:
                profile.pain_tolerance = PainToleranceLevel(data['pain_tolerance'])
            except: pass
        
        # Date fields
        if data.get('date_of_birth'):
            try:
                profile.date_of_birth = date.fromisoformat(data['date_of_birth'])
            except: pass
        if data.get('created_at'):
            try:
                profile.created_at = datetime.fromisoformat(data['created_at'])
            except: pass
        if data.get('updated_at'):
            try:
                profile.updated_at = datetime.fromisoformat(data['updated_at'])
            except: pass
        
        # Medical history
        mh_data = data.get('medical_history', {})
        if mh_data:
            mh = profile.medical_history
            if mh_data.get('arthritis_type'):
                try: mh.arthritis_type = ArthritisType(mh_data['arthritis_type'])
                except: pass
            if mh_data.get('arthritis_severity'):
                try: mh.arthritis_severity = ArthritisSeverity(mh_data['arthritis_severity'])
                except: pass
            for field in ['has_osteoporosis', 'has_cardiovascular_issues', 'has_balance_issues',
                          'has_vision_problems', 'has_hearing_problems', 'fear_of_falling',
                          'on_blood_thinners', 'on_pain_medication', 'on_steroids',
                          'previous_physiotherapy', 'falls_last_year', 'physiotherapy_notes']:
                if field in mh_data and mh_data[field] is not None:
                    setattr(mh, field, mh_data[field])
            if mh_data.get('affected_joints'):
                mh.affected_joints = []
                for j in mh_data['affected_joints']:
                    try:
                        joint = AffectedJoint(
                            location=JointLocation(j['location']),
                            severity=ArthritisSeverity(j.get('severity', 'mild')),
                            pain_level=j.get('pain_level', 0),
                            stiffness_level=j.get('stiffness_level', 0),
                            range_of_motion_percent=j.get('range_of_motion_percent', 100),
                            notes=j.get('notes', '')
                        )
                        mh.affected_joints.append(joint)
                    except: pass
        
        # Lifestyle
        ls_data = data.get('lifestyle', {})
        if ls_data:
            ls = profile.lifestyle
            if ls_data.get('activity_level'):
                try: ls.activity_level = ActivityLevel(ls_data['activity_level'])
                except: pass
            if ls_data.get('mobility_level'):
                try: ls.mobility_level = MobilityLevel(ls_data['mobility_level'])
                except: pass
            for field in ['lives_alone', 'has_caregiver', 'caregiver_available_during_exercise',
                          'has_exercise_space', 'has_chair_for_support', 'has_wall_for_support',
                          'avg_sleep_hours', 'best_exercise_time', 'exercise_motivation', 'adherence_confidence']:
                if field in ls_data and ls_data[field] is not None:
                    setattr(ls, field, ls_data[field])
        
        return profile
    
    def _save_to_firestore(self, user_id: str, profile: PatientProfile):
        """Save profile to Firestore."""
        if not self._db:
            return
        try:
            data = self._profile_to_dict(profile)
            self._db.collection(self.COLLECTION_NAME).document(user_id).set(data)
            logger.debug(f"💾 Saved profile to Firestore: {user_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save to Firestore: {e}")
    
    def _load_from_firestore(self, user_id: str) -> Optional[PatientProfile]:
        """Load profile from Firestore."""
        if not self._db:
            return None
        try:
            doc = self._db.collection(self.COLLECTION_NAME).document(user_id).get()
            if doc.exists:
                profile = self._dict_to_profile(user_id, doc.to_dict())
                logger.debug(f"📥 Loaded profile from Firestore: {user_id}")
                return profile
        except Exception as e:
            logger.error(f"❌ Failed to load from Firestore: {e}")
        return None
    
    def create_profile(self, user_id: str) -> PatientProfile:
        """Create a new empty profile for a user."""
        profile = PatientProfile(user_id=user_id)
        self._cache[user_id] = profile
        self._save_to_firestore(user_id, profile)
        logger.info(f"📋 Created new patient profile for user: {user_id}")
        return profile
    
    def get_profile(self, user_id: str) -> Optional[PatientProfile]:
        """Get profile for a user, returns None if not found."""
        # Check cache first
        if user_id in self._cache:
            return self._cache[user_id]
        
        # Try loading from Firestore
        profile = self._load_from_firestore(user_id)
        if profile:
            self._cache[user_id] = profile
            return profile
        
        return None
    
    def get_or_create_profile(self, user_id: str) -> PatientProfile:
        """Get existing profile or create new one."""
        profile = self.get_profile(user_id)
        if not profile:
            return self.create_profile(user_id)
        return profile
    
    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[PatientProfile]:
        """
        Update profile with provided data.
        
        Handles nested updates for medical_history and lifestyle.
        """
        profile = self.get_profile(user_id)
        if not profile:
            return None
        
        profile.updated_at = datetime.now()
        
        # Handle simple fields
        simple_fields = [
            'first_name', 'last_name', 'gender', 'height_cm', 'weight_kg',
            'emergency_contact_name', 'emergency_contact_phone',
            'pain_tolerance', 'baseline_pain_level', 'baseline_fatigue_level',
            'baseline_mobility_score', 'primary_goal', 'secondary_goals'
        ]
        
        for field in simple_fields:
            if field in updates:
                value = updates[field]
                # Convert enums
                if field == 'gender' and isinstance(value, str):
                    value = Gender(value)
                elif field == 'pain_tolerance' and isinstance(value, str):
                    value = PainToleranceLevel(value)
                setattr(profile, field, value)
        
        # Handle date_of_birth
        if 'date_of_birth' in updates:
            dob = updates['date_of_birth']
            if isinstance(dob, str):
                profile.date_of_birth = date.fromisoformat(dob)
            elif isinstance(dob, date):
                profile.date_of_birth = dob
        
        # Handle medical history updates
        if 'medical_history' in updates:
            self._update_medical_history(profile, updates['medical_history'])
        
        # Handle lifestyle updates
        if 'lifestyle' in updates:
            self._update_lifestyle(profile, updates['lifestyle'])
        
        # Check if profile is complete (fully complete) or physio-ready (enough for exercises)
        completeness = profile.check_profile_completeness()
        profile.profile_complete = all(completeness.values())
        
        # If physio-ready, mark as complete enough for exercise generation
        if not profile.profile_complete and profile.is_physio_ready():
            profile.profile_complete = True  # Mark as complete for practical purposes
        
        # Save to Firestore
        self._cache[user_id] = profile
        self._save_to_firestore(user_id, profile)
        
        logger.info(f"📋 Updated patient profile for user: {user_id}")
        return profile
    
    def _update_medical_history(self, profile: PatientProfile, updates: Dict[str, Any]):
        """Update medical history fields."""
        mh = profile.medical_history
        
        if 'arthritis_type' in updates:
            mh.arthritis_type = ArthritisType(updates['arthritis_type'])
        if 'arthritis_severity' in updates:
            mh.arthritis_severity = ArthritisSeverity(updates['arthritis_severity'])
        
        # Boolean fields
        bool_fields = [
            'has_osteoporosis', 'has_cardiovascular_issues', 'has_balance_issues',
            'has_vision_problems', 'has_hearing_problems', 'fear_of_falling',
            'on_blood_thinners', 'on_pain_medication', 'on_steroids',
            'previous_physiotherapy'
        ]
        for field in bool_fields:
            if field in updates:
                setattr(mh, field, updates[field])
        
        if 'falls_last_year' in updates:
            mh.falls_last_year = updates['falls_last_year']
        if 'physiotherapy_notes' in updates:
            mh.physiotherapy_notes = updates['physiotherapy_notes']
        if 'joint_replacements' in updates:
            mh.joint_replacements = updates['joint_replacements']
        
        # Handle affected joints
        if 'affected_joints' in updates:
            mh.affected_joints = []
            for joint_data in updates['affected_joints']:
                joint = AffectedJoint(
                    location=JointLocation(joint_data['location']),
                    severity=ArthritisSeverity(joint_data.get('severity', 'mild')),
                    pain_level=joint_data.get('pain_level', 0),
                    stiffness_level=joint_data.get('stiffness_level', 0),
                    range_of_motion_percent=joint_data.get('range_of_motion_percent', 100),
                    notes=joint_data.get('notes', '')
                )
                mh.affected_joints.append(joint)
    
    def _update_lifestyle(self, profile: PatientProfile, updates: Dict[str, Any]):
        """Update lifestyle fields."""
        ls = profile.lifestyle
        
        if 'activity_level' in updates:
            ls.activity_level = ActivityLevel(updates['activity_level'])
        if 'mobility_level' in updates:
            ls.mobility_level = MobilityLevel(updates['mobility_level'])
        
        bool_fields = [
            'lives_alone', 'has_caregiver', 'caregiver_available_during_exercise',
            'has_exercise_space', 'has_chair_for_support', 'has_wall_for_support'
        ]
        for field in bool_fields:
            if field in updates:
                setattr(ls, field, updates[field])
        
        if 'avg_sleep_hours' in updates:
            ls.avg_sleep_hours = updates['avg_sleep_hours']
        if 'best_exercise_time' in updates:
            ls.best_exercise_time = updates['best_exercise_time']
        if 'exercise_motivation' in updates:
            ls.exercise_motivation = updates['exercise_motivation']
        if 'adherence_confidence' in updates:
            ls.adherence_confidence = updates['adherence_confidence']
    
    def delete_profile(self, user_id: str) -> bool:
        """Delete a profile from cache and Firestore."""
        deleted = False
        
        # Delete from cache
        if user_id in self._cache:
            del self._cache[user_id]
            deleted = True
        
        # Delete from Firestore
        if self._db:
            try:
                self._db.collection(self.COLLECTION_NAME).document(user_id).delete()
                deleted = True
            except Exception as e:
                logger.error(f"❌ Failed to delete from Firestore: {e}")
        
        if deleted:
            logger.info(f"🗑️ Deleted patient profile for user: {user_id}")
        return deleted
    
    def list_profiles(self) -> List[str]:
        """List all user IDs with profiles (from Firestore and cache)."""
        user_ids = set(self._cache.keys())
        
        # Also get from Firestore
        if self._db:
            try:
                docs = self._db.collection(self.COLLECTION_NAME).stream()
                for doc in docs:
                    user_ids.add(doc.id)
            except Exception as e:
                logger.error(f"❌ Failed to list from Firestore: {e}")
        
        return list(user_ids)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

_profile_store_instance: Optional[PatientProfileStore] = None


def get_patient_profile_store() -> PatientProfileStore:
    """Get or create global patient profile store instance."""
    global _profile_store_instance
    if _profile_store_instance is None:
        _profile_store_instance = PatientProfileStore()
    return _profile_store_instance

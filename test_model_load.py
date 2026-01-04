"""Quick test script to load model from weights."""
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

from guardian_service.models.video_classifier import VideoClassifier

print("="*60)
print("Testing VideoClassifier Model Loading")
print("="*60)

vc = VideoClassifier()
print("\nLoading model...")
success = vc.load_model()

print(f"\n✅ Model loaded: {success}")
print(f"📊 Using fallback: {getattr(vc, '_use_fallback', True)}")

if vc.model is not None:
    print(f"\n🔧 Model Details:")
    print(f"   Input shape: {vc.model.input_shape}")
    print(f"   Output shape: {vc.model.output_shape}")
    print(f"   Parameters: {vc.model.count_params():,}")

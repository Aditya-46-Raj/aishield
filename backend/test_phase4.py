"""
Test script to verify Phase 4 anti-spoof and enhanced liveness detection
"""
import json
from pathlib import Path

def test_antispoof_detector():
    """Test standalone anti-spoof detector"""
    print("\n" + "="*70)
    print("TEST 1: Anti-Spoof Detector (Phase 4.1)")
    print("="*70)
    
    from antispoof_detector import AntiSpoofDetector
    
    detector = AntiSpoofDetector()
    video_path = Path("samples/video_sample.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    result = detector.predict_video(str(video_path))
    
    print(f"\n📹 Video: {video_path}")
    print(f"🖼️  Frames analyzed: {result['frame_count']}")
    print(f"✅ Realness probability: {result['realness_prob']:.3f}")
    print(f"⚠️  Spoof score: {result['spoof_score']:.3f}")
    print(f"📝 Explanation: {result['explanation']}")
    
    # Verify deliverables
    assert result['frame_count'] >= 5, "Should extract 5-10 frames"
    assert 0 <= result['realness_prob'] <= 1, "Probability should be 0-1"
    assert 0 <= result['spoof_score'] <= 1, "Spoof score should be 0-1"
    
    print("\n✅ Phase 4.1 VERIFIED: Model loaded and inference works!")
    return True

def test_enhanced_liveness():
    """Test enhanced liveness detection with multi-modal scoring"""
    print("\n" + "="*70)
    print("TEST 2: Enhanced Liveness Detection (Phase 4.2)")
    print("="*70)
    
    from models import analyze_video_liveness_v2
    
    video_path = Path("samples/video_sample.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    result = analyze_video_liveness_v2(str(video_path))
    
    print(f"\n📹 Video: {video_path}")
    print(f"🖼️  Total frames: {result['frames']}")
    print(f"👁️  Blink events: {result['blink_events']}")
    print(f"🏃 Motion events: {result['motion_events']}")
    print(f"\n📊 Component Scores:")
    print(f"   • Anti-spoof score: {result['components']['antispoof_score']:.3f}")
    print(f"   • Motion score: {result['components']['motion_score']:.3f}")
    print(f"   • Blink score: {result['components']['blink_score']:.3f}")
    print(f"\n🎯 Combined liveness score: {result['score']:.3f}")
    print(f"⚖️  Verdict: {result['verdict'].upper()}")
    print(f"📝 Explanation: {result['explanation']}")
    
    # Verify deliverables
    assert 'components' in result, "Should have component breakdown"
    assert 'antispoof_score' in result['components'], "Should have anti-spoof score"
    assert 'motion_score' in result['components'], "Should have motion score"
    assert 'blink_score' in result['components'], "Should have blink score"
    assert result['verdict'] in ['live', 'suspicious', 'spoofed', 'error', 'no_frames'], "Should have valid verdict"
    
    print("\n✅ Phase 4.2 VERIFIED: Multi-modal liveness detection works!")
    return True

def test_scoring_weights():
    """Verify scoring formula is correct"""
    print("\n" + "="*70)
    print("TEST 3: Scoring Formula Verification")
    print("="*70)
    
    from models import analyze_video_liveness_v2
    
    video_path = Path("samples/video_sample.mp4")
    result = analyze_video_liveness_v2(str(video_path))
    
    components = result['components']
    
    # Recalculate score manually
    expected_score = (
        0.5 * components['antispoof_score'] +
        0.3 * components['motion_score'] +
        0.2 * components['blink_score']
    )
    
    print(f"\n🧮 Manual calculation:")
    print(f"   0.5 × {components['antispoof_score']:.3f} (anti-spoof)")
    print(f" + 0.3 × {components['motion_score']:.3f} (motion)")
    print(f" + 0.2 × {components['blink_score']:.3f} (blink)")
    print(f" = {expected_score:.3f}")
    print(f"\n📊 Returned score: {result['score']:.3f}")
    print(f"✅ Match: {abs(expected_score - result['score']) < 0.001}")
    
    assert abs(expected_score - result['score']) < 0.001, "Score calculation mismatch!"
    
    print("\n✅ VERIFIED: 50% anti-spoof + 30% motion + 20% blink formula correct!")
    return True

if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("PHASE 4 VERIFICATION SUITE")
    print("Liveness & Anti-Spoof Detection")
    print("🚀" * 35)
    
    try:
        test1 = test_antispoof_detector()
        test2 = test_enhanced_liveness()
        test3 = test_scoring_weights()
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"✅ Phase 4.1 (Anti-Spoof Model): {'PASS' if test1 else 'FAIL'}")
        print(f"✅ Phase 4.2 (Enhanced Liveness): {'PASS' if test2 else 'FAIL'}")
        print(f"✅ Scoring Formula: {'PASS' if test3 else 'FAIL'}")
        
        if test1 and test2 and test3:
            print("\n🎉 ALL TESTS PASSED! Phase 4 is complete and verified! 🎉")
        else:
            print("\n❌ Some tests failed. Review output above.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

"""
Phase 4.4 Test: Combined Liveness Score
Tests the new formula: liveness_score = 0.5 * (1 - deepfake_score) + 0.5 * motion_score
"""
from models import analyze_video_liveness_v2
from pathlib import Path
import json

def format_json_output(result, video_name):
    """Format output as required by Phase 4.4"""
    output = {
        "video": video_name,
        "deepfake_score": result["deepfake_score"],
        "motion_score": result["motion_score"],
        "liveness_score": result["liveness_score"],
        "liveness_reason": result["liveness_reason"],
        "verdict": result["verdict"],
        "frames_analyzed": result["frames"],
        "blink_events": result["blink_events"],
        "motion_events": result["motion_events"],
        "components": result["components"]
    }
    return output

def test_clean_video():
    """Test on clean selfie video"""
    print("\n" + "="*70)
    print("PHASE 4.4 TEST 1: CLEAN SELFIE VIDEO")
    print("="*70)
    
    video_path = Path("samples/video_sample.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    result = analyze_video_liveness_v2(str(video_path))
    output = format_json_output(result, video_path.name)
    
    print("\n📹 VIDEO: video_sample.mp4 (Clean)")
    print("\n📊 FINAL JSON OUTPUT:")
    print(json.dumps(output, indent=2))
    
    # Verification
    print("\n✅ VERIFICATION:")
    print(f"   Deepfake score: {output['deepfake_score']:.3f} (lower = more real)")
    print(f"   Motion score: {output['motion_score']:.3f} (higher = more live-like)")
    print(f"   Liveness score: {output['liveness_score']:.3f} (higher = more live)")
    print(f"   Formula check: 0.5 × (1 - {output['deepfake_score']:.3f}) + 0.5 × {output['motion_score']:.3f}")
    
    manual_calc = 0.5 * (1 - output['deepfake_score']) + 0.5 * output['motion_score']
    print(f"   Manual calculation: {manual_calc:.3f}")
    print(f"   Match: {'✅ YES' if abs(manual_calc - output['liveness_score']) < 0.001 else '❌ NO'}")
    
    return output

def test_spoofed_video():
    """Test on spoofed/replayed video"""
    print("\n" + "="*70)
    print("PHASE 4.4 TEST 2: SPOOFED/REPLAYED VIDEO")
    print("="*70)
    
    video_path = Path("samples/video_spoofed.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    result = analyze_video_liveness_v2(str(video_path))
    output = format_json_output(result, video_path.name)
    
    print("\n📹 VIDEO: video_spoofed.mp4 (Spoofed)")
    print("\n📊 FINAL JSON OUTPUT:")
    print(json.dumps(output, indent=2))
    
    # Verification
    print("\n✅ VERIFICATION:")
    print(f"   Deepfake score: {output['deepfake_score']:.3f} (higher = more fake)")
    print(f"   Motion score: {output['motion_score']:.3f} (lower = less live-like)")
    print(f"   Liveness score: {output['liveness_score']:.3f} (lower = less live)")
    print(f"   Formula check: 0.5 × (1 - {output['deepfake_score']:.3f}) + 0.5 × {output['motion_score']:.3f}")
    
    manual_calc = 0.5 * (1 - output['deepfake_score']) + 0.5 * output['motion_score']
    print(f"   Manual calculation: {manual_calc:.3f}")
    print(f"   Match: {'✅ YES' if abs(manual_calc - output['liveness_score']) < 0.001 else '❌ NO'}")
    
    return output

def compare_results(clean_output, spoofed_output):
    """Compare clean vs spoofed results"""
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    print("\n📊 Score Comparison:")
    print(f"   {'Metric':<20} {'Clean':<15} {'Spoofed':<15} {'Delta':<10}")
    print("   " + "-"*60)
    print(f"   {'Deepfake Score':<20} {clean_output['deepfake_score']:<15.3f} {spoofed_output['deepfake_score']:<15.3f} {spoofed_output['deepfake_score'] - clean_output['deepfake_score']:<10.3f}")
    print(f"   {'Motion Score':<20} {clean_output['motion_score']:<15.3f} {spoofed_output['motion_score']:<15.3f} {spoofed_output['motion_score'] - clean_output['motion_score']:<10.3f}")
    print(f"   {'Liveness Score':<20} {clean_output['liveness_score']:<15.3f} {spoofed_output['liveness_score']:<15.3f} {spoofed_output['liveness_score'] - clean_output['liveness_score']:<10.3f}")
    
    print("\n🎯 Discrimination:")
    if clean_output['liveness_score'] > spoofed_output['liveness_score']:
        print(f"   ✅ SUCCESS: Clean video has higher liveness score")
        print(f"   Difference: {clean_output['liveness_score'] - spoofed_output['liveness_score']:.3f}")
    else:
        print(f"   ❌ FAILED: Spoofed video has higher liveness score")
    
    print("\n💬 Liveness Reasons:")
    print(f"   Clean: {clean_output['liveness_reason']}")
    print(f"   Spoofed: {spoofed_output['liveness_reason']}")

def save_outputs(clean_output, spoofed_output):
    """Save test outputs to JSON files"""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save clean sample
    clean_file = output_dir / "test_clean_sample.json"
    with open(clean_file, 'w') as f:
        json.dump(clean_output, f, indent=2)
    print(f"\n💾 Clean sample saved to: {clean_file}")
    
    # Save spoofed sample
    spoofed_file = output_dir / "test_forged_sample.json"
    with open(spoofed_file, 'w') as f:
        json.dump(spoofed_output, f, indent=2)
    print(f"💾 Spoofed sample saved to: {spoofed_file}")

if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("PHASE 4.4 VERIFICATION")
    print("Combined Deepfake, Motion & Blink → Liveness Score")
    print("🚀" * 35)
    
    clean_output = test_clean_video()
    spoofed_output = test_spoofed_video()
    
    if clean_output and spoofed_output:
        compare_results(clean_output, spoofed_output)
        save_outputs(clean_output, spoofed_output)
        
        print("\n" + "="*70)
        print("PHASE 4.4 DELIVERABLES")
        print("="*70)
        
        print("\n✅ Deliverable 1: Clean video final JSON output")
        print(f"   File: outputs/test_clean_sample.json")
        print(f"   Liveness score: {clean_output['liveness_score']:.3f}")
        
        print("\n✅ Deliverable 2: Spoofed video final JSON output")
        print(f"   File: outputs/test_forged_sample.json")
        print(f"   Liveness score: {spoofed_output['liveness_score']:.3f}")
        
        print("\n" + "="*70)
        print("🎉 PHASE 4.4 COMPLETE! 🎉")
        print("="*70)

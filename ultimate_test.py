import cv2
import numpy as np
import os
from core.sticker_detector import StickerDetector

def ultimate_detection_test(golden_path, sheet_path):
    """
    Test all detection methods to find maximum stickers
    """
    print("=" * 70)
    print("ULTIMATE STICKER DETECTION TEST")
    print("=" * 70)
    
    # Load images
    golden = cv2.imread(golden_path)
    sheet = cv2.imread(sheet_path)
    
    if golden is None or sheet is None:
        print("Error: Could not load images")
        return
    
    print(f"\nGolden sticker: {golden.shape[1]}x{golden.shape[0]}")
    print(f"Production sheet: {sheet.shape[1]}x{sheet.shape[0]}")
    
    # Create detector with very low threshold
    detector = StickerDetector(golden, match_threshold=0.35)
    
    # Try all methods
    print("\n" + "=" * 70)
    print("METHOD 1: Standard Template Matching (various thresholds)")
    print("=" * 70)
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    best_count = 0
    best_threshold = None
    best_stickers = []
    
    for thresh in thresholds:
        detector.match_threshold = thresh
        stickers = detector.detect_all_stickers(sheet)
        print(f"  Threshold {thresh}: {len(stickers)} stickers")
        
        if len(stickers) > best_count:
            best_count = len(stickers)
            best_threshold = thresh
            best_stickers = stickers
    
    print(f"\n  Best: {best_count} stickers at threshold {best_threshold}")
    
    print("\n" + "=" * 70)
    print("METHOD 2: Multi-Scale Detection")
    print("=" * 70)
    
    detector.match_threshold = 0.35
    stickers2 = detector.detect_with_multiple_scales(sheet)
    print(f"  Found: {len(stickers2)} stickers")
    
    print("\n" + "=" * 70)
    print("METHOD 3: Sliding Window Detection")
    print("=" * 70)
    
    stickers3 = detector.detect_with_sliding_window(sheet)
    print(f"  Found: {len(stickers3)} stickers")
    
    print("\n" + "=" * 70)
    print("METHOD 4: ALL METHODS COMBINED")
    print("=" * 70)
    
    all_stickers = detector.detect_all_methods(sheet)
    print(f"  Total unique: {len(all_stickers)} stickers")
    
    # Analyze grid pattern
    grid = detector.get_grid_pattern(all_stickers, tolerance=40)
    
    print("\n" + "=" * 70)
    print("GRID PATTERN ANALYSIS")
    print("=" * 70)
    print(f"  Rows detected: {grid['rows']}")
    print(f"  Columns detected: {grid['columns']}")
    print(f"  Expected stickers: {grid['total_expected']}")
    print(f"  Found stickers: {grid['total_found']}")
    print(f"  Missing stickers: {grid['missing']}")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Best from method 1
    vis1 = detector.visualize_detections(sheet, best_stickers)
    cv2.imwrite('detection_method1.png', vis1)
    print("  Saved: detection_method1.png")
    
    # Multi-scale
    vis2 = detector.visualize_detections(sheet, stickers2)
    cv2.imwrite('detection_multiscale.png', vis2)
    print("  Saved: detection_multiscale.png")
    
    # Sliding window
    vis3 = detector.visualize_detections(sheet, stickers3)
    cv2.imwrite('detection_sliding.png', vis3)
    print("  Saved: detection_sliding.png")
    
    # Combined
    vis_all = detector.visualize_detections(sheet, all_stickers)
    cv2.imwrite('detection_all_methods.png', vis_all)
    print("  Saved: detection_all_methods.png")
    
    # Print all sticker positions
    print("\n" + "=" * 70)
    print("ALL DETECTED STICKER POSITIONS")
    print("=" * 70)
    
    for i, sticker in enumerate(all_stickers):
        x, y = sticker['position']
        conf = sticker.get('confidence', 0)
        method = sticker.get('method', 'unknown')
        scale = sticker.get('scale', 1.0)
        
        scale_info = f" scale={scale:.2f}" if scale != 1.0 else ""
        print(f"  #{i+1:2d}: ({x:4d}, {y:4d}) - conf={conf:.3f} [{method}{scale_info}]")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if grid['missing'] > 0:
        print(f"⚠️  Missing {grid['missing']} stickers. Try:")
        print("  1. Use the 'All Methods' detection in main GUI")
        print("  2. Manually mark missing stickers using the GUI")
        print("  3. Check if stickers have different sizes (use multi-scale)")
        print("  4. Verify lighting is consistent across sheet")
    else:
        print("✓ All expected stickers detected!")
    
    print("\n" + "=" * 70)
    print(f"BEST RESULT: {len(all_stickers)} stickers found")
    print("=" * 70)
    
    return all_stickers

if __name__ == "__main__":
    print("Ultimate Sticker Detection Test")
    print("-" * 50)
    
    golden_path = input("Enter path to golden sticker: ").strip().strip('"')
    sheet_path = input("Enter path to production sheet: ").strip().strip('"')
    
    if os.path.exists(golden_path) and os.path.exists(sheet_path):
        stickers = ultimate_detection_test(golden_path, sheet_path)
    else:
        print("Error: File not found")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from core.sticker_detector import StickerDetector

def detect_and_visualize(golden_path, sheet_path, output_path="detection_result.png"):
    """
    Test sticker detection and visualize results
    """
    print("=" * 60)
    print("STICKER DETECTION TOOL")
    print("=" * 60)
    
    # Load images
    golden = cv2.imread(golden_path)
    sheet = cv2.imread(sheet_path)
    
    if golden is None:
        print(f"Error: Could not load golden image: {golden_path}")
        return
    if sheet is None:
        print(f"Error: Could not load sheet image: {sheet_path}")
        return
    
    print(f"Golden sticker: {golden.shape[1]}x{golden.shape[0]}")
    print(f"Production sheet: {sheet.shape[1]}x{sheet.shape[0]}")
    
    # Try different match thresholds
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    
    print("\nTesting different match thresholds:")
    print("-" * 50)
    
    best_stickers = []
    best_threshold = None
    max_stickers = 0
    
    for thresh in thresholds:
        detector = StickerDetector(golden, match_threshold=thresh)
        stickers = detector.detect_all_stickers(sheet)
        
        print(f"Threshold {thresh}: Found {len(stickers)} stickers")
        
        if len(stickers) > max_stickers:
            max_stickers = len(stickers)
            best_stickers = stickers
            best_threshold = thresh
    
    print(f"\nBest result: Threshold {best_threshold} -> {max_stickers} stickers")
    
    # Try multi-scale detection if single scale found few stickers
    if max_stickers < 3:
        print("\nTrying multi-scale detection...")
        detector = StickerDetector(golden, match_threshold=0.55)
        stickers = detector.detect_with_multiple_scales(sheet)
        
        print(f"Multi-scale detection: Found {len(stickers)} stickers")
        
        if len(stickers) > max_stickers:
            max_stickers = len(stickers)
            best_stickers = stickers
    
    # Create visualization
    vis = detector.visualize_detections(sheet, best_stickers)
    
    # Get grid pattern
    grid = detector.get_grid_pattern(best_stickers)
    if grid['total_expected'] > 0:
        print(f"\nGrid Pattern Analysis:")
        print(f"  Rows detected: {grid['rows']}")
        print(f"  Columns detected: {grid['columns']}")
        print(f"  Expected stickers: {grid['total_expected']}")
        print(f"  Found stickers: {grid['total_found']}")
        
        if grid['total_found'] < grid['total_expected']:
            print(f"  ⚠️  Missing {grid['total_expected'] - grid['total_found']} stickers")
    
    # Save result
    cv2.imwrite(output_path, vis)
    print(f"\nVisualization saved to: {output_path}")
    
    # Create detailed view with bounding boxes
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Original sheet
    axes[0].imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Sheet')
    axes[0].axis('off')
    
    # Detections
    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected Stickers ({len(best_stickers)} found)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_analysis.png', dpi=150)
    print(f"Analysis plot saved to: detection_analysis.png")
    
    # Print all sticker positions
    print(f"\nAll detected sticker positions:")
    print("-" * 50)
    for i, sticker in enumerate(best_stickers):
        x, y = sticker['position']
        confidence = sticker.get('confidence', 0)
        scale = sticker.get('scale', 1.0)
        scale_info = f" (scale={scale:.2f})" if scale != 1.0 else ""
        print(f"  #{i+1}: ({x:4d}, {y:4d}) - confidence: {confidence:.3f}{scale_info}")
    
    return best_stickers

if __name__ == "__main__":
    print("Sticker Detection Test Tool")
    print("=" * 60)
    
    golden_path = input("Enter path to golden sticker image: ").strip().strip('"')
    sheet_path = input("Enter path to production sheet image: ").strip().strip('"')
    
    if os.path.exists(golden_path) and os.path.exists(sheet_path):
        stickers = detect_and_visualize(golden_path, sheet_path)
        
        if stickers:
            print(f"\n✓ Detection complete! Found {len(stickers)} stickers.")
        else:
            print("\n⚠️  No stickers detected. Try adjusting the match threshold.")
    else:
        print("Error: File not found")

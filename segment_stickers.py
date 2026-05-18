import cv2
import numpy as np
import os
from core.sticker_segmenter import StickerSegmenter

def test_segmentation(sheet_path):
    """
    Test region-based sticker segmentation
    """
    print("=" * 70)
    print("REGION-BASED STICKER SEGMENTATION")
    print("=" * 70)
    
    # Load sheet
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return None, None
    
    print(f"Sheet size: {sheet.shape[1]}x{sheet.shape[0]}")
    
    # Calculate area ranges
    sheet_area = sheet.shape[0] * sheet.shape[1]
    min_area = int(sheet_area * 0.005)  # 0.5% of sheet
    max_area = int(sheet_area * 0.2)    # 20% of sheet
    
    print(f"Sticker area range: {min_area} - {max_area} pixels")
    
    # Create segmenter
    segmenter = StickerSegmenter(min_sticker_area=min_area, max_sticker_area=max_area)
    
    # Segment stickers
    stickers, grid = segmenter.segment_stickers(sheet)
    
    # Create visualization
    vis = segmenter.visualize_segmentation(sheet, stickers)
    
    # Save result
    output_path = "segmented_stickers.png"
    cv2.imwrite(output_path, vis)
    print(f"\nVisualization saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total stickers found: {len(stickers)}")
    print(f"Grid pattern: {grid['rows']} rows × {grid['columns']} columns")
    print(f"Expected stickers: {grid['total_expected']}")
    print(f"Found stickers: {grid['total_found']}")
    
    if grid['missing'] > 0:
        print(f"\n⚠️  Missing {grid['missing']} stickers")
        print("\nSticker positions found:")
        for i, sticker in enumerate(stickers):
            x, y, w, h = sticker['bbox']
            print(f"  #{i+1}: ({x}, {y}) - {w}x{h}")
    else:
        print("\n✓ All stickers successfully detected!")
        print("\nSticker positions:")
        for i, sticker in enumerate(stickers):
            x, y, w, h = sticker['bbox']
            print(f"  #{i+1}: ({x}, {y}) - {w}x{h}")
    
    return stickers, grid

if __name__ == "__main__":
    print("Region-Based Sticker Segmentation Test")
    print("-" * 50)
    
    sheet_path = input("Enter path to production sheet: ").strip().strip('"')
    
    if os.path.exists(sheet_path):
        stickers, grid = test_segmentation(sheet_path)
    else:
        print(f"Error: File not found - {sheet_path}")
        print("\nYou can also test with a sample image:")
        print("python create_test_sheet.py")

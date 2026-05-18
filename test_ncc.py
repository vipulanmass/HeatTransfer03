import cv2
import os
from core.sticker_detector_ncc import StickerDetectorNCC

def quick_test(template_path, sheet_path):
    """
    Quick test of NCC-based sticker detection
    """
    print("=" * 70)
    print("NCC STICKER DETECTION - QUICK TEST")
    print("=" * 70)
    
    # Check files
    if not os.path.exists(template_path):
        print(f"Error: Template not found - {template_path}")
        return
    
    if not os.path.exists(sheet_path):
        print(f"Error: Sheet not found - {sheet_path}")
        return
    
    # Create detector
    detector = StickerDetectorNCC(
        golden_template_path=template_path,
        match_threshold=0.7,
        red_threshold=0.05
    )
    
    # Process sheet
    stickers, grid_info = detector.process_sheet(sheet_path, output_dir="ncc_output")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total stickers detected: {len(stickers)}")
    print(f"Grid: {grid_info['rows']} × {grid_info['cols']}")
    
    defects = [s for s in stickers if s.get('is_defect', False)]
    print(f"Defects found: {len(defects)}")
    
    print("\nDefective stickers:")
    for sticker in defects:
        if 'grid_pos' in sticker:
            print(f"  Row {sticker['grid_pos'][0]}, Col {sticker['grid_pos'][1]}: Red ratio = {sticker['red_ratio']:.3f}")

if __name__ == "__main__":
    print("NCC Sticker Detection Test")
    print("-" * 50)
    
    template = input("Enter path to golden template: ").strip().strip('"')
    sheet = input("Enter path to production sheet: ").strip().strip('"')
    
    quick_test(template, sheet)

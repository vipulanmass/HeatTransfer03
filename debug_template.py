import cv2
import numpy as np
import os

def debug_template_matching(template_path, sheet_path):
    """
    Debug template matching to see what's happening
    """
    print("=" * 70)
    print("DEBUG: Template Matching Analysis")
    print("=" * 70)
    
    # Load images
    template = cv2.imread(template_path)
    sheet = cv2.imread(sheet_path)
    
    if template is None:
        print(f"Error: Could not load template: {template_path}")
        return
    if sheet is None:
        print(f"Error: Could not load sheet: {sheet_path}")
        return
    
    print(f"\nTemplate: {template.shape[1]}x{template.shape[0]}")
    print(f"Sheet: {sheet.shape[1]}x{sheet.shape[0]}")
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    
    # Check if sheet is large enough
    if sheet_gray.shape[0] < template_gray.shape[0] or sheet_gray.shape[1] < template_gray.shape[1]:
        print(f"\n⚠️  WARNING: Sheet is smaller than template!")
        print(f"   Template requires at least {template_gray.shape[1]}x{template_gray.shape[0]}")
        print(f"   Sheet is {sheet_gray.shape[1]}x{sheet_gray.shape[0]}")
        return
    
    # Perform template matching
    result = cv2.matchTemplate(sheet_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Statistics
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    print(f"\nTemplate Matching Statistics:")
    print(f"  Minimum score: {min_val:.4f}")
    print(f"  Maximum score: {max_val:.4f}")
    print(f"  Mean score: {result.mean():.4f}")
    print(f"  Std deviation: {result.std():.4f}")
    
    # Find all matches at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\nMatches at different thresholds:")
    print("-" * 50)
    for thresh in thresholds:
        locations = np.where(result >= thresh)
        num_matches = len(locations[0])
        print(f"  Threshold {thresh:.1f}: {num_matches} matches")
    
    # Best match location
    print(f"\nBest match location: {max_loc}")
    print(f"Best match score: {max_val:.4f}")
    
    # Check if template is in the sheet at all
    if max_val < 0.5:
        print(f"\n⚠️  WARNING: Template matching score is low ({max_val:.3f})")
        print("   This suggests the template may not be present in the sheet")
        print("   Possible issues:")
        print("   - Template is a different sticker type")
        print("   - Sheet has different lighting")
        print("   - Template is rotated relative to sheet")
    
    # Create visualization
    h, w = template_gray.shape
    vis = sheet.copy()
    
    # Draw best match
    cv2.rectangle(vis, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 3)
    cv2.putText(vis, f"Score: {max_val:.3f}", (max_loc[0], max_loc[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("debug_best_match.png", vis)
    print(f"\n✓ Saved best match visualization: debug_best_match.png")
    
    # Create heatmap
    heatmap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite("debug_heatmap.png", heatmap_color)
    print(f"✓ Saved heatmap: debug_heatmap.png")
    
    return max_val, max_loc

if __name__ == "__main__":
    print("Template Matching Debug Tool")
    print("-" * 50)
    
    template_path = input("Enter path to golden template: ").strip().strip('"')
    sheet_path = input("Enter path to production sheet: ").strip().strip('"')
    
    if os.path.exists(template_path) and os.path.exists(sheet_path):
        debug_template_matching(template_path, sheet_path)
    else:
        print("Error: One or both files not found")

import cv2
import numpy as np
import os

def find_coordinates_interactive(sheet_path):
    """
    Simple tool to help find coordinates by scanning the image
    """
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return
    
    h, w = sheet.shape[:2]
    
    print("=" * 70)
    print("COORDINATE FINDER")
    print("=" * 70)
    print(f"Sheet size: {w}x{h}")
    print("\nLet's find a good sticker by scanning regions.")
    
    # Try different starting points based on common sticker locations
    test_positions = [
        (500, 500),    # Center area
        (800, 800),    # Right-center
        (300, 300),    # Left-center
        (1000, 1000),  # Lower right
        (1200, 500),   # Top-right
        (500, 1500),   # Middle-lower
        (200, 200),    # Top-left
        (1500, 200),   # Top-right corner
        (800, 2000),   # Lower area
        (400, 2500),   # Bottom area
    ]
    
    print("\nChecking potential sticker locations:")
    print("-" * 60)
    
    found_regions = []
    for i, (x, y) in enumerate(test_positions):
        # Check if within bounds
        if y + 150 < h and x + 150 < w:
            # Extract region
            region = sheet[y:y+150, x:x+150]
            
            # Calculate features
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (150 * 150)
            
            print(f"\n  Region {i+1}: X={x}, Y={y}")
            print(f"    Contrast: {contrast:.0f}, Edge density: {edge_density:.3f}")
            
            # Save preview
            filename = f"test_region_{i+1}.png"
            cv2.imwrite(filename, region)
            print(f"    Saved: {filename}")
            
            found_regions.append((x, y, contrast, edge_density))
        else:
            print(f"\n  Region {i+1}: X={x}, Y={y} - out of bounds")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Open the test_region_*.png files to see what they look like")
    print("2. Pick the one that shows a good, clean sticker")
    print("3. Use those coordinates to extract your template")
    print("\nExample command:")
    print("  python extract_template_at_coords.py")
    
    return found_regions

if __name__ == "__main__":
    print("Coordinate Finder Tool")
    print("-" * 50)
    sheet_path = input("Enter path to your production sheet: ").strip().strip('"')
    
    if os.path.exists(sheet_path):
        find_coordinates_interactive(sheet_path)
    else:
        print(f"File not found: {sheet_path}")

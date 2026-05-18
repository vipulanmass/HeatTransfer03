import cv2
import numpy as np
import os

def visualize_with_grid(sheet_path, output_path="sheet_with_grid.png"):
    """
    Create a visualization with grid lines to help identify sticker positions
    """
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return
    
    h, w = sheet.shape[:2]
    
    # Create a copy for drawing
    vis = sheet.copy()
    
    # Draw grid lines every 100 pixels to help with coordinates
    for x in range(0, w, 100):
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(vis, str(x), (x+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    for y in range(0, h, 100):
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(vis, str(y), (5, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Add center cross
    cv2.line(vis, (w//2, 0), (w//2, h), (255, 0, 0), 1)
    cv2.line(vis, (0, h//2), (w, h//2), (255, 0, 0), 1)
    
    cv2.imwrite(output_path, vis)
    print(f"✓ Saved grid visualization: {output_path}")
    print(f"  Sheet size: {w}x{h}")
    print(f"\nOpen '{output_path}' to see the sheet with coordinate grid")
    print("The green numbers show X and Y coordinates at 100-pixel intervals")
    
    return output_path

if __name__ == "__main__":
    print("Create Grid Visualization")
    print("-" * 50)
    sheet_path = input("Enter path to your production sheet: ").strip().strip('"')
    
    if os.path.exists(sheet_path):
        visualize_with_grid(sheet_path)
    else:
        print(f"File not found: {sheet_path}")

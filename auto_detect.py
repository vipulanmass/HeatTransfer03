import cv2
import numpy as np
import os

def auto_detect_sticker_positions(sheet_path):
    """
    Automatically detect sticker positions using edge detection and contour analysis
    """
    print("=" * 70)
    print("AUTO-DETECTING STICKER POSITIONS")
    print("=" * 70)
    
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return
    
    h, w = sheet.shape[:2]
    print(f"Sheet size: {w}x{h}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size (typical sticker size ~ 100-150 pixels)
    min_width = 80
    max_width = 180
    min_height = 90
    max_height = 180
    
    stickers = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        
        if min_width < cw < max_width and min_height < ch < max_height:
            stickers.append({
                'x': x,
                'y': y,
                'w': cw,
                'h': ch,
                'area': cw * ch
            })
    
    # Sort stickers by position (top to bottom, left to right)
    stickers.sort(key=lambda s: (s['y'], s['x']))
    
    print(f"\nFound {len(stickers)} potential sticker positions")
    
    # Create visualization
    vis = sheet.copy()
    for i, sticker in enumerate(stickers):
        x, y, w, h = sticker['x'], sticker['y'], sticker['w'], sticker['h']
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis, f"#{i+1}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite("auto_detected_stickers.png", vis)
    print(f"✓ Saved: auto_detected_stickers.png")
    
    # Print the first few sticker positions
    print("\nSticker positions (first 10):")
    print("-" * 50)
    for i, sticker in enumerate(stickers[:10]):
        print(f"  Sticker {i+1}: X={sticker['x']}, Y={sticker['y']}, Size={sticker['w']}x{sticker['h']}")
    
    # Suggest a template
    if stickers:
        first_sticker = stickers[0]
        print(f"\nSuggested template from first sticker:")
        print(f"  X: {first_sticker['x']}, Y: {first_sticker['y']}")
        print(f"  Width: {first_sticker['w']}, Height: {first_sticker['h']}")
        
        # Extract a template
        template = sheet[first_sticker['y']:first_sticker['y']+first_sticker['h'], 
                         first_sticker['x']:first_sticker['x']+first_sticker['w']]
        cv2.imwrite("auto_extracted_template.png", template)
        print(f"\n✓ Auto-extracted template saved: auto_extracted_template.png")
        print("  Use this as your golden template for detection")
    
    return stickers

if __name__ == "__main__":
    print("Auto-Detect Sticker Positions")
    print("-" * 50)
    sheet_path = input("Enter path to your production sheet: ").strip().strip('"')
    
    if os.path.exists(sheet_path):
        auto_detect_sticker_positions(sheet_path)
    else:
        print(f"File not found: {sheet_path}")

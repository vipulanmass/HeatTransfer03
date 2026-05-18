import cv2
import numpy as np
import os

def extract_template(sheet_path, x, y, w=109, h=129):
    """
    Extract template at given coordinates
    """
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return
    
    print(f"Sheet size: {sheet.shape[1]}x{sheet.shape[0]}")
    
    # Check bounds
    if y + h > sheet.shape[0] or x + w > sheet.shape[1]:
        print(f"\nError: Region goes out of bounds!")
        print(f"  Sheet size: {sheet.shape[1]}x{sheet.shape[0]}")
        print(f"  Requested: X={x} to {x+w}, Y={y} to {y+h}")
        return None
    
    # Extract template
    template = sheet[y:y+h, x:x+w]
    
    # Save
    cv2.imwrite("extracted_template.png", template)
    print(f"\n✓ Template saved: extracted_template.png")
    print(f"  Size: {w}x{h}")
    print(f"  Position: X={x}, Y={y}")
    
    # Create preview
    preview = sheet.copy()
    cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imwrite("template_location.png", preview)
    print(f"✓ Location preview: template_location.png")
    
    # Show stats
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    print(f"  Contrast: {contrast:.0f}")
    
    return template

if __name__ == "__main__":
    print("Extract Template at Coordinates")
    print("-" * 50)
    
    sheet_path = input("Enter path to your production sheet: ").strip().strip('"')
    
    if os.path.exists(sheet_path):
        print("\nEnter coordinates for a good sticker (from the images you saw):")
        try:
            x = int(input("X coordinate (top-left): "))
            y = int(input("Y coordinate (top-left): "))
            w = input("Width (press Enter for 109): ").strip()
            h = input("Height (press Enter for 129): ").strip()
            
            w = int(w) if w else 109
            h = int(h) if h else 129
            
            extract_template(sheet_path, x, y, w, h)
        except ValueError:
            print("Please enter valid numbers")
    else:
        print(f"File not found: {sheet_path}")

import cv2
import numpy as np
import os

def extract_good_template(sheet_path, output_path="good_template.png"):
    """
    Interactive tool to extract a good template from the sheet
    """
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return
    
    # Display sheet info
    print(f"Sheet loaded: {sheet.shape[1]}x{sheet.shape[0]}")
    print("\nNow you need to manually extract a good sticker.")
    print("\nOption 1: Use mouse to select region (if you have a GUI)")
    print("Option 2: Enter coordinates manually")
    
    # Create a temporary window for selection (if you want to use mouse)
    # For now, let's use manual coordinates
    
    print("\n" + "=" * 50)
    print("Enter the bounding box of a GOOD sticker on the sheet")
    print("(Look for a clean, undamaged sticker with good print quality)")
    print("=" * 50)
    
    try:
        x = int(input("X coordinate (top-left): "))
        y = int(input("Y coordinate (top-left): "))
        w = int(input("Width: "))
        h = int(input("Height: "))
        
        # Extract template
        template = sheet[y:y+h, x:x+w]
        
        # Save
        cv2.imwrite(output_path, template)
        print(f"\n✓ Template saved: {output_path}")
        print(f"  Size: {w}x{h}")
        print(f"\nNow run detection with this new template.")
        
        # Create a preview
        preview = sheet.copy()
        cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.imwrite("template_location.png", preview)
        print(f"  Location saved: template_location.png")
        
        return template
        
    except ValueError:
        print("Invalid coordinates. Please enter numbers only.")
        return None

if __name__ == "__main__":
    sheet_path = input("Enter path to your production sheet: ").strip().strip('"')
    
    if os.path.exists(sheet_path):
        extract_good_template(sheet_path)
    else:
        print("File not found")

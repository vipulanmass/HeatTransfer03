import cv2
import numpy as np

def create_test_sheet():
    """Create a test sheet with simulated stickers"""
    print("Creating test sheet with simulated stickers...")
    
    # Create white background sheet
    sheet = np.ones((1200, 1600, 3), dtype=np.uint8) * 255
    
    # Define sticker positions (grid of 3x4 = 12 stickers)
    sticker_width = 250
    sticker_height = 200
    margin_x = 100
    margin_y = 100
    spacing_x = 50
    spacing_y = 40
    
    colors = [(0, 100, 200), (200, 100, 0), (100, 200, 0), (0, 200, 100)]
    
    sticker_count = 0
    for row in range(3):
        for col in range(4):
            x = margin_x + col * (sticker_width + spacing_x)
            y = margin_y + row * (sticker_height + spacing_y)
            
            # Draw sticker background
            cv2.rectangle(sheet, (x, y), (x+sticker_width, y+sticker_height), 
                         colors[sticker_count % len(colors)], -1)
            
            # Draw border
            cv2.rectangle(sheet, (x, y), (x+sticker_width, y+sticker_height), 
                         (0, 0, 0), 2)
            
            # Add text
            cv2.putText(sheet, f"STICKER {sticker_count+1}", 
                       (x+30, y+100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 2)
            
            sticker_count += 1
    
    # Save the test sheet
    cv2.imwrite("test_sheet_real.png", sheet)
    print(f"Created test_sheet_real.png with {sticker_count} simulated stickers")
    print("You can now use this file to test the segmentation:")
    print("python segment_stickers.py")
    print("Then enter: test_sheet_real.png")

if __name__ == "__main__":
    create_test_sheet()

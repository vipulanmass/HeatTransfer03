import cv2
import numpy as np
import os

print("=" * 60)
print("BINARY TEMPLATE MATCHER - SIMPLE VERSION")
print("=" * 60)

# Get paths
template_path = input("\nEnter path to GOLDEN TEMPLATE (sticker): ").strip().strip('"')
sheet_path = input("Enter path to PRODUCTION SHEET: ").strip().strip('"')

# Check files exist
if not os.path.exists(template_path):
    print(f"Error: Template not found - {template_path}")
    exit()
if not os.path.exists(sheet_path):
    print(f"Error: Sheet not found - {sheet_path}")
    exit()

# Load images
template = cv2.imread(template_path)
sheet = cv2.imread(sheet_path)

print(f"\nTemplate size: {template.shape[1]}x{template.shape[0]}")
print(f"Sheet size: {sheet.shape[1]}x{sheet.shape[0]}")

# Create binary template
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
gray_template = cv2.GaussianBlur(gray_template, (5, 5), 0)

# Get threshold
thresh_input = input("\nEnter binary threshold (0-255, default 120): ").strip()
threshold = int(thresh_input) if thresh_input else 120

# Create binary images
_, binary_template = cv2.threshold(gray_template, threshold, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("binary_template.png", binary_template)
print(f"✓ Binary template saved: binary_template.png")

gray_sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
gray_sheet = cv2.GaussianBlur(gray_sheet, (5, 5), 0)
_, binary_sheet = cv2.threshold(gray_sheet, threshold, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("binary_sheet.png", binary_sheet)
print(f"✓ Binary sheet saved: binary_sheet.png")

# Template matching
match_thresh = input("\nMatch threshold (0.3-0.8, default 0.5): ").strip()
match_thresh = float(match_thresh) if match_thresh else 0.5

result = cv2.matchTemplate(binary_sheet, binary_template, cv2.TM_CCOEFF_NORMED)
locations = np.where(result >= match_thresh)

matches = []
for pt in zip(*locations[::-1]):
    x, y = pt
    matches.append((x, y))
    print(f"Match found at: ({x}, {y})")

print(f"\nFound {len(matches)} matches")

if matches:
    # Draw boxes
    vis = sheet.copy()
    template_h, template_w = binary_template.shape
    
    for i, (x, y) in enumerate(matches):
        cv2.rectangle(vis, (x, y), (x+template_w, y+template_h), (0, 255, 0), 3)
        cv2.putText(vis, f"#{i+1}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite("detected_stickers.png", vis)
    print(f"✓ Saved: detected_stickers.png")
    
    # Extract stickers
    os.makedirs("stickers", exist_ok=True)
    for i, (x, y) in enumerate(matches):
        sticker = sheet[y:y+template_h, x:x+template_w]
        cv2.imwrite(f"stickers/sticker_{i+1:03d}.png", sticker)
    print(f"✓ Saved {len(matches)} stickers to 'stickers' folder")
else:
    print("\nNo matches found! Try:")
    print("1. Lower match threshold (try 0.3)")
    print("2. Adjust binary threshold")
    print("3. Check if template matches the sheet")

print("\nDone!")
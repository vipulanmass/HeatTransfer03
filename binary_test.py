import cv2
import numpy as np
import os

print("=" * 60)
print("MULTI-THRESHOLD BINARY GENERATOR")
print("=" * 60)

# Get image path
image_path = input("\nEnter image path: ").strip().strip('"')

if not os.path.exists(image_path):
    print(f"Error: File not found")
    exit()

# Load image
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Create output folder
output_folder = "binary_tests"
os.makedirs(output_folder, exist_ok=True)

# Test different thresholds
thresholds = [50, 80, 100, 120, 150, 180, 200]

print("\nGenerating binary images with different thresholds:")
print("-" * 50)

for thresh in thresholds:
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    
    white = cv2.countNonZero(binary)
    total = img.shape[0] * img.shape[1]
    white_pct = (white / total) * 100
    
    filename = f"{output_folder}/binary_{thresh}.png"
    cv2.imwrite(filename, binary)
    
    print(f"Threshold {thresh}: {white_pct:.1f}% white -> {filename}")

# Also generate Otsu auto threshold
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
white = cv2.countNonZero(binary_otsu)
white_pct = (white / total) * 100
filename = f"{output_folder}/binary_otsu.png"
cv2.imwrite(filename, binary_otsu)
print(f"Otsu auto: {white_pct:.1f}% white -> {filename}")

print(f"\n✓ All binary images saved to '{output_folder}/'")
print("\nOpen the images to see which threshold works best!")
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_sheet_differences(golden_path, production_path):
    """
    Analyze differences between golden and production sheets
    """
    print("=" * 60)
    print("SHEET DIFFERENCE ANALYSIS")
    print("=" * 60)
    
    # Load images
    golden = cv2.imread(golden_path)
    production = cv2.imread(production_path)
    
    if golden is None or production is None:
        print("Error: Could not load images")
        return
    
    print(f"\nGolden sheet: {golden.shape[1]}x{golden.shape[0]}")
    print(f"Production sheet: {production.shape[1]}x{production.shape[0]}")
    
    # Resize production to match golden if needed
    if golden.shape != production.shape:
        print(f"\nResizing production to match golden...")
        production = cv2.resize(production, (golden.shape[1], golden.shape[0]))
    
    # Convert to grayscale
    golden_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    production_gray = cv2.cvtColor(production, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(golden_gray, production_gray)
    
    # Statistics
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff_max = np.max(diff)
    diff_sum = np.sum(diff)
    
    print(f"\nDifference Statistics:")
    print(f"  Mean difference: {diff_mean:.2f}")
    print(f"  Std deviation: {diff_std:.2f}")
    print(f"  Max difference: {diff_max:.2f}")
    print(f"  Total difference: {diff_sum:.2f}")
    
    # Threshold analysis
    thresholds = [10, 20, 30, 40, 50, 100]
    print(f"\nDefect area at different thresholds:")
    print(f"{'Threshold':<12} {'Defect Pixels':<15} {'Area %':<10}")
    print("-" * 40)
    
    for thresh in thresholds:
        _, defect_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        defect_pixels = cv2.countNonZero(defect_mask)
        total_pixels = diff.shape[0] * diff.shape[1]
        defect_percent = (defect_pixels / total_pixels) * 100
        
        print(f"{thresh:<12} {defect_pixels:<15} {defect_percent:.2f}%")
    
    # Find difference contours
    _, defect_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze contour sizes
    contour_areas = [cv2.contourArea(c) for c in contours]
    contour_areas.sort(reverse=True)
    
    print(f"\nDefect Region Analysis:")
    print(f"  Total defect regions: {len(contours)}")
    print(f"  Largest defect area: {contour_areas[0] if contour_areas else 0:.0f} pixels")
    if len(contour_areas) > 1:
        print(f"  Median defect area: {contour_areas[len(contour_areas)//2]:.0f} pixels")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Golden image
    axes[0, 0].imshow(cv2.cvtColor(golden, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Golden Sheet')
    axes[0, 0].axis('off')
    
    # Production image
    axes[0, 1].imshow(cv2.cvtColor(production, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Production Sheet')
    axes[0, 1].axis('off')
    
    # Difference heatmap
    im = axes[1, 0].imshow(diff, cmap='hot')
    axes[1, 0].set_title('Difference Heatmap (lighter = more difference)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Defect overlay
    overlay = production.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Only show significant defects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Defect Overlay ({len(contours)} regions)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('difference_analysis.png', dpi=150)
    print(f"\nVisualization saved to: difference_analysis.png")
    
    # Recommendation
    print(f"\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Suggest threshold based on your data
    suggested_threshold = diff_sum / 1000000  # Rough suggestion
    
    print(f"Current difference score: {diff_sum:.2f}")
    print(f"\nSuggested threshold values:")
    print(f"  Strict (high quality): diff_sum < 100,000")
    print(f"  Standard: diff_sum < 200,000")
    print(f"  Tolerant (allow variation): diff_sum < 300,000")
    
    return diff_sum, len(contours)

if __name__ == "__main__":
    # Update these paths with your actual files
    golden_path = input("Enter path to golden sheet image: ").strip().strip('"')
    production_path = input("Enter path to production sheet image: ").strip().strip('"')
    
    if os.path.exists(golden_path) and os.path.exists(production_path):
        analyze_sheet_differences(golden_path, production_path)
    else:
        print("Error: One or both files not found")

import cv2
import numpy as np
import os
from scipy.ndimage import maximum_filter

class FullStickerDetector:
    """
    Detect complete stickers by matching the full template
    and drawing bounding boxes that cover the entire sticker
    """
    
    def __init__(self, golden_template_path):
        """
        Initialize with golden sticker template (should be a COMPLETE sticker)
        """
        self.load_template(golden_template_path)
        
    def load_template(self, template_path):
        """Load the golden sticker template (should be a complete sticker)"""
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not load template: {template_path}")
        
        # Store template dimensions - these will be the bounding box size
        self.template_h, self.template_w = self.template.shape[:2]
        
        # Create multiple representations for robust matching
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_gray = cv2.GaussianBlur(self.template_gray, (3, 3), 0)
        
        # Edge map for better matching
        self.template_edges = cv2.Canny(self.template_gray, 50, 150)
        
        print(f"✓ Golden template loaded: {self.template_w}x{self.template_h} pixels")
        print(f"  This template will define the bounding box size for each sticker")
        
    def process_sheet(self, sheet_path, output_dir="sticker_output"):
        """
        Find all stickers on the sheet and draw complete bounding boxes
        """
        print("\n" + "=" * 80)
        print("FULL STICKER DETECTION SYSTEM")
        print("=" * 80)
        
        # Load sheet
        sheet = cv2.imread(sheet_path)
        if sheet is None:
            print(f"Error: Could not load {sheet_path}")
            return None
        
        h, w = sheet.shape[:2]
        print(f"\nSheet loaded: {w}x{h} pixels")
        print(f"Template size: {self.template_w}x{self.template_h}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess sheet
        sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        sheet_gray = cv2.GaussianBlur(sheet_gray, (3, 3), 0)
        sheet_edges = cv2.Canny(sheet_gray, 50, 150)
        
        print("\n" + "=" * 80)
        print("STEP 1: Multi-Method Template Matching")
        print("=" * 80)
        
        # Method 1: Grayscale NCC
        result_gray = cv2.matchTemplate(sheet_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        max_gray = result_gray.max()
        print(f"Grayscale NCC - Max score: {max_gray:.3f}")
        
        # Method 2: Edge-based matching
        result_edge = cv2.matchTemplate(sheet_edges, self.template_edges, cv2.TM_CCOEFF_NORMED)
        max_edge = result_edge.max()
        print(f"Edge-based NCC - Max score: {max_edge:.3f}")
        
        # Choose best method
        best_result = result_gray if max_gray > max_edge else result_edge
        best_method = "grayscale" if max_gray > max_edge else "edge"
        print(f"Using {best_method} method (score: {best_result.max():.3f})")
        
        # Determine optimal threshold
        if best_result.max() < 0.3:
            threshold = max(0.2, best_result.max() - 0.05)
            print(f"⚠️  Low match scores. Using adaptive threshold: {threshold:.3f}")
        else:
            threshold = max(0.3, best_result.max() * 0.7)
            print(f"Using threshold: {threshold:.3f}")
        
        print("\n" + "=" * 80)
        print("STEP 2: Finding Sticker Positions")
        print("=" * 80)
        
        # Find all matches above threshold
        locations = np.where(best_result >= threshold)
        
        # Convert to list of points
        matches = []
        for pt in zip(*locations[::-1]):
            x, y = pt
            score = best_result[y, x]
            matches.append({
                'x': x,
                'y': y,
                'score': score
            })
        
        print(f"Raw matches found: {len(matches)}")
        
        if len(matches) == 0:
            print("\n❌ No stickers detected!")
            print("\nPossible solutions:")
            print("1. Your golden template might not match the stickers on this sheet")
            print("2. Try extracting a template directly from this sheet")
            print("3. Ensure the template is a complete sticker")
            return None
        
        # Step 3: Non-Maximum Suppression
        print("\n" + "=" * 80)
        print("STEP 3: Non-Maximum Suppression")
        print("=" * 80)
        
        # Calculate minimum distance between stickers (based on template size)
        min_distance = int(min(self.template_w, self.template_h) * 0.6)
        
        # Sort by score
        matches.sort(key=lambda m: m['score'], reverse=True)
        
        # Apply NMS
        unique_matches = []
        for match in matches:
            x, y = match['x'], match['y']
            too_close = False
            
            for unique in unique_matches:
                ux, uy = unique['x'], unique['y']
                distance = np.sqrt((x - ux)**2 + (y - uy)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                unique_matches.append(match)
        
        print(f"After NMS: {len(unique_matches)} unique stickers")
        
        # Step 4: Grid verification
        print("\n" + "=" * 80)
        print("STEP 4: Grid Verification")
        print("=" * 80)
        
        grid_matches = self._verify_grid(unique_matches)
        print(f"After grid verification: {len(grid_matches)} stickers")
        
        # Step 5: Generate complete bounding boxes
        print("\n" + "=" * 80)
        print("STEP 5: Generating Complete Bounding Boxes")
        print("=" * 80)
        
        stickers = []
        for match in grid_matches:
            # The match position (x, y) is the top-left corner of the template match
            x = match['x']
            y = match['y']
            
            # Use the full template dimensions for the bounding box
            # This ensures the box covers the entire sticker
            bbox = (x, y, self.template_w, self.template_h)
            
            stickers.append({
                'bbox': bbox,
                'center': (x + self.template_w//2, y + self.template_h//2),
                'score': match['score'],
                'exists': True,
                'index': len(stickers)
            })
            
            print(f"  Sticker {len(stickers)}: position ({x}, {y}), size {self.template_w}x{self.template_h}, score: {match['score']:.3f}")
        
        # Step 6: Save results
        print("\n" + "=" * 80)
        print("STEP 6: Saving Results")
        print("=" * 80)
        
        # Create visualization with full bounding boxes
        vis = self._create_visualization(sheet, stickers)
        cv2.imwrite(f"{output_dir}/full_sticker_detection.png", vis)
        print(f"✓ Visualization saved: {output_dir}/full_sticker_detection.png")
        
        # Save individual stickers
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            roi = sheet[y:y+h, x:x+w]
            filename = f"{output_dir}/sticker_{sticker['index']:03d}.png"
            cv2.imwrite(filename, roi)
            print(f"✓ Saved: {filename}")
        
        # Save report
        self._save_report(sheet, stickers, output_dir)
        
        print("\n" + "=" * 80)
        print(f"✅ COMPLETE! Found {len(stickers)} full stickers")
        print(f"📁 Output saved to: {output_dir}/")
        print("=" * 80)
        
        return stickers
    
    def _verify_grid(self, matches, tolerance=30):
        """
        Verify matches form a grid pattern
        """
        if len(matches) < 2:
            return matches
        
        # Extract coordinates
        x_coords = [m['x'] for m in matches]
        y_coords = [m['y'] for m in matches]
        
        # Find unique rows
        rows = []
        for y in sorted(y_coords):
            found = False
            for ry in rows:
                if abs(y - ry) < tolerance:
                    found = True
                    break
            if not found:
                rows.append(y)
        
        # Find unique columns
        cols = []
        for x in sorted(x_coords):
            found = False
            for cx in cols:
                if abs(x - cx) < tolerance:
                    found = True
                    break
            if not found:
                cols.append(x)
        
        print(f"  Detected grid: {len(rows)} rows x {len(cols)} columns")
        
        # If we have a grid, keep matches that align with it
        if len(rows) > 1 and len(cols) > 1:
            verified = []
            for match in matches:
                x, y = match['x'], match['y']
                # Check if this match aligns with grid
                for ry in rows:
                    if abs(y - ry) < tolerance:
                        for cx in cols:
                            if abs(x - cx) < tolerance:
                                verified.append(match)
                                break
                        break
            
            if verified:
                print(f"  Grid verification kept {len(verified)}/{len(matches)} stickers")
                return verified
        
        return matches
    
    def _create_visualization(self, sheet, stickers):
        """
        Create visualization with complete bounding boxes
        """
        vis = sheet.copy()
        
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            
            # Green for detected stickers
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Add label
            label = f"#{sticker['index']+1}"
            cv2.putText(vis, label, (x+5, y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add score
            score_text = f"{sticker['score']:.2f}"
            cv2.putText(vis, score_text, (x+5, y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis
    
    def _save_report(self, sheet, stickers, output_dir):
        """Save detection report"""
        report_path = f"{output_dir}/detection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FULL STICKER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TEMPLATE INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Template size: {self.template_w}x{self.template_h} pixels\n")
            f.write(f"This defines the bounding box size for each sticker\n\n")
            
            f.write("DETECTION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total stickers found: {len(stickers)}\n\n")
            
            f.write("STICKER DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'#':<6} {'Position (X,Y)':<20} {'Size (WxH)':<15} {'Score':<8}\n")
            f.write("-" * 50 + "\n")
            
            for sticker in stickers:
                x, y, w, h = sticker['bbox']
                f.write(f"{sticker['index']+1:<6} ({x:4d},{y:4d}){' ':<10} {w}x{h:<10} {sticker['score']:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Report saved: {report_path}")

def extract_template_from_sheet(sheet_path):
    """
    Helper function to extract a complete sticker template from the sheet
    """
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return
    
    h, w = sheet.shape[:2]
    
    print("\n" + "=" * 80)
    print("EXTRACT COMPLETE STICKER TEMPLATE")
    print("=" * 80)
    print(f"Sheet size: {w}x{h}")
    print("\nLook at your sheet and find a GOOD, COMPLETE sticker")
    print("Enter its approximate position (you can adjust later)")
    
    try:
        x = int(input("X coordinate (top-left corner): "))
        y = int(input("Y coordinate (top-left corner): "))
        sticker_w = int(input("Sticker width (in pixels): "))
        sticker_h = int(input("Sticker height (in pixels): "))
        
        if y + sticker_h <= h and x + sticker_w <= w:
            template = sheet[y:y+sticker_h, x:x+sticker_w]
            cv2.imwrite("extracted_template.png", template)
            print(f"\n✓ Template saved: extracted_template.png")
            print(f"  Size: {sticker_w}x{sticker_h}")
            print(f"  Position: ({x}, {y})")
            return "extracted_template.png"
        else:
            print("Error: Sticker would be outside sheet bounds")
            return None
            
    except ValueError:
        print("Invalid input")
        return None

def main():
    """Main entry point"""
    print("=" * 80)
    print("FULL STICKER DETECTION SYSTEM")
    print("Detects complete stickers with full bounding boxes")
    print("=" * 80)
    
    # Option to extract template first
    print("\nDo you want to:")
    print("1. Use an existing golden template")
    print("2. Extract a template from your sheet first")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        sheet_path = input("Enter path to your sheet: ").strip().strip('"')
        if os.path.exists(sheet_path):
            template_path = extract_template_from_sheet(sheet_path)
            if not template_path:
                return
        else:
            print("Sheet not found")
            return
    else:
        template_path = input("Enter path to golden template: ").strip().strip('"')
        sheet_path = input("Enter path to production sheet: ").strip().strip('"')
    
    if not os.path.exists(template_path):
        print(f"Template not found: {template_path}")
        return
    
    if not os.path.exists(sheet_path):
        print(f"Sheet not found: {sheet_path}")
        return
    
    # Create detector and process
    detector = FullStickerDetector(template_path)
    stickers = detector.process_sheet(sheet_path)
    
    if stickers:
        print(f"\n✅ Found {len(stickers)} complete stickers!")
        print("Check the 'sticker_output' folder for results")

if __name__ == "__main__":
    main()

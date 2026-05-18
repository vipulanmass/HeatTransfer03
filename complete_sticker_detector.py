import cv2
import numpy as np
import os
from scipy.ndimage import maximum_filter

class FullStickerDetector:
    """
    Detect and crop complete stickers by finding the full sticker boundaries
    """
    
    def __init__(self, golden_template_path):
        """
        Initialize with golden sticker template (should be a COMPLETE sticker)
        """
        self.load_template(golden_template_path)
        
    def load_template(self, template_path):
        """Load the golden sticker template (must be a complete sticker)"""
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
        print(f"  This template defines the bounding box size for each sticker")
        
    def find_sticker_boundaries(self, sheet):
        """
        Alternative method: Find sticker boundaries using contour detection
        This works even when template matching fails
        """
        print("\n" + "=" * 80)
        print("METHOD 2: Contour-Based Sticker Detection")
        print("=" * 80)
        
        # Convert to grayscale
        gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to find sticker areas
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)
        
        # Morphological operations to connect text into blocks
        kernel = np.ones((10, 10), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (sticker-sized regions)
        sheet_area = sheet.shape[0] * sheet.shape[1]
        min_sticker_area = sheet_area * 0.005  # 0.5% of sheet
        max_sticker_area = sheet_area * 0.2    # 20% of sheet
        
        stickers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_sticker_area < area < max_sticker_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Expand slightly to ensure full sticker capture
                expand = 10
                x = max(0, x - expand)
                y = max(0, y - expand)
                w = min(sheet.shape[1] - x, w + expand*2)
                h = min(sheet.shape[0] - y, h + expand*2)
                
                stickers.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'method': 'contour'
                })
        
        print(f"Found {len(stickers)} stickers via contour detection")
        return stickers
    
    def process_sheet(self, sheet_path, output_dir="sticker_output"):
        """
        Find all stickers on the sheet and crop complete stickers
        """
        print("\n" + "=" * 80)
        print("COMPLETE STICKER DETECTION SYSTEM")
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
        
        # METHOD 1: Template Matching with Full Sticker Size
        print("\n" + "=" * 80)
        print("METHOD 1: Template Matching (Full Sticker Size)")
        print("=" * 80)
        
        # Preprocess sheet
        sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        sheet_gray = cv2.GaussianBlur(sheet_gray, (3, 3), 0)
        sheet_edges = cv2.Canny(sheet_gray, 50, 150)
        
        # Try both matching methods
        result_gray = cv2.matchTemplate(sheet_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        result_edge = cv2.matchTemplate(sheet_edges, self.template_edges, cv2.TM_CCOEFF_NORMED)
        
        max_gray = result_gray.max()
        max_edge = result_edge.max()
        
        print(f"Grayscale NCC - Max score: {max_gray:.3f}")
        print(f"Edge-based NCC - Max score: {max_edge:.3f}")
        
        # Choose best method
        if max_gray > max_edge:
            best_result = result_gray
            best_method = "grayscale"
        else:
            best_result = result_edge
            best_method = "edge"
        
        print(f"Using {best_method} method (score: {best_result.max():.3f})")
        
        # Find matches
        threshold = max(0.25, best_result.max() * 0.6)  # Adaptive threshold
        print(f"Using threshold: {threshold:.3f}")
        
        locations = np.where(best_result >= threshold)
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
        
        # Apply Non-Maximum Suppression
        if matches:
            min_distance = int(min(self.template_w, self.template_h) * 0.5)
            matches.sort(key=lambda m: m['score'], reverse=True)
            
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
            template_stickers = unique_matches
        else:
            template_stickers = []
        
        # METHOD 2: Contour-based detection as fallback
        print("\n" + "=" * 80)
        print("METHOD 2: Contour-Based Detection")
        print("=" * 80)
        
        contour_stickers = self.find_sticker_boundaries(sheet)
        
        # COMBINE both methods
        print("\n" + "=" * 80)
        print("COMBINING RESULTS")
        print("=" * 80)
        
        all_stickers = []
        used_positions = set()
        
        # Add template matches first (they are more accurate)
        for match in template_stickers:
            x, y = match['x'], match['y']
            # Use template dimensions for bounding box
            bbox = (x, y, self.template_w, self.template_h)
            all_stickers.append({
                'bbox': bbox,
                'score': match['score'],
                'method': 'template',
                'center': (x + self.template_w//2, y + self.template_h//2)
            })
            used_positions.add((x, y))
        
        # Add contour stickers that don't overlap with template matches
        for sticker in contour_stickers:
            x, y, w, h = sticker['bbox']
            center = (x + w//2, y + h//2)
            
            # Check if this area is already covered by a template match
            overlap = False
            for existing in all_stickers:
                ex, ey, ew, eh = existing['bbox']
                if (abs(center[0] - (ex + ew//2)) < ew and 
                    abs(center[1] - (ey + eh//2)) < eh):
                    overlap = True
                    break
            
            if not overlap:
                all_stickers.append({
                    'bbox': (x, y, w, h),
                    'score': 0.5,
                    'method': 'contour',
                    'center': center
                })
        
        print(f"Total unique stickers: {len(all_stickers)}")
        
        # Sort stickers by position (top to bottom, left to right)
        all_stickers.sort(key=lambda s: (s['center'][1], s['center'][0]))
        
        # Assign indices
        for i, sticker in enumerate(all_stickers):
            sticker['index'] = i
        
        # Save results
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Create visualization
        vis = sheet.copy()
        for sticker in all_stickers:
            x, y, w, h = sticker['bbox']
            
            # Different colors for different detection methods
            if sticker['method'] == 'template':
                color = (0, 255, 0)  # Green for template matches
            else:
                color = (0, 255, 255)  # Yellow for contour matches
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 3)
            
            # Add label
            label = f"#{sticker['index']+1}"
            cv2.putText(vis, label, (x+5, y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(f"{output_dir}/complete_stickers.png", vis)
        print(f"✓ Visualization saved: {output_dir}/complete_stickers.png")
        
        # Save individual stickers
        sticker_dir = os.path.join(output_dir, "individual_stickers")
        os.makedirs(sticker_dir, exist_ok=True)
        
        for sticker in all_stickers:
            x, y, w, h = sticker['bbox']
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, sheet.shape[1] - 1))
            y = max(0, min(y, sheet.shape[0] - 1))
            w = min(w, sheet.shape[1] - x)
            h = min(h, sheet.shape[0] - y)
            
            # Extract the full sticker
            full_sticker = sheet[y:y+h, x:x+w]
            
            # Save
            filename = f"{sticker_dir}/sticker_{sticker['index']+1:03d}.png"
            cv2.imwrite(filename, full_sticker)
            print(f"✓ Saved: {filename} ({w}x{h})")
        
        # Save report
        self._save_report(sheet, all_stickers, output_dir)
        
        print("\n" + "=" * 80)
        print(f"✅ COMPLETE! Found {len(all_stickers)} full stickers")
        print(f"📁 Output saved to: {output_dir}/")
        print(f"📁 Individual stickers: {output_dir}/individual_stickers/")
        print("=" * 80)
        
        return all_stickers
    
    def _save_report(self, sheet, stickers, output_dir):
        """Save detection report"""
        report_path = f"{output_dir}/detection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE STICKER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TEMPLATE INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Template size: {self.template_w}x{self.template_h} pixels\n")
            f.write(f"This defines the bounding box size for each sticker\n\n")
            
            f.write("DETECTION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total stickers found: {len(stickers)}\n")
            
            template_count = sum(1 for s in stickers if s['method'] == 'template')
            contour_count = sum(1 for s in stickers if s['method'] == 'contour')
            f.write(f"  Template matches: {template_count}\n")
            f.write(f"  Contour matches: {contour_count}\n\n")
            
            f.write("STICKER DETAILS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'#':<6} {'Method':<12} {'Position (X,Y)':<20} {'Size (WxH)':<15} {'Score':<8}\n")
            f.write("-" * 70 + "\n")
            
            for sticker in stickers:
                x, y, w, h = sticker['bbox']
                score = sticker.get('score', 0.5)
                f.write(f"{sticker['index']+1:<6} {sticker['method']:<12} ({x:4d},{y:4d}){' ':<10} {w}x{h:<10} {score:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Report saved: {report_path}")

def extract_template_from_sheet(sheet_path):
    """
    Helper function to extract a COMPLETE sticker template from the sheet
    """
    sheet = cv2.imread(sheet_path)
    if sheet is None:
        print(f"Error: Could not load {sheet_path}")
        return None
    
    h, w = sheet.shape[:2]
    
    print("\n" + "=" * 80)
    print("EXTRACT COMPLETE STICKER TEMPLATE")
    print("=" * 80)
    print(f"Sheet size: {w}x{h}")
    print("\nLook at your sheet and find a GOOD, COMPLETE sticker")
    print("The template should include the ENTIRE sticker (borders, background, all content)")
    print("\nEnter the bounding box that covers the ENTIRE sticker:")
    
    try:
        x = int(input("X coordinate (top-left corner): "))
        y = int(input("Y coordinate (top-left corner): "))
        sticker_w = int(input("Sticker width (in pixels): "))
        sticker_h = int(input("Sticker height (in pixels): "))
        
        # Validate bounds
        if y + sticker_h > h or x + sticker_w > w:
            print("Error: Sticker would be outside sheet bounds")
            return None
        
        # Extract the full sticker
        template = sheet[y:y+sticker_h, x:x+sticker_w]
        
        # Save
        cv2.imwrite("complete_template.png", template)
        print(f"\n✓ Template saved: complete_template.png")
        print(f"  Size: {sticker_w}x{sticker_h}")
        print(f"  Position: ({x}, {y})")
        print(f"\nThis template will now be used to detect all stickers of this exact size")
        
        return "complete_template.png"
        
    except ValueError:
        print("Invalid input")
        return None

def main():
    """Main entry point"""
    print("=" * 80)
    print("COMPLETE STICKER DETECTION SYSTEM")
    print("Detects and crops FULL stickers (entire sticker area)")
    print("=" * 80)
    
    # Option to extract template first
    print("\nDo you want to:")
    print("1. Use an existing golden template (complete sticker)")
    print("2. Extract a complete sticker template from your sheet")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        sheet_path = input("Enter path to your sheet: ").strip().strip('"')
        if os.path.exists(sheet_path):
            template_path = extract_template_from_sheet(sheet_path)
            if not template_path:
                return
            sheet_path = input("Enter path to production sheet (or same as above): ").strip().strip('"')
        else:
            print("Sheet not found")
            return
    else:
        template_path = input("Enter path to golden template (complete sticker): ").strip().strip('"')
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
        print(f"\n✅ Successfully extracted {len(stickers)} complete stickers!")
        print("Check the 'sticker_output/individual_stickers' folder")

if __name__ == "__main__":
    main()

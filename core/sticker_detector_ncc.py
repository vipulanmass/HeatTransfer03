import cv2
import numpy as np
from typing import List, Dict, Tuple
import os

class StickerDetectorNCC:
    """
    Sticker detection using Normalized Cross-Correlation (NCC) template matching
    with preprocessing for better matching
    """
    
    def __init__(self, 
                 golden_template_path=None,
                 match_threshold=0.3,  # Lower threshold for challenging matches
                 min_distance_ratio=0.6,
                 red_threshold=0.05):
        self.match_threshold = match_threshold
        self.min_distance_ratio = min_distance_ratio
        self.red_threshold = red_threshold
        
        if golden_template_path:
            self.load_template(golden_template_path)
    
    def load_template(self, template_path):
        """Load and preprocess the golden sticker template"""
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not load template: {template_path}")
        
        # Preprocess template
        self.template_gray = self._preprocess(cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY))
        self.template_h, self.template_w = self.template_gray.shape
        print(f"✓ Loaded template: {self.template_w}x{self.template_h}")
        
        return self.template
    
    def _preprocess(self, image):
        """
        Preprocess image to improve matching:
        - CLAHE for contrast enhancement
        - Gaussian blur to reduce noise
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    def detect_stickers(self, sheet_image):
        """
        Detect all stickers using NCC template matching with preprocessing
        """
        print("\n" + "=" * 70)
        print("STEP 1: Preprocessing Sheet")
        print("=" * 70)
        
        # Preprocess sheet
        if len(sheet_image.shape) == 3:
            sheet_gray = cv2.cvtColor(sheet_image, cv2.COLOR_BGR2GRAY)
        else:
            sheet_gray = sheet_image
        
        # Apply same preprocessing to sheet
        sheet_processed = self._preprocess(sheet_gray)
        
        print(f"Sheet size: {sheet_processed.shape[1]}x{sheet_processed.shape[0]}")
        print(f"Template size: {self.template_w}x{self.template_h}")
        
        # Check if sheet is large enough
        if sheet_processed.shape[0] < self.template_h or sheet_processed.shape[1] < self.template_w:
            print(f"⚠️  Warning: Sheet is smaller than template!")
            return [], {'rows': 0, 'cols': 0, 'total_expected': 0, 'total_found': 0, 'missing': 0}
        
        print("\n" + "=" * 70)
        print("STEP 2: Template Matching with NCC")
        print("=" * 70)
        
        # Perform template matching
        result = cv2.matchTemplate(sheet_processed, self.template_gray, cv2.TM_CCOEFF_NORMED)
        
        print(f"Match result shape: {result.shape}")
        print(f"Max match score: {result.max():.3f}")
        print(f"Min match score: {result.min():.3f}")
        print(f"Mean match score: {result.mean():.3f}")
        
        # If max score is still too low, try with edge detection
        if result.max() < self.match_threshold:
            print(f"\n⚠️  Low match scores. Trying edge-based matching...")
            
            # Try edge-based matching
            template_edges = cv2.Canny(self.template_gray, 50, 150)
            sheet_edges = cv2.Canny(sheet_processed, 50, 150)
            result_edges = cv2.matchTemplate(sheet_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            
            print(f"Edge-based max score: {result_edges.max():.3f}")
            
            if result_edges.max() > result.max():
                result = result_edges
                print("Using edge-based matching (better results)")
        
        # Find matches above threshold
        locations = np.where(result >= self.match_threshold)
        
        print(f"\nMatches found: {len(locations[0])}")
        
        if len(locations[0]) == 0:
            # Try with lower threshold
            lower_threshold = max(0.2, result.max() - 0.1)
            print(f"No matches at threshold {self.match_threshold}")
            print(f"Trying lower threshold: {lower_threshold:.3f}")
            locations = np.where(result >= lower_threshold)
            print(f"Matches at lower threshold: {len(locations[0])}")
            
            if len(locations[0]) == 0:
                print("Still no matches. Template may not match any stickers on sheet.")
                return [], {'rows': 0, 'cols': 0, 'total_expected': 0, 'total_found': 0, 'missing': 0}
        
        # Convert to list of points
        points = []
        for pt in zip(*locations[::-1]):
            x, y = pt
            score = result[y, x]
            points.append({'position': (x, y), 'score': score})
        
        print("\n" + "=" * 70)
        print("STEP 3: Non-Maximum Suppression")
        print("=" * 70)
        
        # Non-Maximum Suppression
        min_distance = int(min(self.template_w, self.template_h) * self.min_distance_ratio)
        unique_stickers = self._non_max_suppression(points, min_distance)
        
        print(f"After NMS: {len(unique_stickers)} unique stickers")
        
        if len(unique_stickers) == 0:
            print("No unique stickers after NMS.")
            return [], {'rows': 0, 'cols': 0, 'total_expected': 0, 'total_found': 0, 'missing': 0}
        
        print("\n" + "=" * 70)
        print("STEP 4: Grid Fitting")
        print("=" * 70)
        
        # Fit to grid
        grid_stickers, grid_info = self._fit_to_grid(unique_stickers)
        
        print(f"After grid fitting: {len(grid_stickers)} stickers")
        
        if grid_info['rows'] > 0:
            print(f"Grid: {grid_info['rows']} rows × {grid_info['cols']} columns")
            print(f"Expected stickers: {grid_info['total_expected']}")
            print(f"Found: {grid_info['total_found']}")
            if grid_info['missing'] > 0:
                print(f"Missing: {grid_info['missing']}")
        else:
            print("No clear grid pattern detected, using raw matches")
        
        print("\n" + "=" * 70)
        print("STEP 5: Extracting ROIs")
        print("=" * 70)
        
        # Extract ROIs
        for sticker in grid_stickers:
            x, y = sticker['position']
            if y + self.template_h <= sheet_image.shape[0] and x + self.template_w <= sheet_image.shape[1]:
                sticker['roi'] = sheet_image[y:y+self.template_h, x:x+self.template_w]
            else:
                h_actual = min(self.template_h, sheet_image.shape[0] - y)
                w_actual = min(self.template_w, sheet_image.shape[1] - x)
                sticker['roi'] = sheet_image[y:y+h_actual, x:x+w_actual]
            sticker['bbox'] = (x, y, self.template_w, self.template_h)
        
        print(f"Extracted {len(grid_stickers)} sticker ROIs")
        
        return grid_stickers, grid_info
    
    def _non_max_suppression(self, points, min_distance):
        """Non-Maximum Suppression - keep only local maxima"""
        if not points:
            return []
        
        points.sort(key=lambda p: p['score'], reverse=True)
        
        kept = []
        for point in points:
            x, y = point['position']
            too_close = False
            for kept_point in kept:
                kx, ky = kept_point['position']
                distance = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                if distance < min_distance:
                    too_close = True
                    break
            if not too_close:
                kept.append(point)
        
        return kept
    
    def _fit_to_grid(self, stickers):
        """Fit detected stickers to a grid pattern"""
        if not stickers:
            return [], {'rows': 0, 'cols': 0, 'total_expected': 0, 'total_found': 0, 'missing': 0}
        
        positions = np.array([s['position'] for s in stickers])
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Cluster rows
        y_sorted = np.sort(y_coords)
        rows = []
        for y in y_sorted:
            if not rows or abs(y - rows[-1]) > self.template_h * 0.5:
                rows.append(y)
        
        # Cluster columns
        x_sorted = np.sort(x_coords)
        cols = []
        for x in x_sorted:
            if not cols or abs(x - cols[-1]) > self.template_w * 0.5:
                cols.append(x)
        
        # Create grid
        grid_stickers = []
        for r, row_y in enumerate(rows):
            for c, col_x in enumerate(cols):
                exists = False
                for sticker in stickers:
                    sx, sy = sticker['position']
                    if abs(sx - col_x) < self.template_w * 0.5 and abs(sy - row_y) < self.template_h * 0.5:
                        exists = True
                        grid_stickers.append({
                            'position': (col_x, row_y),
                            'score': sticker['score'],
                            'grid_pos': (r, c),
                            'predicted': False
                        })
                        break
                
                if not exists:
                    grid_stickers.append({
                        'position': (col_x, row_y),
                        'score': 0.0,
                        'grid_pos': (r, c),
                        'predicted': True
                    })
        
        grid_stickers.sort(key=lambda s: (s['grid_pos'][0], s['grid_pos'][1]))
        
        grid_info = {
            'rows': len(rows),
            'cols': len(cols),
            'total_expected': len(rows) * len(cols),
            'total_found': len(stickers),
            'missing': (len(rows) * len(cols)) - len(stickers) if len(rows) > 0 and len(cols) > 0 else 0
        }
        
        return grid_stickers, grid_info
    
    def flag_defects(self, stickers):
        """Flag defective stickers based on red pixel analysis"""
        if not stickers:
            return stickers
        
        print("\n" + "=" * 70)
        print("STEP 6: Defect Flagging")
        print("=" * 70)
        
        for sticker in stickers:
            roi = sticker.get('roi')
            if roi is None:
                sticker['red_ratio'] = 0
                sticker['is_defect'] = False
                continue
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            total_pixels = roi.shape[0] * roi.shape[1]
            red_pixels = cv2.countNonZero(red_mask)
            red_ratio = red_pixels / total_pixels
            
            sticker['red_ratio'] = red_ratio
            sticker['is_defect'] = red_ratio > self.red_threshold
            
            if sticker['is_defect']:
                print(f"  Sticker {sticker.get('grid_pos', '?')}: Red ratio = {red_ratio:.3f} -> DEFECT")
        
        return stickers
    
    def create_visualization(self, sheet_image, stickers):
        """Create visualization with bounding boxes"""
        vis = sheet_image.copy()
        
        for sticker in stickers:
            x, y = sticker['position']
            w, h = self.template_w, self.template_h
            
            if sticker.get('is_defect', False):
                color = (0, 0, 255)  # Red
            elif sticker.get('predicted', False):
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            if sticker.get('grid_pos'):
                label = f"{sticker['grid_pos'][0]},{sticker['grid_pos'][1]}"
                cv2.putText(vis, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis
    
    def process_sheet(self, sheet_image_path, output_dir="output"):
        """Complete processing pipeline"""
        print("\n" + "=" * 70)
        print("STICKER DETECTION SYSTEM")
        print("=" * 70)
        
        sheet = cv2.imread(sheet_image_path)
        if sheet is None:
            print(f"Error: Could not load {sheet_image_path}")
            return None, None
        
        print(f"Loaded sheet: {sheet.shape[1]}x{sheet.shape[0]}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        stickers, grid_info = self.detect_stickers(sheet)
        
        if not stickers:
            print("\n⚠️  No stickers detected!")
            return stickers, grid_info
        
        stickers = self.flag_defects(stickers)
        
        vis = self.create_visualization(sheet, stickers)
        cv2.imwrite(f"{output_dir}/detected_stickers.png", vis)
        print(f"\n✓ Visualization: {output_dir}/detected_stickers.png")
        
        # Save individual stickers
        for sticker in stickers:
            if 'grid_pos' in sticker and 'roi' in sticker:
                r, c = sticker['grid_pos']
                cv2.imwrite(f"{output_dir}/sticker_r{r}_c{c}.png", sticker['roi'])
        
        return stickers, grid_info

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

class StickerProcessor:
    """
    Process heat transfer sheets by removing red marks and segmenting stickers
    """
    
    def __init__(self, 
                 red_remove=True,
                 min_sticker_width=100,
                 min_sticker_height=100,
                 use_fixed_grid=True):
        self.red_remove = red_remove
        self.min_sticker_width = min_sticker_width
        self.min_sticker_height = min_sticker_height
        self.use_fixed_grid = use_fixed_grid
    
    def remove_red_marks(self, image):
        """
        Remove red marks (X's, logos, annotations) using inpainting
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges (red wraps around HSV wheel)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create red masks
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Dilate mask to cover more area around red marks
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        
        # Inpaint to remove red marks
        cleaned = cv2.inpaint(image, red_mask, 3, cv2.INPAINT_TELEA)
        
        return cleaned, red_mask
    
    def segment_by_contours(self, image):
        """
        Segment stickers using contour detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold to find sticker boundaries
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Dilate to connect text into blocks
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        stickers = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if w > self.min_sticker_width and h > self.min_sticker_height:
                stickers.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': (x + w//2, y + h//2)
                })
        
        # Sort by position (top to bottom, left to right)
        stickers.sort(key=lambda s: (s['centroid'][1], s['centroid'][0]))
        
        return stickers
    
    def segment_by_fixed_grid(self, image, rows=None, cols=None):
        """
        Segment using fixed grid (if you know the exact grid layout)
        """
        h, w = image.shape[:2]
        
        # Auto-detect grid if not provided
        if rows is None or cols is None:
            # Use contour detection to guess grid
            contours_stickers = self.segment_by_contours(image)
            if contours_stickers:
                # Estimate rows and columns from centroids
                y_coords = [s['centroid'][1] for s in contours_stickers]
                x_coords = [s['centroid'][0] for s in contours_stickers]
                
                # Cluster to find unique rows and columns
                rows = len(set(int(y/50) for y in y_coords))
                cols = len(set(int(x/50) for x in x_coords))
                
                # Validate
                if rows == 0 or cols == 0:
                    rows, cols = 3, 4  # Default fallback
        
        # Calculate cell dimensions
        cell_h = h // rows
        cell_w = w // cols
        
        # Create grid
        stickers = []
        for row in range(rows):
            for col in range(cols):
                x = col * cell_w
                y = row * cell_h
                
                # Adjust for last row/column to include full area
                if row == rows - 1:
                    cell_h_actual = h - y
                else:
                    cell_h_actual = cell_h
                    
                if col == cols - 1:
                    cell_w_actual = w - x
                else:
                    cell_w_actual = cell_w
                
                stickers.append({
                    'bbox': (x, y, cell_w_actual, cell_h_actual),
                    'grid_pos': (row, col),
                    'area': cell_w_actual * cell_h_actual,
                    'centroid': (x + cell_w_actual//2, y + cell_h_actual//2)
                })
        
        return stickers
    
    def process_sheet(self, image_path, rows=None, cols=None, output_dir="output"):
        """
        Complete processing pipeline
        """
        print("=" * 70)
        print("STICKER PROCESSING PIPELINE")
        print("=" * 70)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load {image_path}")
            return None, None
        
        print(f"Loaded: {image_path} ({img.shape[1]}x{img.shape[0]})")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Remove red marks
        if self.red_remove:
            print("\n1. Removing red marks...")
            cleaned, red_mask = self.remove_red_marks(img)
            cv2.imwrite(f"{output_dir}/01_cleaned_sheet.png", cleaned)
            cv2.imwrite(f"{output_dir}/01_red_mask.png", red_mask)
            print(f"   ✓ Saved cleaned sheet to {output_dir}/01_cleaned_sheet.png")
        else:
            cleaned = img
        
        # Step 2: Segment stickers
        print("\n2. Segmenting stickers...")
        if self.use_fixed_grid:
            stickers = self.segment_by_fixed_grid(cleaned, rows, cols)
            print(f"   Using fixed grid: {len(stickers)} stickers")
        else:
            stickers = self.segment_by_contours(cleaned)
            print(f"   Using contour detection: {len(stickers)} stickers")
        
        # Step 3: Extract and save individual stickers
        print("\n3. Extracting individual stickers...")
        sticker_rois = []
        for i, sticker in enumerate(stickers):
            x, y, w, h = sticker['bbox']
            roi = cleaned[y:y+h, x:x+w]
            
            # Save with descriptive name
            if 'grid_pos' in sticker:
                filename = f"{output_dir}/sticker_row{sticker['grid_pos'][0]}_col{sticker['grid_pos'][1]}.png"
            else:
                filename = f"{output_dir}/sticker_{i:03d}.png"
            
            cv2.imwrite(filename, roi)
            sticker_rois.append({
                'index': i,
                'bbox': sticker['bbox'],
                'grid_pos': sticker.get('grid_pos', None),
                'roi': roi,
                'filepath': filename
            })
            print(f"   Saved: {filename}")
        
        # Step 4: Create visualization
        print("\n4. Creating visualization...")
        vis = self.create_visualization(img, cleaned, stickers)
        cv2.imwrite(f"{output_dir}/02_visualization.png", vis)
        print(f"   Saved: {output_dir}/02_visualization.png")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total stickers extracted: {len(sticker_rois)}")
        print(f"Output directory: {output_dir}")
        
        return sticker_rois, stickers
    
    def create_visualization(self, original, cleaned, stickers):
        """
        Create visualization showing original, cleaned, and segmented stickers
        """
        h, w = original.shape[:2]
        
        # Create canvas for side-by-side comparison
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        canvas[:, :w] = original
        canvas[:, w:] = cleaned
        
        # Draw bounding boxes on cleaned side
        for sticker in stickers:
            x, y, w_box, h_box = sticker['bbox']
            cv2.rectangle(canvas, (w + x, y), (w + x + w_box, y + h_box), (0, 255, 0), 2)
            
            # Add label
            if 'grid_pos' in sticker:
                label = f"{sticker['grid_pos'][0]},{sticker['grid_pos'][1]}"
            else:
                label = f"#{sticker.get('index', 0)}"
            
            cv2.putText(canvas, label, (w + x + 5, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add titles
        cv2.putText(canvas, "ORIGINAL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "CLEANED + SEGMENTED", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return canvas
    
    def analyze_grid(self, stickers):
        """
        Analyze the grid layout of stickers
        """
        if not stickers:
            return None
        
        # Get grid positions if available
        grid_positions = [s.get('grid_pos', None) for s in stickers]
        if None not in grid_positions:
            rows = max(p[0] for p in grid_positions) + 1
            cols = max(p[1] for p in grid_positions) + 1
            
            # Check for missing stickers
            found_positions = set(grid_positions)
            missing = []
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in found_positions:
                        missing.append((r, c))
            
            return {
                'rows': rows,
                'cols': cols,
                'total_expected': rows * cols,
                'total_found': len(stickers),
                'missing_positions': missing,
                'complete': len(missing) == 0
            }
        
        return None

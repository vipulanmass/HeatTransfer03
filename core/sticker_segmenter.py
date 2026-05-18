import cv2
import numpy as np
from typing import List, Dict, Tuple

class StickerSegmenter:
    """
    Segment stickers based on region properties (no template matching)
    """
    
    def __init__(self, min_sticker_area=1000, max_sticker_area=50000):
        self.min_sticker_area = min_sticker_area
        self.max_sticker_area = max_sticker_area
    
    def segment_stickers(self, sheet_image):
        """
        Segment all stickers from the sheet using region-based methods
        """
        print("\n" + "=" * 60)
        print("STICKER SEGMENTATION - Region Based")
        print("=" * 60)
        
        results = {}
        
        # Method 1: Color-based segmentation
        print("\n1. Color-based segmentation...")
        stickers_color = self._segment_by_color(sheet_image)
        print(f"   Found: {len(stickers_color)} stickers")
        results['color'] = stickers_color
        
        # Method 2: Edge-based segmentation
        print("\n2. Edge-based segmentation...")
        stickers_edge = self._segment_by_edges(sheet_image)
        print(f"   Found: {len(stickers_edge)} stickers")
        results['edge'] = stickers_edge
        
        # Method 3: Contour-based segmentation
        print("\n3. Contour-based segmentation...")
        stickers_contour = self._segment_by_contours(sheet_image)
        print(f"   Found: {len(stickers_contour)} stickers")
        results['contour'] = stickers_contour
        
        # Method 4: Adaptive thresholding
        print("\n4. Adaptive threshold segmentation...")
        stickers_adaptive = self._segment_by_adaptive_threshold(sheet_image)
        print(f"   Found: {len(stickers_adaptive)} stickers")
        results['adaptive'] = stickers_adaptive
        
        # Combine all methods
        print("\n5. Combining all methods...")
        combined = self._combine_results(results, sheet_image.shape[:2])
        print(f"   Total unique: {len(combined)} stickers")
        
        # Analyze grid pattern
        grid = self._analyze_grid(combined)
        print(f"\nGrid Analysis: {grid['rows']} rows × {grid['columns']} columns = {grid['total_expected']} expected stickers")
        print(f"Found: {grid['total_found']} stickers")
        
        if grid['missing'] > 0:
            print(f"⚠️  Missing {grid['missing']} stickers")
        else:
            print(f"✓ All stickers detected!")
        
        return combined, grid
    
    def _segment_by_color(self, image):
        """Segment stickers based on color differences"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect white/light background
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        
        background_mask = cv2.inRange(hsv, lower_white, upper_white)
        foreground = cv2.bitwise_not(background_mask)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stickers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_sticker_area < area < self.max_sticker_area:
                x, y, w, h = cv2.boundingRect(contour)
                stickers.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': (x + w//2, y + h//2),
                    'method': 'color'
                })
        
        return stickers
    
    def _segment_by_edges(self, image):
        """Segment stickers using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stickers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_sticker_area < area < self.max_sticker_area:
                x, y, w, h = cv2.boundingRect(contour)
                stickers.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': (x + w//2, y + h//2),
                    'method': 'edge'
                })
        
        return stickers
    
    def _segment_by_contours(self, image):
        """Segment using contour detection on binary image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stickers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_sticker_area < area < self.max_sticker_area:
                x, y, w, h = cv2.boundingRect(contour)
                stickers.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': (x + w//2, y + h//2),
                    'method': 'contour'
                })
        
        return stickers
    
    def _segment_by_adaptive_threshold(self, image):
        """Segment using adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stickers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_sticker_area < area < self.max_sticker_area:
                x, y, w, h = cv2.boundingRect(contour)
                stickers.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': (x + w//2, y + h//2),
                    'method': 'adaptive'
                })
        
        return stickers
    
    def _combine_results(self, results, image_shape, iou_threshold=0.5):
        """Combine results from multiple methods and remove duplicates"""
        all_stickers = []
        
        for method, stickers in results.items():
            for sticker in stickers:
                all_stickers.append(sticker)
        
        # Remove duplicates using IoU
        unique_stickers = []
        
        for sticker in all_stickers:
            is_duplicate = False
            x1, y1, w1, h1 = sticker['bbox']
            
            for existing in unique_stickers:
                x2, y2, w2, h2 = existing['bbox']
                iou = self._compute_iou((x1, y1, w1, h1), (x2, y2, w2, h2))
                
                if iou > iou_threshold:
                    is_duplicate = True
                    if sticker['area'] > existing['area']:
                        unique_stickers.remove(existing)
                        unique_stickers.append(sticker)
                    break
            
            if not is_duplicate:
                unique_stickers.append(sticker)
        
        unique_stickers.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
        return unique_stickers
    
    def _analyze_grid(self, stickers, tolerance=50):
        """Analyze grid pattern from sticker positions"""
        if not stickers:
            return {'rows': 0, 'columns': 0, 'total_expected': 0, 'total_found': 0}
        
        # Group by Y coordinates (rows)
        y_coords = [s['centroid'][1] for s in stickers]
        y_coords.sort()
        
        rows = []
        for y in y_coords:
            if not rows or abs(y - rows[-1]) > tolerance:
                rows.append(y)
        
        # Group by X coordinates (columns)
        x_coords = [s['centroid'][0] for s in stickers]
        x_coords.sort()
        
        cols = []
        for x in x_coords:
            if not cols or abs(x - cols[-1]) > tolerance:
                cols.append(x)
        
        return {
            'rows': len(rows),
            'columns': len(cols),
            'total_expected': len(rows) * len(cols),
            'total_found': len(stickers),
            'missing': (len(rows) * len(cols)) - len(stickers),
            'row_positions': rows,
            'col_positions': cols
        }
    
    def _compute_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        
        return intersection / (area1 + area2 - intersection)
    
    def visualize_segmentation(self, image, stickers):
        """Create visualization with all segmented stickers"""
        vis = image.copy()
        
        for i, sticker in enumerate(stickers):
            x, y, w, h = sticker['bbox']
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f"#{i+1}"
            cv2.putText(vis, label, (x+5, y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def get_sticker_rois(self, image, stickers):
        """Extract each sticker ROI for inspection"""
        rois = []
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            roi = image[y:y+h, x:x+w]
            rois.append({
                'bbox': sticker['bbox'],
                'roi': roi,
                'area': sticker['area']
            })
        return rois

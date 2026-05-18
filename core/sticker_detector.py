import cv2
import numpy as np
from typing import List, Tuple, Dict
import os

class StickerDetector:
    """
    Maximum detection sticker detector - finds ALL stickers
    """
    
    def __init__(self, template=None, match_threshold=0.4):  # Lower threshold
        self.template = template
        self.match_threshold = match_threshold
        
        if template is not None:
            self.template_h, self.template_w = template.shape[:2]
    
    def set_template(self, template):
        """Set the golden sticker template"""
        self.template = template
        self.template_h, self.template_w = template.shape[:2]
    
    def preprocess_image(self, image):
        """Enhance image for better template matching"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge enhancement
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Combine original with edges
        combined = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)
        
        return combined
    
    def detect_all_stickers(self, sheet_image, min_distance=None):
        """
        Detect ALL stickers using aggressive template matching
        """
        if self.template is None:
            raise ValueError("Template not set")
        
        # Set minimum distance (allow closer stickers)
        if min_distance is None:
            min_distance = int(min(self.template_w, self.template_h) * 0.3)  # Allow closer
        
        # Preprocess images
        sheet_enhanced = self.preprocess_image(sheet_image)
        template_enhanced = self.preprocess_image(self.template)
        
        # Method 1: Standard template matching
        result = cv2.matchTemplate(sheet_enhanced, template_enhanced, cv2.TM_CCOEFF_NORMED)
        
        # Find ALL locations above threshold (use lower threshold)
        locations = np.where(result >= self.match_threshold)
        
        # Method 2: Edge-based matching (for better detection)
        sheet_edges = cv2.Canny(sheet_enhanced, 50, 150)
        template_edges = cv2.Canny(template_enhanced, 50, 150)
        edge_result = cv2.matchTemplate(sheet_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        edge_locations = np.where(edge_result >= self.match_threshold - 0.1)
        
        # Combine both methods
        all_points = []
        
        # Add points from standard matching
        for pt in zip(*locations[::-1]):
            x, y = pt
            confidence = result[y, x]
            all_points.append({'position': (x, y), 'confidence': confidence, 'method': 'standard'})
        
        # Add points from edge matching
        for pt in zip(*edge_locations[::-1]):
            x, y = pt
            confidence = edge_result[y, x]
            all_points.append({'position': (x, y), 'confidence': confidence, 'method': 'edge'})
        
        # Apply NMS with smaller distance
        stickers = self._non_max_suppression(all_points, min_distance)
        
        # Sort by position
        stickers.sort(key=lambda s: (s['position'][1], s['position'][0]))
        
        # Extract ROIs
        for sticker in stickers:
            x, y = sticker['position']
            sticker['roi'] = sheet_image[y:y+self.template_h, x:x+self.template_w]
            sticker['bbox'] = (x, y, self.template_w, self.template_h)
        
        return stickers
    
    def detect_with_multiple_scales(self, sheet_image, scales=[0.8, 0.9, 1.0, 1.1, 1.2]):
        """
        Detect with multiple scales
        """
        all_stickers = []
        
        for scale in scales:
            new_w = int(self.template_w * scale)
            new_h = int(self.template_h * scale)
            
            if new_w < 30 or new_h < 30:
                continue
            
            scaled_template = cv2.resize(self.template, (new_w, new_h))
            
            # Store original
            original_template = self.template
            original_h, original_w = self.template_h, self.template_w
            
            # Set scaled template
            self.set_template(scaled_template)
            
            # Detect at this scale (use lower threshold for scaled versions)
            original_threshold = self.match_threshold
            self.match_threshold = 0.35  # Even lower for scaled
            stickers = self.detect_all_stickers(sheet_image)
            self.match_threshold = original_threshold
            
            # Adjust coordinates
            for sticker in stickers:
                x, y = sticker['position']
                sticker['position'] = (int(x / scale), int(y / scale))
                sticker['bbox'] = (int(x / scale), int(y / scale), original_w, original_h)
                sticker['scale'] = scale
            
            all_stickers.extend(stickers)
            
            # Restore original
            self.set_template(original_template)
        
        # Remove duplicates with small distance
        unique = self._non_max_suppression(all_stickers, min_distance=int(min(self.template_w, self.template_h) * 0.3))
        
        return unique
    
    def detect_with_sliding_window(self, sheet_image, step_percent=0.5):
        """
        Sliding window detection - finds stickers even if template matching fails
        """
        h, w = sheet_image.shape[:2]
        template_h, template_w = self.template_h, self.template_w
        
        # Calculate step size
        step_x = int(template_w * step_percent)
        step_y = int(template_h * step_percent)
        
        stickers = []
        
        # Slide window across image
        for y in range(0, h - template_h, step_y):
            for x in range(0, w - template_w, step_x):
                # Extract window
                window = sheet_image[y:y+template_h, x:x+template_w]
                
                # Compare with template
                if len(window.shape) == 3 and len(self.template.shape) == 3:
                    diff = cv2.absdiff(window, self.template)
                    diff_score = np.mean(diff)
                    
                    # If similar enough
                    if diff_score < 50:  # Lower = more similar
                        confidence = 1.0 - (diff_score / 255)
                        if confidence > 0.4:
                            stickers.append({
                                'position': (x, y),
                                'confidence': confidence,
                                'method': 'sliding',
                                'bbox': (x, y, template_w, template_h)
                            })
        
        # Remove duplicates
        unique = self._non_max_suppression(stickers, min_distance=int(template_w * 0.3))
        
        return unique
    
    def detect_all_methods(self, sheet_image):
        """
        Use ALL detection methods and combine results
        """
        print("Using multiple detection methods...")
        
        # Method 1: Standard template matching
        stickers1 = self.detect_all_stickers(sheet_image)
        print(f"  Template matching: {len(stickers1)} stickers")
        
        # Method 2: Multi-scale
        stickers2 = self.detect_with_multiple_scales(sheet_image)
        print(f"  Multi-scale: {len(stickers2)} stickers")
        
        # Method 3: Sliding window
        stickers3 = self.detect_with_sliding_window(sheet_image)
        print(f"  Sliding window: {len(stickers3)} stickers")
        
        # Combine all
        all_stickers = stickers1 + stickers2 + stickers3
        
        # Remove duplicates with very small distance
        unique = self._non_max_suppression(all_stickers, min_distance=int(min(self.template_w, self.template_h) * 0.2))
        
        print(f"  Total unique: {len(unique)} stickers")
        
        return unique
    
    def _non_max_suppression(self, points, min_distance):
        """
        Non-Maximum Suppression with adjustable distance
        """
        if not points:
            return []
        
        # Sort by confidence
        points.sort(key=lambda p: p['confidence'], reverse=True)
        
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
    
    def visualize_detections(self, sheet_image, stickers, show_confidences=True):
        """
        Create detailed visualization
        """
        vis = sheet_image.copy()
        
        # Different colors for different detection methods
        colors = {
            'standard': (0, 255, 0),    # Green
            'edge': (255, 165, 0),      # Orange
            'scaled': (255, 0, 255),    # Purple
            'sliding': (0, 255, 255),   # Yellow
            'default': (0, 255, 0)
        }
        
        for i, sticker in enumerate(stickers):
            x, y, w, h = sticker['bbox']
            method = sticker.get('method', 'default')
            confidence = sticker.get('confidence', 0)
            scale = sticker.get('scale', 1.0)
            
            # Choose color
            if 'scale' in sticker and scale != 1.0:
                color = colors['scaled']
            elif method == 'edge':
                color = colors['edge']
            elif method == 'sliding':
                color = colors['sliding']
            else:
                color = colors['standard']
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            if show_confidences:
                label = f"#{i+1} ({confidence:.2f})"
                cv2.putText(vis, label, (x+5, y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis
    
    def get_grid_pattern(self, stickers, tolerance=30):
        """
        Detect grid pattern from stickers
        """
        if not stickers:
            return {'rows': 0, 'columns': 0, 'total_expected': 0, 'total_found': 0}
        
        # Get all positions
        positions = [s['position'] for s in stickers]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        # Group by rows
        rows = []
        used_y = []
        for y in sorted(y_coords):
            if not any(abs(y - uy) < tolerance for uy in used_y):
                rows.append(y)
                used_y.append(y)
        
        # Group by columns
        cols = []
        used_x = []
        for x in sorted(x_coords):
            if not any(abs(x - ux) < tolerance for ux in used_x):
                cols.append(x)
                used_x.append(x)
        
        return {
            'rows': len(rows),
            'columns': len(cols),
            'total_expected': len(rows) * len(cols),
            'total_found': len(stickers),
            'row_positions': rows,
            'col_positions': cols,
            'missing': (len(rows) * len(cols)) - len(stickers)
        }

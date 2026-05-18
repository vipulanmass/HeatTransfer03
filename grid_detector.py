import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class GridStickerDetector:
    """
    Grid-based sticker detection using Normalized Cross-Correlation (NCC)
    with geometric verification and pitch-based grid fitting
    """
    
    def __init__(self, golden_template_path, match_threshold=0.7):
        """
        Initialize detector with golden template
        
        Args:
            golden_template_path: Path to the golden sticker template
            match_threshold: Minimum NCC score (0.7 = 70% similarity)
        """
        self.load_template(golden_template_path)
        self.match_threshold = match_threshold
        
    def load_template(self, template_path):
        """Load and preprocess the golden template"""
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not load template: {template_path}")
        
        # Step 1: Convert to grayscale and reduce noise
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_gray = cv2.GaussianBlur(self.template_gray, (3, 3), 0)
        
        self.template_h, self.template_w = self.template_gray.shape
        print(f"✓ Golden template loaded: {self.template_w}x{self.template_h}")
        
    def process_sheet(self, sheet_path, output_dir="output"):
        """
        Complete processing pipeline for sheet detection
        """
        print("\n" + "=" * 80)
        print("GRID-BASED STICKER DETECTION SYSTEM")
        print("=" * 80)
        
        # Load sheet
        sheet = cv2.imread(sheet_path)
        if sheet is None:
            print(f"Error: Could not load {sheet_path}")
            return None
        
        print(f"\n📷 Sheet loaded: {sheet.shape[1]}x{sheet.shape[0]}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Preprocess sheet
        sheet_gray = self._preprocess_sheet(sheet)
        
        # Step 2: Template matching with NCC
        print("\n" + "=" * 80)
        print("STEP 1: Template Matching with Normalized Cross-Correlation (NCC)")
        print("=" * 80)
        
        correlation_map = cv2.matchTemplate(sheet_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        
        print(f"Correlation map shape: {correlation_map.shape}")
        print(f"Max correlation: {correlation_map.max():.4f}")
        print(f"Min correlation: {correlation_map.min():.4f}")
        print(f"Mean correlation: {correlation_map.mean():.4f}")
        
        # Step 3: Find local maxima (peaks)
        print("\n" + "=" * 80)
        print("STEP 2: Local Maxima Detection")
        print("=" * 80)
        
        peaks = self._find_local_maxima(correlation_map)
        print(f"Raw peaks found: {len(peaks)}")
        
        # Step 4: Filter by threshold
        peaks = [p for p in peaks if p['score'] >= self.match_threshold]
        print(f"After threshold ({self.match_threshold}): {len(peaks)} peaks")
        
        # Step 5: Non-Maximum Suppression (NMS)
        print("\n" + "=" * 80)
        print("STEP 3: Non-Maximum Suppression (NMS)")
        print("=" * 80)
        
        nms_peaks = self._non_maximum_suppression(peaks)
        print(f"After NMS: {len(nms_peaks)} unique stickers")
        
        # Step 6: Grid geometric verification
        print("\n" + "=" * 80)
        print("STEP 4: Grid Geometric Verification")
        print("=" * 80)
        
        grid_peaks, grid_info = self._fit_to_grid(nms_peaks)
        print(f"Grid detected: {grid_info['rows']} rows × {grid_info['cols']} columns")
        print(f"Row pitch: {grid_info['row_pitch']:.1f} pixels")
        print(f"Column pitch: {grid_info['col_pitch']:.1f} pixels")
        print(f"Expected stickers: {grid_info['total_expected']}")
        print(f"Found: {grid_info['total_found']}")
        
        if grid_info['missing'] > 0:
            print(f"⚠️  Missing: {grid_info['missing']} stickers (grid positions added)")
        
        # Step 7: Generate bounding boxes
        print("\n" + "=" * 80)
        print("STEP 5: Bounding Box Generation")
        print("=" * 80)
        
        stickers = self._generate_bounding_boxes(grid_peaks, sheet.shape)
        print(f"Generated {len(stickers)} bounding boxes")
        
        # Save results
        self._save_results(sheet, stickers, correlation_map, grid_info, output_dir)
        
        # Save individual stickers
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            roi = sheet[y:y+h, x:x+w]
            filename = f"{output_dir}/sticker_r{sticker['row']}_c{sticker['col']}.png"
            cv2.imwrite(filename, roi)
        
        print(f"\n✅ Complete! {len(stickers)} stickers extracted to '{output_dir}/'")
        
        return stickers, grid_info
    
    def _preprocess_sheet(self, sheet):
        """Step 1: Convert to grayscale and reduce noise"""
        if len(sheet.shape) == 3:
            gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        else:
            gray = sheet
        
        # Gaussian blur to remove noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray
    
    def _find_local_maxima(self, correlation_map):
        """
        Find local maxima in the correlation map
        Uses scipy's find_peaks for robust peak detection
        """
        # Find peaks in 2D using scipy
        from scipy.ndimage import maximum_filter
        
        # Define neighborhood size (avoid detecting multiple peaks for same sticker)
        neighborhood = max(self.template_w // 4, self.template_h // 4, 10)
        
        # Find local maxima
        local_max = maximum_filter(correlation_map, size=neighborhood) == correlation_map
        
        # Get coordinates of peaks
        peak_coords = np.argwhere(local_max)
        
        peaks = []
        for y, x in peak_coords:
            if correlation_map[y, x] > 0:  # Only positive correlations
                peaks.append({
                    'x': x,
                    'y': y,
                    'score': correlation_map[y, x]
                })
        
        # Sort by score (highest first)
        peaks.sort(key=lambda p: p['score'], reverse=True)
        
        return peaks
    
    def _non_maximum_suppression(self, peaks, min_distance=30):
        """
        Non-Maximum Suppression to remove overlapping detections
        """
        if not peaks:
            return []
        
        kept = []
        for peak in peaks:
            x, y = peak['x'], peak['y']
            
            # Check if too close to any kept peak
            too_close = False
            for kept_peak in kept:
                kx, ky = kept_peak['x'], kept_peak['y']
                distance = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                kept.append(peak)
        
        return kept
    
    def _fit_to_grid(self, peaks):
        """
        Step 4: Fit detected peaks to a grid using geometric verification
        """
        if not peaks:
            return [], {'rows': 0, 'cols': 0, 'total_expected': 0, 'total_found': 0, 
                       'missing': 0, 'row_pitch': 0, 'col_pitch': 0}
        
        # Extract x and y coordinates
        x_coords = np.array([p['x'] for p in peaks])
        y_coords = np.array([p['y'] for p in peaks])
        
        # Calculate pitches (distances between consecutive peaks)
        x_sorted = np.sort(x_coords)
        y_sorted = np.sort(y_coords)
        
        # Find unique rows by clustering Y coordinates
        row_threshold = self.template_h * 0.5
        rows = []
        row_centers = []
        
        for y in y_sorted:
            found = False
            for i, row_y in enumerate(row_centers):
                if abs(y - row_y) < row_threshold:
                    found = True
                    break
            if not found:
                row_centers.append(y)
                rows.append(y)
        
        rows.sort()
        
        # Find unique columns by clustering X coordinates
        col_threshold = self.template_w * 0.5
        cols = []
        col_centers = []
        
        for x in x_sorted:
            found = False
            for i, col_x in enumerate(col_centers):
                if abs(x - col_x) < col_threshold:
                    found = True
                    break
            if not found:
                col_centers.append(x)
                cols.append(x)
        
        cols.sort()
        
        # Calculate grid pitches
        row_pitch = np.mean([rows[i+1] - rows[i] for i in range(len(rows)-1)]) if len(rows) > 1 else self.template_h
        col_pitch = np.mean([cols[i+1] - cols[i] for i in range(len(cols)-1)]) if len(cols) > 1 else self.template_w
        
        # Create complete grid
        grid_peaks = []
        for r, row_y in enumerate(rows):
            for c, col_x in enumerate(cols):
                # Check if peak exists at this grid position
                exists = False
                peak_score = 0
                for peak in peaks:
                    if abs(peak['x'] - col_x) < self.template_w * 0.5 and abs(peak['y'] - row_y) < self.template_h * 0.5:
                        exists = True
                        peak_score = peak['score']
                        break
                
                grid_peaks.append({
                    'x': col_x,
                    'y': row_y,
                    'score': peak_score,
                    'exists': exists,
                    'row': r,
                    'col': c
                })
        
        grid_info = {
            'rows': len(rows),
            'cols': len(cols),
            'total_expected': len(rows) * len(cols),
            'total_found': len(peaks),
            'missing': (len(rows) * len(cols)) - len(peaks),
            'row_pitch': row_pitch,
            'col_pitch': col_pitch,
            'row_positions': rows,
            'col_positions': cols
        }
        
        return grid_peaks, grid_info
    
    def _generate_bounding_boxes(self, grid_peaks, sheet_shape):
        """
        Step 5: Generate bounding boxes using template dimensions
        """
        stickers = []
        
        for peak in grid_peaks:
            x_center = peak['x']
            y_center = peak['y']
            
            # Calculate bounding box using template dimensions
            x = int(x_center - self.template_w // 2)
            y = int(y_center - self.template_h // 2)
            w = self.template_w
            h = self.template_h
            
            # Ensure box is within sheet bounds
            x = max(0, min(x, sheet_shape[1] - w))
            y = max(0, min(y, sheet_shape[0] - h))
            
            stickers.append({
                'bbox': (x, y, w, h),
                'center': (x_center, y_center),
                'row': peak['row'],
                'col': peak['col'],
                'exists': peak['exists'],
                'score': peak['score']
            })
        
        return stickers
    
    def _save_results(self, sheet, stickers, correlation_map, grid_info, output_dir):
        """
        Save visualization and results
        """
        # Create visualization with bounding boxes
        vis = sheet.copy()
        
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            
            if sticker['exists']:
                color = (0, 255, 0)  # Green for detected
                thickness = 2
            else:
                color = (0, 165, 255)  # Orange for predicted
                thickness = 2
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, thickness)
            
            # Add label
            label = f"{sticker['row']},{sticker['col']}"
            cv2.putText(vis, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if sticker['exists']:
                score_text = f"{sticker['score']:.2f}"
                cv2.putText(vis, score_text, (x+5, y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(f"{output_dir}/grid_detection.png", vis)
        print(f"\n✓ Visualization saved: {output_dir}/grid_detection.png")
        
        # Save correlation map heatmap
        self._save_correlation_heatmap(correlation_map, output_dir)
        
        # Save report
        self._save_report(sheet, stickers, grid_info, output_dir)
    
    def _save_correlation_heatmap(self, correlation_map, output_dir):
        """Save correlation map as heatmap visualization"""
        plt.figure(figsize=(12, 8))
        plt.imshow(correlation_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Correlation Score')
        plt.title('Template Matching Correlation Map')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Correlation heatmap saved: {output_dir}/correlation_heatmap.png")
    
    def _save_report(self, sheet, stickers, grid_info, output_dir):
        """Generate detailed detection report"""
        report_path = f"{output_dir}/detection_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRID-BASED STICKER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DETECTION PARAMETERS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Template size: {self.template_w}x{self.template_h}\n")
            f.write(f"Match threshold: {self.match_threshold}\n\n")
            
            f.write("GRID ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Rows detected: {grid_info['rows']}\n")
            f.write(f"Columns detected: {grid_info['cols']}\n")
            f.write(f"Row pitch: {grid_info['row_pitch']:.1f} pixels\n")
            f.write(f"Column pitch: {grid_info['col_pitch']:.1f} pixels\n")
            f.write(f"Expected stickers: {grid_info['total_expected']}\n")
            f.write(f"Detected stickers: {grid_info['total_found']}\n")
            f.write(f"Missing stickers: {grid_info['missing']}\n\n")
            
            f.write("STICKER DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Row':<6} {'Col':<6} {'Status':<12} {'Score':<8} {'Position (X,Y)':<20}\n")
            f.write("-" * 60 + "\n")
            
            for sticker in stickers:
                status = "DETECTED" if sticker['exists'] else "PREDICTED"
                score = f"{sticker['score']:.3f}" if sticker['exists'] else "N/A"
                pos = f"({sticker['center'][0]:.0f}, {sticker['center'][1]:.0f})"
                f.write(f"{sticker['row']:<6} {sticker['col']:<6} {status:<12} {score:<8} {pos:<20}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("NCC FORMULA USED:\n")
            f.write("R(x,y) = Σ(T(x',y')·I(x+x',y+y')) / √(ΣT²·ΣI²)\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Report saved: {report_path}")

def main():
    """Main entry point"""
    print("=" * 80)
    print("GRID-BASED STICKER DETECTION SYSTEM")
    print("Normalized Cross-Correlation + Geometric Grid Fitting")
    print("=" * 80)
    
    # Get file paths
    template_path = input("\n📷 Enter path to GOLDEN TEMPLATE (single sticker): ").strip().strip('"')
    sheet_path = input("📷 Enter path to PRODUCTION SHEET: ").strip().strip('"')
    
    if not os.path.exists(template_path):
        print(f"\n❌ Template not found: {template_path}")
        return
    
    if not os.path.exists(sheet_path):
        print(f"\n❌ Sheet not found: {sheet_path}")
        return
    
    # Get threshold
    try:
        threshold = float(input("🎯 Match threshold (0.5-0.9, default 0.7): ").strip() or "0.7")
        threshold = max(0.3, min(0.95, threshold))
    except:
        threshold = 0.7
    
    print(f"\n🚀 Starting detection with threshold: {threshold}")
    
    # Create detector and process
    detector = GridStickerDetector(template_path, match_threshold=threshold)
    stickers, grid_info = detector.process_sheet(sheet_path)
    
    if stickers:
        print("\n" + "=" * 80)
        print("✅ DETECTION COMPLETE!")
        print("=" * 80)
        print(f"📁 Results saved to: output/")
        print(f"📊 Found {len(stickers)} stickers in {grid_info['rows']}×{grid_info['cols']} grid")
    else:
        print("\n❌ No stickers detected. Try lowering the match threshold.")

if __name__ == "__main__":
    main()

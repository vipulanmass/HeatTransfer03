import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class GridStickerDetector:
    """
    Grid-based sticker detection using multiple matching methods
    """
    
    def __init__(self, golden_template_path, match_threshold=0.3):
        """
        Initialize detector with golden template
        """
        self.load_template(golden_template_path)
        self.match_threshold = match_threshold
        
    def load_template(self, template_path):
        """Load and preprocess the golden template"""
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not load template: {template_path}")
        
        # Create multiple representations of template
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_gray = cv2.GaussianBlur(self.template_gray, (3, 3), 0)
        
        # Edge-based template
        self.template_edges = cv2.Canny(self.template_gray, 50, 150)
        
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
        
        print(f"\nSheet loaded: {sheet.shape[1]}x{sheet.shape[0]}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Preprocess sheet
        sheet_gray = self._preprocess_sheet(sheet)
        sheet_edges = cv2.Canny(sheet_gray, 50, 150)
        
        # Try multiple matching methods
        print("\n" + "=" * 80)
        print("STEP 1: Multi-Method Template Matching")
        print("=" * 80)
        
        all_peaks = []
        
        # Method 1: Grayscale NCC
        print("\nMethod 1: Grayscale NCC...")
        result1 = cv2.matchTemplate(sheet_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        peaks1 = self._find_local_maxima(result1)
        print(f"  Found {len(peaks1)} peaks, max score: {result1.max():.3f}")
        all_peaks.extend(peaks1)
        
        # Method 2: Edge-based matching (more robust)
        print("\nMethod 2: Edge-based matching...")
        result2 = cv2.matchTemplate(sheet_edges, self.template_edges, cv2.TM_CCOEFF_NORMED)
        peaks2 = self._find_local_maxima(result2)
        print(f"  Found {len(peaks2)} peaks, max score: {result2.max():.3f}")
        all_peaks.extend(peaks2)
        
        # Method 3: Lower threshold if needed
        if max(result1.max(), result2.max()) < self.match_threshold:
            print(f"\n⚠️  Low match scores. Trying lower threshold...")
            lower_threshold = max(0.25, max(result1.max(), result2.max()) - 0.05)
            print(f"  Using threshold: {lower_threshold:.3f}")
            
            # Get peaks with lower threshold
            peaks_low = self._find_local_maxima(result1, threshold=lower_threshold)
            all_peaks.extend(peaks_low)
            peaks_low2 = self._find_local_maxima(result2, threshold=lower_threshold)
            all_peaks.extend(peaks_low2)
        
        # Remove duplicates
        all_peaks = self._remove_duplicate_peaks(all_peaks)
        print(f"\nTotal unique peaks: {len(all_peaks)}")
        
        # Filter by threshold
        peaks = [p for p in all_peaks if p['score'] >= self.match_threshold]
        print(f"After threshold ({self.match_threshold}): {len(peaks)} peaks")
        
        # If still no peaks, use the best ones we have
        if len(peaks) == 0 and len(all_peaks) > 0:
            print(f"\nNo peaks above threshold. Using best {min(20, len(all_peaks))} peaks...")
            all_peaks.sort(key=lambda p: p['score'], reverse=True)
            peaks = all_peaks[:min(20, len(all_peaks))]
        
        if len(peaks) == 0:
            print("\n❌ No stickers detected! The template may not match any stickers on this sheet.")
            print("   Possible solutions:")
            print("   1. Extract a template directly from this sheet")
            print("   2. Use a different template that matches the stickers")
            return None, None
        
        # Step 2: Non-Maximum Suppression
        print("\n" + "=" * 80)
        print("STEP 2: Non-Maximum Suppression")
        print("=" * 80)
        
        min_distance = int(min(self.template_w, self.template_h) * 0.5)
        nms_peaks = self._non_maximum_suppression(peaks, min_distance)
        print(f"After NMS: {len(nms_peaks)} unique stickers")
        
        # Step 3: Grid geometric verification
        print("\n" + "=" * 80)
        print("STEP 3: Grid Geometric Verification")
        print("=" * 80)
        
        grid_peaks, grid_info = self._fit_to_grid(nms_peaks)
        
        if grid_info['rows'] > 0:
            print(f"Grid detected: {grid_info['rows']} rows x {grid_info['cols']} columns")
            print(f"Row pitch: {grid_info['row_pitch']:.1f} pixels")
            print(f"Column pitch: {grid_info['col_pitch']:.1f} pixels")
            print(f"Expected stickers: {grid_info['total_expected']}")
            print(f"Found: {grid_info['total_found']}")
            
            if grid_info['missing'] > 0:
                print(f"Missing: {grid_info['missing']} stickers (grid positions added)")
        else:
            print("No clear grid pattern detected. Using raw detections.")
            grid_peaks = [{'x': p['x'], 'y': p['y'], 'score': p['score'], 
                          'exists': True, 'row': i, 'col': 0} 
                         for i, p in enumerate(nms_peaks)]
            grid_info = {'rows': len(nms_peaks), 'cols': 1, 'total_expected': len(nms_peaks),
                        'total_found': len(nms_peaks), 'missing': 0}
        
        # Step 4: Generate bounding boxes
        print("\n" + "=" * 80)
        print("STEP 4: Bounding Box Generation")
        print("=" * 80)
        
        stickers = self._generate_bounding_boxes(grid_peaks, sheet.shape)
        print(f"Generated {len(stickers)} bounding boxes")
        
        # Save results
        self._save_results(sheet, stickers, result1, result2, grid_info, output_dir)
        
        # Save individual stickers
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            roi = sheet[y:y+h, x:x+w]
            filename = f"{output_dir}/sticker_r{sticker['row']}_c{sticker['col']}.png"
            cv2.imwrite(filename, roi)
        
        print(f"\n✅ Complete! {len(stickers)} stickers extracted to '{output_dir}/'")
        
        return stickers, grid_info
    
    def _preprocess_sheet(self, sheet):
        """Preprocess sheet"""
        if len(sheet.shape) == 3:
            gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        else:
            gray = sheet
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray
    
    def _find_local_maxima(self, correlation_map, threshold=None):
        """Find local maxima in correlation map"""
        from scipy.ndimage import maximum_filter
        
        if threshold is None:
            threshold = self.match_threshold
        
        # Define neighborhood size
        neighborhood = max(self.template_w // 3, self.template_h // 3, 15)
        
        # Find local maxima
        local_max = maximum_filter(correlation_map, size=neighborhood) == correlation_map
        
        # Get coordinates
        peak_coords = np.argwhere(local_max)
        
        peaks = []
        for y, x in peak_coords:
            score = correlation_map[y, x]
            if score >= threshold:
                peaks.append({
                    'x': int(x),
                    'y': int(y),
                    'score': float(score)
                })
        
        return peaks
    
    def _remove_duplicate_peaks(self, peaks, distance=20):
        """Remove duplicate peaks from multiple methods"""
        if not peaks:
            return []
        
        unique = []
        for peak in peaks:
            x, y = peak['x'], peak['y']
            is_duplicate = False
            
            for u in unique:
                dist = np.sqrt((x - u['x'])**2 + (y - u['y'])**2)
                if dist < distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(peak)
        
        return unique
    
    def _non_maximum_suppression(self, peaks, min_distance):
        """Non-Maximum Suppression"""
        if not peaks:
            return []
        
        peaks.sort(key=lambda p: p['score'], reverse=True)
        
        kept = []
        for peak in peaks:
            x, y = peak['x'], peak['y']
            too_close = False
            
            for kept_peak in kept:
                dist = np.sqrt((x - kept_peak['x'])**2 + (y - kept_peak['y'])**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                kept.append(peak)
        
        return kept
    
    def _fit_to_grid(self, peaks):
        """Fit peaks to a grid"""
        if not peaks or len(peaks) < 2:
            return [], {'rows': 0, 'cols': 0, 'total_expected': 0, 'total_found': 0, 
                       'missing': 0, 'row_pitch': 0, 'col_pitch': 0}
        
        # Extract coordinates
        x_coords = [p['x'] for p in peaks]
        y_coords = [p['y'] for p in peaks]
        
        # Cluster rows
        row_threshold = self.template_h * 0.6
        rows = []
        for y in sorted(y_coords):
            found = False
            for ry in rows:
                if abs(y - ry) < row_threshold:
                    found = True
                    break
            if not found:
                rows.append(y)
        
        rows.sort()
        
        # Cluster columns
        col_threshold = self.template_w * 0.6
        cols = []
        for x in sorted(x_coords):
            found = False
            for cx in cols:
                if abs(x - cx) < col_threshold:
                    found = True
                    break
            if not found:
                cols.append(x)
        
        cols.sort()
        
        # Calculate pitches
        row_pitch = np.mean([rows[i+1] - rows[i] for i in range(len(rows)-1)]) if len(rows) > 1 else self.template_h
        col_pitch = np.mean([cols[i+1] - cols[i] for i in range(len(cols)-1)]) if len(cols) > 1 else self.template_w
        
        # Create grid
        grid_peaks = []
        for r, row_y in enumerate(rows):
            for c, col_x in enumerate(cols):
                exists = False
                score = 0
                for peak in peaks:
                    if abs(peak['x'] - col_x) < self.template_w * 0.5 and abs(peak['y'] - row_y) < self.template_h * 0.5:
                        exists = True
                        score = peak['score']
                        break
                
                grid_peaks.append({
                    'x': col_x,
                    'y': row_y,
                    'score': score,
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
            'col_pitch': col_pitch
        }
        
        return grid_peaks, grid_info
    
    def _generate_bounding_boxes(self, grid_peaks, sheet_shape):
        """Generate bounding boxes"""
        stickers = []
        
        for peak in grid_peaks:
            x_center = peak['x']
            y_center = peak['y']
            
            x = int(x_center - self.template_w // 2)
            y = int(y_center - self.template_h // 2)
            w = self.template_w
            h = self.template_h
            
            # Ensure within bounds
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
    
    def _save_results(self, sheet, stickers, result1, result2, grid_info, output_dir):
        """Save visualization and results"""
        # Create visualization
        vis = sheet.copy()
        
        for sticker in stickers:
            x, y, w, h = sticker['bbox']
            
            if sticker['exists']:
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)
            
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            label = f"{sticker['row']},{sticker['col']}"
            cv2.putText(vis, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(f"{output_dir}/grid_detection.png", vis)
        print(f"\n✓ Visualization: {output_dir}/grid_detection.png")
        
        # Save heatmaps
        self._save_heatmap(result1, f"{output_dir}/heatmap_grayscale.png", "Grayscale NCC")
        self._save_heatmap(result2, f"{output_dir}/heatmap_edges.png", "Edge-based NCC")
        
        # Save report
        self._save_report(sheet, stickers, grid_info, output_dir)
    
    def _save_heatmap(self, correlation_map, filename, title):
        """Save correlation heatmap"""
        plt.figure(figsize=(12, 8))
        plt.imshow(correlation_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Correlation Score')
        plt.title(f'Template Matching - {title}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap saved: {filename}")
    
    def _save_report(self, sheet, stickers, grid_info, output_dir):
        """Save detection report"""
        report_path = f"{output_dir}/detection_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
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
            f.write(f"{'Row':<6} {'Col':<6} {'Status':<12} {'Score':<8}\n")
            f.write("-" * 40 + "\n")
            
            for sticker in stickers:
                status = "DETECTED" if sticker['exists'] else "PREDICTED"
                score = f"{sticker['score']:.3f}" if sticker['exists'] else "N/A"
                f.write(f"{sticker['row']:<6} {sticker['col']:<6} {status:<12} {score:<8}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("NCC FORMULA USED:\n")
            f.write("R(x,y) = Sum(T * I) / sqrt(Sum(T^2) * Sum(I^2))\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Report saved: {report_path}")

def main():
    """Main entry point"""
    print("=" * 80)
    print("GRID-BASED STICKER DETECTION SYSTEM")
    print("Multi-Method Template Matching")
    print("=" * 80)
    
    # Get file paths
    template_path = input("\nEnter path to GOLDEN TEMPLATE: ").strip().strip('"')
    sheet_path = input("Enter path to PRODUCTION SHEET: ").strip().strip('"')
    
    if not os.path.exists(template_path):
        print(f"\nTemplate not found: {template_path}")
        return
    
    if not os.path.exists(sheet_path):
        print(f"\nSheet not found: {sheet_path}")
        return
    
    # Get threshold
    try:
        threshold = float(input("Match threshold (0.2-0.5, default 0.3): ").strip() or "0.3")
        threshold = max(0.2, min(0.6, threshold))
    except:
        threshold = 0.3
    
    print(f"\nStarting detection with threshold: {threshold}")
    
    # Create detector and process
    detector = GridStickerDetector(template_path, match_threshold=threshold)
    stickers, grid_info = detector.process_sheet(sheet_path)
    
    if stickers and len(stickers) > 0:
        print("\n" + "=" * 80)
        print("DETECTION COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: output/")
        print(f"Found {len(stickers)} stickers")
        if grid_info and grid_info['rows'] > 0:
            print(f"Grid: {grid_info['rows']} x {grid_info['cols']}")
    else:
        print("\nNo stickers detected.")
        print("\nTroubleshooting tips:")
        print("1. Try extracting a template directly from this sheet")
        print("2. Lower the match threshold further (try 0.2)")
        print("3. Check if the sheet has clear, visible stickers")

if __name__ == "__main__":
    main()

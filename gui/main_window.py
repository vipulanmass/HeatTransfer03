# gui/main_window.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from datetime import datetime
import threading

from core.alignment import SheetAligner
from core.character_detector import CharacterDetector
from core.character_matcher import CharacterMatcher
from core.defect_classifier import DefectClassifier
from data.database import ProductDatabase
from gui.setup_wizard import ProductSetupWizard

class MainWindow:
    """Main operator interface for inspection"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Heat Transfer Sheet Inspector")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.db = ProductDatabase()
        self.aligner = SheetAligner()
        self.detector = CharacterDetector()
        self.matcher = CharacterMatcher()
        self.classifier = None  # Will be initialized when product selected
        
        self.current_product = None
        self.current_image = None
        self.current_results = None
        
        self._create_menu()
        self._create_main_layout()
        
    def _create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Product Setup...", command=self._open_setup_wizard)
        file_menu.add_separator()
        file_menu.add_command(label="Load Sheet Image...", command=self._load_sheet)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Product menu
        product_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Product", menu=product_menu)
        product_menu.add_command(label="Select Product...", command=self._select_product)
        product_menu.add_command(label="Edit Product...", command=self._edit_product)
        
        # Reports menu
        reports_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reports", menu=reports_menu)
        reports_menu.add_command(label="View Logs", command=self._view_logs)
        reports_menu.add_command(label="Export Results...", command=self._export_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Manual", command=self._show_manual)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_main_layout(self):
        """Create main application layout"""
        # Top frame: Product info and controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(top_frame, text="Product:").pack(side='left')
        self.product_label = ttk.Label(top_frame, text="Not selected", 
                                        font=('Arial', 10, 'bold'))
        self.product_label.pack(side='left', padx=5)
        
        ttk.Button(top_frame, text="Select Product", 
                   command=self._select_product).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Load Sheet", 
                   command=self._load_sheet).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Start Inspection", 
                   command=self._start_inspection).pack(side='left', padx=5)
        
        # Main content: Image display
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left: Image display
        left_frame = ttk.LabelFrame(main_frame, text="Sheet Image")
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_frame, bg='gray')
        self.image_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right: Results panel
        right_frame = ttk.LabelFrame(main_frame, text="Inspection Results")
        right_frame.pack(side='right', fill='y', padx=5, ipadx=10, ipady=10)
        
        # Results text area
        self.results_text = tk.Text(right_frame, width=40, height=20)
        self.results_text.pack(padx=5, pady=5, fill='both', expand=True)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_text)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        # Bottom: Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken')
        self.status_bar.pack(fill='x', padx=5, pady=2)
    
    def _select_product(self):
        """Select product for inspection"""
        products = self.db.list_products()
        
        if not products:
            messagebox.showinfo("No Products", 
                "No products configured. Please set up a product first.")
            self._open_setup_wizard()
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Product")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Select Product:", font=('Arial', 12)).pack(pady=10)
        
        listbox = tk.Listbox(dialog, height=10)
        listbox.pack(fill='both', expand=True, padx=20, pady=10)
        
        for product in products:
            listbox.insert(tk.END, f"{product['id']} - {product['name']} (Modified: {product['modified']})")
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                product_info = products[selection[0]]
                self.current_product = self.db.load_product(product_info['id'])
                self.product_label.config(text=f"{self.current_product.name} ({self.current_product.product_id})")
                
                # Initialize classifier with product thresholds
                self.classifier = DefectClassifier(
                    missing_threshold=self.current_product.defect_thresholds.get('missing_threshold', 0.30),
                    smear_threshold=self.current_product.defect_thresholds.get('smear_threshold', 0.10),
                    shape_threshold=self.current_product.defect_thresholds.get('shape_threshold', 0.30)
                )
                
                self.status_bar.config(text=f"Product loaded: {self.current_product.name}")
                dialog.destroy()
        
        ttk.Button(dialog, text="Select", command=on_select).pack(pady=10)
    
    def _open_setup_wizard(self):
        """Open product setup wizard"""
        wizard = ProductSetupWizard(self.root, self.db)
        self.root.wait_window(wizard.window)
    
    def _edit_product(self):
        """Edit existing product"""
        if not self.current_product:
            messagebox.showinfo("No Product", "Please select a product first")
            return
        
        # Open wizard with existing product data
        wizard = ProductSetupWizard(self.root, self.db)
        # TODO: Pre-populate with current product data
        self.root.wait_window(wizard.window)
    
    def _load_sheet(self):
        """Load a sheet image for inspection"""
        filepath = filedialog.askopenfilename(
            title="Select Sheet Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if filepath:
            self.current_image = cv2.imread(filepath)
            self._display_image(self.current_image, self.image_canvas)
            self.status_bar.config(text=f"Loaded: {os.path.basename(filepath)}")
    
    def _start_inspection(self):
        """Run inspection on loaded sheet"""
        if self.current_product is None:
            messagebox.showwarning("No Product", "Please select a product first")
            return
        
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load a sheet image first")
            return
        
        # Run inspection in background thread
        self.status_bar.config(text="Inspecting...")
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', "Running inspection...\n")
        
        thread = threading.Thread(target=self._run_inspection)
        thread.start()
    
    def _run_inspection(self):
        """Run the actual inspection (in background thread)"""
        try:
            # Load golden image
            golden = cv2.imread(self.current_product.golden_image_path)
            
            # Align sheet
            aligned = self.aligner.align(self.current_image, golden)
            
            # Results
            results = {
                'sheet_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'product': self.current_product.product_id,
                'stickers': []
            }
            
            overall_pass = True
            
            # Inspect each sticker
            for sticker in self.current_product.stickers:
                x, y, w, h = sticker.bbox
                sticker_roi = aligned[y:y+h, x:x+w]
                
                # Detect characters in current sticker
                detected_chars = self.detector.detect(sticker_roi)
                
                # Match to golden
                matches, missing, extra = self.matcher.match(
                    sticker.characters, detected_chars
                )
                
                defects = []
                
                # Check matched characters
                for match in matches:
                    golden_char = match.golden
                    detected_char = match.current
                    
                    # Extract ROI
                    cx, cy, cw, ch = detected_char.bbox
                    char_roi = sticker_roi[cy:cy+ch, cx:cx+cw]
                    
                    # Load golden mask
                    golden_mask = cv2.imread(golden_char.mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Classify
                    defect = self.classifier.classify(golden_char, char_roi, golden_mask)
                    if defect:
                        defects.append(defect)
                
                # Add missing characters
                for missing_char in missing:
                    defects.append({
                        'type': 'missing_letter',
                        'character_text': missing_char.text,
                        'severity': 1.0,
                        'bbox': missing_char.bbox
                    })
                
                # Add extra characters
                for extra_char in extra:
                    defects.append({
                        'type': 'extra_ink_smear',
                        'severity': 0.5,
                        'bbox': extra_char.bbox
                    })
                
                # Decide pass/fail
                sticker_pass = self._decide_sticker(defects)
                if not sticker_pass:
                    overall_pass = False
                
                results['stickers'].append({
                    'id': sticker.id,
                    'pass': sticker_pass,
                    'defects': [d if isinstance(d, dict) else d.__dict__ 
                               for d in defects]
                })
            
            results['overall_pass'] = overall_pass
            self.current_results = results
            
            # Update GUI in main thread
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _decide_sticker(self, defects) -> bool:
        """Decide if sticker passes inspection"""
        for defect in defects:
            # Convert Defect object to dict if needed
            defect_type = defect['type'] if isinstance(defect, dict) else defect.type
            
            if defect_type == 'missing_letter':
                return False
            if defect_type == 'missing_or_broken' and defect.get('severity', 0) > 0.5:
                return False
        
        return True
    
    def _display_results(self, results):
        """Display inspection results in GUI"""
        # Update status
        status = "PASS" if results['overall_pass'] else "REJECT"
        self.status_bar.config(text=f"Inspection complete: {status}")
        
        # Clear and update results text
        self.results_text.delete('1.0', tk.END)
        
        # Overall result
        if results['overall_pass']:
            self.results_text.insert('1.0', "✅ OVERALL: PASS\n\n", 'pass')
        else:
            self.results_text.insert('1.0', "❌ OVERALL: REJECT\n\n", 'fail')
        
        # Configure tags
        self.results_text.tag_configure('pass', foreground='green', font=('Arial', 12, 'bold'))
        self.results_text.tag_configure('fail', foreground='red', font=('Arial', 12, 'bold'))
        self.results_text.tag_configure('defect', foreground='red')
        self.results_text.tag_configure('warning', foreground='orange')
        
        # Per-sticker results
        for sticker in results['stickers']:
            if sticker['pass']:
                self.results_text.insert(tk.END, f"Sticker {sticker['id']}: ✅ PASS\n", 'pass')
            else:
                self.results_text.insert(tk.END, f"Sticker {sticker['id']}: ❌ REJECT\n", 'fail')
                
                # Show defects
                for defect in sticker['defects']:
                    if isinstance(defect, dict):
                        defect_type = defect.get('type', 'unknown')
                        char_text = defect.get('character_text', '?')
                        severity = defect.get('severity', 0)
                    else:
                        defect_type = defect.type
                        char_text = defect.character_text
                        severity = defect.severity
                    
                    self.results_text.insert(tk.END, 
                        f"  • {defect_type}: '{char_text}' (severity: {severity:.1%})\n", 
                        'defect')
            
            self.results_text.insert(tk.END, "\n")
        
        # Draw overlay on image
        self._draw_overlay(results)
    
    def _draw_overlay(self, results):
        """Draw defect overlay on image"""
        if self.current_image is None:
            return
        
        overlay = self.current_image.copy()
        
        # Draw stickers and defects
        for sticker_idx, sticker_result in enumerate(results['stickers']):
            if sticker_idx >= len(self.current_product.stickers):
                continue
                
            sticker = self.current_product.stickers[sticker_idx]
            x, y, w, h = sticker.bbox
            
            # Choose color based on result
            color = (0, 255, 0) if sticker_result['pass'] else (0, 0, 255)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
            
            # Draw defect highlights
            for defect in sticker_result['defects']:
                if isinstance(defect, dict):
                    bbox = defect.get('bbox')
                    defect_type = defect.get('type', 'unknown')
                else:
                    bbox = defect.bbox
                    defect_type = defect.type
                
                if bbox:
                    dx, dy, dw, dh = bbox
                    cv2.rectangle(overlay, (x+dx, y+dy), (x+dx+dw, y+dy+dh), 
                                 (0, 0, 255), 2)
                    cv2.putText(overlay, defect_type[:3], (x+dx, y+dy-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Update display
        self._display_image(overlay, self.image_canvas)
    
    def _display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """Display image in canvas"""
        if image is None:
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 10:
            img_w, img_h = pil_img.size
            scale = min(canvas_width / img_w, canvas_height / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Display
            photo = ImageTk.PhotoImage(pil_img)
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            canvas.image = photo  # Keep reference
    
    def _export_results(self):
        """Export inspection results"""
        if not self.current_results:
            messagebox.showinfo("No Results", "No inspection results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
        )
        
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.current_results, f, indent=2)
            messagebox.showinfo("Export Complete", f"Results saved to {filepath}")
    
    def _view_logs(self):
        """View inspection logs"""
        # TODO: Implement log viewer
        messagebox.showinfo("Logs", "Log viewer coming soon")
    
    def _show_manual(self):
        """Show user manual"""
        messagebox.showinfo("User Manual", 
            "Heat Transfer Sheet Inspector\n\n"
            "1. Set up products using 'Product Setup'\n"
            "2. Select a product\n"
            "3. Load a sheet image\n"
            "4. Click 'Start Inspection'\n"
            "5. View results\n\n"
            "For detailed instructions, refer to the documentation.")
    
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
            "Heat Transfer Sheet Inspector\n"
            "Version 1.0\n\n"
            "Pure Computer Vision Solution\n"
            "OpenCV-based character detection and defect classification")
    
    def _show_error(self, error_msg):
        """Show error message"""
        messagebox.showerror("Inspection Error", error_msg)
        self.status_bar.config(text="Error during inspection")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = MainWindow()
    app.run()
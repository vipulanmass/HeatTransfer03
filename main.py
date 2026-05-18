import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image, ImageTk

from data.database import ProductDatabase
from core.character_detector import CharacterDetector
from core.sticker_detector import StickerDetector, StickerInspector

class HeatTransferInspector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Heat Transfer Sheet Inspector - Sticker Detection Mode")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.db = ProductDatabase()
        self.golden_sticker = None
        self.current_sheet = None
        self.detected_stickers = []
        self.inspection_results = []
        
        self._create_menu()
        self._create_main_layout()
        
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Golden Sticker...", command=self.load_golden)
        file_menu.add_command(label="Load Production Sheet...", command=self.load_sheet)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        inspect_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Inspect", menu=inspect_menu)
        inspect_menu.add_command(label="Detect Stickers", command=self.detect_stickers)
        inspect_menu.add_command(label="Inspect All Stickers", command=self.inspect_all)
        inspect_menu.add_separator()
        inspect_menu.add_command(label="Export Results", command=self.export_results)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_main_layout(self):
        # Top frame
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        # Status info
        self.status_frame = ttk.LabelFrame(top_frame, text="Status")
        self.status_frame.pack(fill='x', padx=5, pady=5)
        
        self.golden_status = ttk.Label(self.status_frame, text="Golden Sticker: Not loaded", font=('Arial', 9))
        self.golden_status.pack(anchor='w', padx=5, pady=2)
        
        self.sheet_status = ttk.Label(self.status_frame, text="Production Sheet: Not loaded", font=('Arial', 9))
        self.sheet_status.pack(anchor='w', padx=5, pady=2)
        
        self.sticker_count = ttk.Label(self.status_frame, text="Stickers Detected: 0", font=('Arial', 9, 'bold'))
        self.sticker_count.pack(anchor='w', padx=5, pady=2)
        
        # Control buttons
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill='x', pady=5)
        
        ttk.Button(btn_frame, text="1. Load Golden Sticker", command=self.load_golden).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="2. Load Production Sheet", command=self.load_sheet).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="3. Detect Stickers", command=self.detect_stickers).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="4. Inspect All", command=self.inspect_all).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Export Results", command=self.export_results).pack(side='left', padx=5)
        
        # Main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Image display
        left_frame = ttk.LabelFrame(main_frame, text="Sheet with Sticker Detections")
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_frame, bg='gray')
        self.image_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results panel
        right_frame = ttk.LabelFrame(main_frame, text="Inspection Results")
        right_frame.pack(side='right', fill='y', padx=5, ipadx=10, ipady=10)
        
        # Results treeview
        self.results_tree = ttk.Treeview(right_frame, columns=('Sticker', 'Result', 'Score', 'Defects'), height=15)
        self.results_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_tree.heading('#0', text='#')
        self.results_tree.heading('Sticker', text='Sticker')
        self.results_tree.heading('Result', text='Result')
        self.results_tree.heading('Score', text='Diff Score')
        self.results_tree.heading('Defects', text='Defects')
        
        self.results_tree.column('#0', width=40)
        self.results_tree.column('Sticker', width=80)
        self.results_tree.column('Result', width=80)
        self.results_tree.column('Score', width=100)
        self.results_tree.column('Defects', width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(right_frame, orient='vertical', command=self.results_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Detail text
        self.detail_text = tk.Text(right_frame, height=8, font=('Consolas', 9))
        self.detail_text.pack(fill='x', padx=5, pady=5)
        
        # Summary label
        self.summary_label = ttk.Label(right_frame, text="", font=('Arial', 10, 'bold'))
        self.summary_label.pack(pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken')
        self.status_bar.pack(fill='x', padx=5, pady=2)
    
    def load_golden(self):
        """Load golden sticker template"""
        filepath = filedialog.askopenfilename(
            title="Select Golden Sticker Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.golden_sticker = cv2.imread(filepath)
            h, w = self.golden_sticker.shape[:2]
            self.golden_status.config(text=f"Golden Sticker: {os.path.basename(filepath)} ({w}x{h})")
            self.status_bar.config(text=f"Loaded golden sticker: {w}x{h}")
            
            # Show golden sticker preview
            self._show_golden_preview()
    
    def _show_golden_preview(self):
        """Show golden sticker in a separate window"""
        if self.golden_sticker is None:
            return
        
        preview = tk.Toplevel(self.root)
        preview.title("Golden Sticker Template")
        
        # Convert to RGB
        rgb = cv2.cvtColor(self.golden_sticker, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Resize for preview
        max_size = 400
        if pil_img.width > max_size or pil_img.height > max_size:
            scale = min(max_size / pil_img.width, max_size / pil_img.height)
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        label = ttk.Label(preview, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
        
        ttk.Label(preview, text=f"Size: {self.golden_sticker.shape[1]}x{self.golden_sticker.shape[0]}").pack()
    
    def load_sheet(self):
        """Load production sheet image"""
        filepath = filedialog.askopenfilename(
            title="Select Production Sheet Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.current_sheet = cv2.imread(filepath)
            h, w = self.current_sheet.shape[:2]
            self.sheet_status.config(text=f"Production Sheet: {os.path.basename(filepath)} ({w}x{h})")
            self.status_bar.config(text=f"Loaded production sheet: {w}x{h}")
            self._display_sheet()
    
    def detect_stickers(self):
        """Detect all stickers on the sheet"""
        if self.golden_sticker is None:
            messagebox.showwarning("No Golden", "Please load golden sticker first")
            return
        
        if self.current_sheet is None:
            messagebox.showwarning("No Sheet", "Please load production sheet first")
            return
        
        self.status_bar.config(text="Detecting stickers...")
        self.detail_text.delete('1.0', tk.END)
        self.detail_text.insert('1.0', "Detecting stickers...\n")
        self.root.update()
        
        try:
            # Create sticker detector
            detector = StickerDetector(self.golden_sticker, match_threshold=0.7)
            
            # Detect stickers
            self.detected_stickers = detector.detect_stickers(self.current_sheet)
            
            # Update display
            self.sticker_count.config(text=f"Stickers Detected: {len(self.detected_stickers)}")
            
            # Draw on sheet
            sheet_with_boxes = self.current_sheet.copy()
            for i, sticker in enumerate(self.detected_stickers):
                x, y, w, h = sticker['bbox']
                cv2.rectangle(sheet_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(sheet_with_boxes, f"#{i+1}", (x+5, y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self._display_sheet(sheet_with_boxes)
            
            self.detail_text.insert('1.0', f"✓ Found {len(self.detected_stickers)} stickers\n\n")
            for i, sticker in enumerate(self.detected_stickers):
                self.detail_text.insert(tk.END, f"Sticker {i+1}: position {sticker['position']}, confidence {sticker['confidence']:.2f}\n")
            
            self.status_bar.config(text=f"Detected {len(self.detected_stickers)} stickers")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_bar.config(text="Error during detection")
    
    def inspect_all(self):
        """Inspect all detected stickers"""
        if not self.detected_stickers:
            messagebox.showwarning("No Stickers", "Please detect stickers first")
            return
        
        self.status_bar.config(text="Inspecting stickers...")
        self.detail_text.delete('1.0', tk.END)
        self.detail_text.insert('1.0', "Inspecting stickers...\n\n")
        self.root.update()
        
        # Clear treeview
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Create inspector
        inspector = StickerInspector(self.golden_sticker)
        
        self.inspection_results = []
        passed_count = 0
        failed_count = 0
        
        for i, sticker in enumerate(self.detected_stickers):
            roi = sticker['roi']
            
            # Inspect
            result = inspector.inspect_sticker(roi)
            self.inspection_results.append({
                'index': i+1,
                'position': sticker['position'],
                'result': result
            })
            
            # Add to tree
            result_text = "✅ PASS" if result['pass'] else "❌ FAIL"
            score_text = f"{result['diff_score']:.0f}"
            defect_text = f"{len(result['defects'])} defects"
            
            if result['pass']:
                passed_count += 1
                tag = 'pass'
            else:
                failed_count += 1
                tag = 'fail'
            
            self.results_tree.insert('', 'end', 
                text=str(i+1),
                values=(f"Sticker {i+1}", result_text, score_text, defect_text),
                tags=(tag,))
            
            # Show first few defects in detail
            if result['defects']:
                self.detail_text.insert(tk.END, f"Sticker {i+1}: {len(result['defects'])} defects\n")
                for defect in result['defects'][:3]:
                    self.detail_text.insert(tk.END, f"  - {defect['type']}\n")
                self.detail_text.insert(tk.END, "\n")
        
        # Configure tags
        self.results_tree.tag_configure('pass', foreground='green')
        self.results_tree.tag_configure('fail', foreground='red')
        
        # Update summary
        total = len(self.detected_stickers)
        pass_rate = (passed_count / total) * 100 if total > 0 else 0
        summary = f"Summary: {passed_count}/{total} PASS ({pass_rate:.1f}%) | {failed_count} FAIL"
        self.summary_label.config(text=summary)
        
        # Overall sheet result
        if failed_count == 0:
            self.status_bar.config(text=f"Inspection complete: ALL {total} STICKERS PASS")
        else:
            self.status_bar.config(text=f"Inspection complete: {failed_count} STICKERS FAIL")
        
        # Draw inspection overlay
        self._draw_inspection_overlay()
    
    def _draw_inspection_overlay(self):
        """Draw inspection results on sheet"""
        if self.current_sheet is None:
            return
        
        overlay = self.current_sheet.copy()
        
        for result in self.inspection_results:
            sticker = self.detected_stickers[result['index']-1]
            x, y, w, h = sticker['bbox']
            
            # Color based on pass/fail
            color = (0, 255, 0) if result['result']['pass'] else (0, 0, 255)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
            
            # Add label
            label = f"#{result['index']}"
            cv2.putText(overlay, label, (x+5, y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        self._display_sheet(overlay)
    
    def _display_sheet(self, image=None):
        """Display sheet in canvas"""
        if image is None:
            if self.current_sheet is None:
                return
            image = self.current_sheet
        
        # Convert to RGB
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(rgb)
        
        # Resize to fit canvas
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 10:
            img_w, img_h = pil_img.size
            scale = min(canvas_width / img_w, canvas_height / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            self.image_canvas.image = photo
    
    def export_results(self):
        """Export inspection results"""
        if not self.inspection_results:
            messagebox.showinfo("No Results", "No inspection results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
        )
        
        if filepath:
            # Prepare export data
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'golden_size': self.golden_sticker.shape[:2] if self.golden_sticker is not None else None,
                'sheet_size': self.current_sheet.shape[:2] if self.current_sheet is not None else None,
                'total_stickers': len(self.detected_stickers),
                'passed_stickers': sum(1 for r in self.inspection_results if r['result']['pass']),
                'failed_stickers': sum(1 for r in self.inspection_results if not r['result']['pass']),
                'results': []
            }
            
            for result in self.inspection_results:
                export_data['results'].append({
                    'sticker_index': result['index'],
                    'position': result['position'],
                    'pass': result['result']['pass'],
                    'diff_score': result['result']['diff_score'],
                    'missing_ratio': result['result']['missing_ratio'],
                    'extra_ratio': result['result']['extra_ratio'],
                    'defects': result['result']['defects']
                })
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Results saved to {filepath}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Heat Transfer Sheet Inspector
        Version 2.0 - Sticker Detection Mode
        
        Pure Computer Vision Solution
        
        Workflow:
        1. Load golden sticker (single perfect sticker)
        2. Load production sheet (full sheet with multiple stickers)
        3. Detect all stickers on sheet
        4. Inspect each sticker against golden
        5. Export results
        
        Features:
        • Automatic sticker detection
        • Per-sticker defect analysis
        • Visual overlay with pass/fail colors
        • Export to JSON/CSV
        
        © 2025
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = HeatTransferInspector()
    app.run()

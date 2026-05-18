import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image, ImageTk

from core.sticker_segmenter import StickerSegmenter
from core.character_detector import CharacterDetector

class HeatTransferInspector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Heat Transfer Inspector - Region Segmentation")
        self.root.geometry("1400x900")
        
        # Initialize
        self.current_sheet = None
        self.segmented_stickers = []
        self.segmenter = None
        self.golden_sticker = None
        
        self._create_menu()
        self._create_main_layout()
    
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Production Sheet...", command=self.load_sheet)
        file_menu.add_command(label="Load Golden Sticker (Optional)...", command=self.load_golden)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Segment Stickers", command=self.segment_stickers)
        process_menu.add_command(label="Inspect All Stickers", command=self.inspect_all)
        process_menu.add_separator()
        process_menu.add_command(label="Export Results", command=self.export_results)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_main_layout(self):
        # Top frame
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        # Status
        self.status_frame = ttk.LabelFrame(top_frame, text="Status")
        self.status_frame.pack(fill='x', padx=5, pady=5)
        
        self.sheet_status = ttk.Label(self.status_frame, text="Sheet: Not loaded", font=('Arial', 9))
        self.sheet_status.pack(anchor='w', padx=5, pady=2)
        
        self.sticker_count = ttk.Label(self.status_frame, text="Stickers: 0", font=('Arial', 9, 'bold'))
        self.sticker_count.pack(anchor='w', padx=5, pady=2)
        
        # Control buttons
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill='x', pady=5)
        
        ttk.Button(btn_frame, text="1. Load Sheet", command=self.load_sheet).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="2. Segment Stickers", command=self.segment_stickers).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="3. Inspect All", command=self.inspect_all).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Export Results", command=self.export_results).pack(side='left', padx=5)
        
        # Main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Image display
        left_frame = ttk.LabelFrame(main_frame, text="Sheet with Stickers")
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_frame, bg='gray')
        self.image_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results panel
        right_frame = ttk.LabelFrame(main_frame, text="Results")
        right_frame.pack(side='right', fill='y', padx=5, ipadx=10, ipady=10)
        
        # Treeview for stickers
        self.tree = ttk.Treeview(right_frame, columns=('ID', 'Position', 'Size', 'Status'), height=12)
        self.tree.pack(fill='x', padx=5, pady=5)
        
        self.tree.heading('#0', text='#')
        self.tree.heading('ID', text='Sticker')
        self.tree.heading('Position', text='Position')
        self.tree.heading('Size', text='Size')
        self.tree.heading('Status', text='Status')
        
        self.tree.column('#0', width=40)
        self.tree.column('ID', width=60)
        self.tree.column('Position', width=100)
        self.tree.column('Size', width=80)
        self.tree.column('Status', width=80)
        
        # Detail text
        self.detail_text = tk.Text(right_frame, height=8, font=('Consolas', 9))
        self.detail_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Summary
        self.summary_label = ttk.Label(right_frame, text="", font=('Arial', 10, 'bold'))
        self.summary_label.pack(pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken')
        self.status_bar.pack(fill='x', padx=5, pady=2)
    
    def load_sheet(self):
        """Load production sheet"""
        filepath = filedialog.askopenfilename(
            title="Select Production Sheet",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.current_sheet = cv2.imread(filepath)
            h, w = self.current_sheet.shape[:2]
            self.sheet_status.config(text=f"Sheet: {os.path.basename(filepath)} ({w}x{h})")
            self.status_bar.config(text=f"Loaded: {os.path.basename(filepath)}")
            self._display_sheet()
    
    def load_golden(self):
        """Load golden sticker (optional for inspection)"""
        filepath = filedialog.askopenfilename(
            title="Select Golden Sticker",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.golden_sticker = cv2.imread(filepath)
            h, w = self.golden_sticker.shape[:2]
            self.status_bar.config(text=f"Loaded golden sticker: {w}x{h}")
    
    def segment_stickers(self):
        """Segment stickers using region-based method"""
        if self.current_sheet is None:
            messagebox.showwarning("No Sheet", "Please load a sheet first")
            return
        
        self.status_bar.config(text="Segmenting stickers...")
        self.detail_text.delete('1.0', tk.END)
        self.detail_text.insert('1.0', "Segmenting stickers...\n")
        self.root.update()
        
        try:
            # Calculate area ranges
            sheet_area = self.current_sheet.shape[0] * self.current_sheet.shape[1]
            min_area = int(sheet_area * 0.01)  # 1% of sheet
            max_area = int(sheet_area * 0.2)   # 20% of sheet
            
            self.detail_text.insert(tk.END, f"Area filter: {min_area} - {max_area} pixels\n")
            
            # Create segmenter
            self.segmenter = StickerSegmenter(min_area, max_area)
            
            # Segment stickers
            self.segmented_stickers, grid = self.segmenter.segment_stickers(self.current_sheet)
            
            # Update display
            self.sticker_count.config(text=f"Stickers: {len(self.segmented_stickers)}")
            
            # Create visualization
            vis = self.segmenter.visualize_segmentation(self.current_sheet, self.segmented_stickers)
            self._display_sheet(vis)
            
            # Update treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for i, sticker in enumerate(self.segmented_stickers):
                x, y, w, h = sticker['bbox']
                self.tree.insert('', 'end', text=str(i+1),
                               values=(f"#{i+1}", f"({x},{y})", f"{w}x{h}", "Pending"))
            
            # Show grid info
            self.detail_text.insert(tk.END, f"\n✓ Found {len(self.segmented_stickers)} stickers\n")
            self.detail_text.insert(tk.END, f"Grid: {grid['rows']} rows × {grid['columns']} columns\n")
            self.detail_text.insert(tk.END, f"Expected: {grid['total_expected']}\n")
            self.detail_text.insert(tk.END, f"Found: {grid['total_found']}\n")
            
            if grid['missing'] > 0:
                self.detail_text.insert(tk.END, f"⚠️  Missing {grid['missing']} stickers\n")
            
            self.status_bar.config(text=f"Segmented {len(self.segmented_stickers)} stickers")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_bar.config(text="Error during segmentation")
            import traceback
            traceback.print_exc()
    
    def inspect_all(self):
        """Inspect all segmented stickers"""
        if not self.segmented_stickers:
            messagebox.showwarning("No Stickers", "Please segment stickers first")
            return
        
        if self.golden_sticker is None:
            messagebox.showwarning("No Golden", "Please load a golden sticker for inspection")
            return
        
        self.status_bar.config(text="Inspecting stickers...")
        self.detail_text.insert(tk.END, "\nInspecting stickers...\n")
        self.root.update()
        
        # Extract ROIs
        rois = self.segmenter.get_sticker_rois(self.current_sheet, self.segmented_stickers)
        
        # Simple inspection (compare to golden)
        passed = 0
        failed = 0
        
        for i, roi_data in enumerate(rois):
            roi = roi_data['roi']
            
            # Resize to match golden if needed
            if roi.shape != self.golden_sticker.shape:
                roi = cv2.resize(roi, (self.golden_sticker.shape[1], self.golden_sticker.shape[0]))
            
            # Calculate difference
            diff = cv2.absdiff(roi, self.golden_sticker)
            diff_score = np.mean(diff)
            
            # Determine pass/fail (adjust threshold as needed)
            is_pass = diff_score < 30
            
            if is_pass:
                passed += 1
                status = "PASS"
                color = "green"
            else:
                failed += 1
                status = "FAIL"
                color = "red"
            
            # Update tree
            self.tree.item(self.tree.get_children()[i], values=(f"#{i+1}", f"({roi_data['bbox'][0]},{roi_data['bbox'][1]})", 
                                                                f"{roi_data['bbox'][2]}x{roi_data['bbox'][3]}", status))
            
            # Color the item
            self.tree.tag_configure('pass', foreground='green')
            self.tree.tag_configure('fail', foreground='red')
            
            if is_pass:
                self.tree.item(self.tree.get_children()[i], tags=('pass',))
            else:
                self.tree.item(self.tree.get_children()[i], tags=('fail',))
            
            self.detail_text.insert(tk.END, f"Sticker {i+1}: {status} (score={diff_score:.2f})\n")
        
        # Update summary
        total = len(self.segmented_stickers)
        summary = f"Summary: {passed}/{total} PASS ({passed/total*100:.1f}%) | {failed} FAIL"
        self.summary_label.config(text=summary)
        
        if failed == 0:
            self.status_bar.config(text=f"Inspection complete: ALL {total} STICKERS PASS")
        else:
            self.status_bar.config(text=f"Inspection complete: {failed} STICKERS FAIL")
    
    def export_results(self):
        """Export segmentation and inspection results"""
        if not self.segmented_stickers:
            messagebox.showinfo("No Results", "No results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
        )
        
        if filepath:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'sheet_size': self.current_sheet.shape[:2] if self.current_sheet is not None else None,
                'total_stickers': len(self.segmented_stickers),
                'stickers': []
            }
            
            for i, sticker in enumerate(self.segmented_stickers):
                x, y, w, h = sticker['bbox']
                export_data['stickers'].append({
                    'index': i+1,
                    'bbox': [x, y, w, h],
                    'area': sticker['area']
                })
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", f"Results saved to {filepath}")
    
    def _display_sheet(self, image=None):
        """Display sheet in canvas"""
        if image is None:
            if self.current_sheet is None:
                return
            image = self.current_sheet
        
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pil_img = Image.fromarray(rgb)
        
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
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Heat Transfer Inspector - Region Segmentation Mode
        
        This system uses region-based segmentation to detect stickers
        without requiring template matching.
        
        Workflow:
        1. Load production sheet
        2. Click "Segment Stickers" - automatically finds all stickers
        3. Load golden sticker (optional for inspection)
        4. Click "Inspect All" - compares each sticker to golden
        5. Export results
        
        Features:
        • No template matching needed
        • Multiple segmentation methods combined
        • Automatic grid detection
        • Visual overlay with sticker IDs
        
        © 2025
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = HeatTransferInspector()
    app.run()

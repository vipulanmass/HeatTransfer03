import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from core.sticker_detector_ncc import StickerDetectorNCC

class StickerDetectionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sticker Detection System - NCC Template Matching")
        self.root.geometry("1400x900")
        
        self.template = None
        self.sheet = None
        self.stickers = None
        self.detector = None
        self.current_vis = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Top frame
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(top_frame, text="Controls")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Row 1: File loading
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Template (Golden)", 
                  command=self.load_template).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Load Sheet", 
                  command=self.load_sheet).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Detect Stickers", 
                  command=self.detect_stickers).pack(side='left', padx=5)
        
        # Row 2: Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Match Threshold:").pack(side='left', padx=5)
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(settings_frame, from_=0.5, to=0.95, 
                                    variable=self.threshold_var, orient='horizontal', length=150)
        threshold_scale.pack(side='left', padx=5)
        self.threshold_label = ttk.Label(settings_frame, text="0.70")
        self.threshold_label.pack(side='left', padx=5)
        
        ttk.Label(settings_frame, text="Red Threshold:").pack(side='left', padx=(20, 5))
        self.red_threshold_var = tk.DoubleVar(value=0.05)
        red_scale = ttk.Scale(settings_frame, from_=0.01, to=0.20, 
                              variable=self.red_threshold_var, orient='horizontal', length=150)
        red_scale.pack(side='left', padx=5)
        self.red_label = ttk.Label(settings_frame, text="0.05")
        self.red_label.pack(side='left', padx=5)
        
        # Update labels
        threshold_scale.config(command=lambda v: self.threshold_label.config(text=f"{float(v):.2f}"))
        red_scale.config(command=lambda v: self.red_label.config(text=f"{float(v):.2f}"))
        
        # Row 3: Status
        self.status_label = ttk.Label(control_frame, text="Ready", relief='sunken')
        self.status_label.pack(fill='x', padx=5, pady=5)
        
        # Main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left: Image display
        left_frame = ttk.LabelFrame(main_frame, text="Detection Results")
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_frame, bg='gray')
        self.image_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right: Results panel
        right_frame = ttk.LabelFrame(main_frame, text="Results")
        right_frame.pack(side='right', fill='y', padx=5, ipadx=10, ipady=10)
        
        # Sticker list
        self.sticker_listbox = tk.Listbox(right_frame, height=15, width=35)
        self.sticker_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        self.sticker_listbox.bind('<<ListboxSelect>>', self.on_sticker_select)
        
        # Preview
        preview_frame = ttk.LabelFrame(right_frame, text="Preview")
        preview_frame.pack(fill='x', padx=5, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, width=200, height=150, bg='gray')
        self.preview_canvas.pack(padx=5, pady=5)
        
        # Info text
        self.info_text = tk.Text(right_frame, height=10, width=35, font=('Consolas', 9))
        self.info_text.pack(fill='x', padx=5, pady=5)
        
        # Export button
        ttk.Button(right_frame, text="Export Report", 
                  command=self.export_report).pack(pady=5)
    
    def load_template(self):
        """Load golden template"""
        filepath = filedialog.askopenfilename(
            title="Select Golden Sticker Template",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.template = cv2.imread(filepath)
            h, w = self.template.shape[:2]
            self.status_label.config(text=f"Loaded template: {w}x{h}")
            
            # Show template in a separate window
            self._show_template_preview()
    
    def load_sheet(self):
        """Load production sheet"""
        filepath = filedialog.askopenfilename(
            title="Select Production Sheet",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.sheet = cv2.imread(filepath)
            h, w = self.sheet.shape[:2]
            self.status_label.config(text=f"Loaded sheet: {w}x{h}")
            self._display_image(self.sheet)
    
    def detect_stickers(self):
        """Run sticker detection"""
        if self.template is None:
            messagebox.showwarning("No Template", "Please load a golden template first")
            return
        
        if self.sheet is None:
            messagebox.showwarning("No Sheet", "Please load a production sheet first")
            return
        
        self.status_label.config(text="Detecting stickers...")
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', "Processing...\n")
        self.root.update()
        
        try:
            # Save template temporarily
            cv2.imwrite("temp_template.png", self.template)
            
            # Create detector
            self.detector = StickerDetectorNCC(
                golden_template_path="temp_template.png",
                match_threshold=self.threshold_var.get(),
                red_threshold=self.red_threshold_var.get()
            )
            
            # Process sheet
            self.stickers, grid_info = self.detector.process_sheet(
                "temp_sheet.png", 
                output_dir="detection_output"
            )
            
            # Save sheet temporarily
            cv2.imwrite("temp_sheet.png", self.sheet)
            
            # Load visualization
            vis_path = "detection_output/detected_stickers.png"
            if os.path.exists(vis_path):
                vis = cv2.imread(vis_path)
                self.current_vis = vis
                self._display_image(vis)
            
            # Update sticker list
            self.sticker_listbox.delete(0, tk.END)
            for sticker in self.stickers:
                if 'grid_pos' in sticker:
                    r, c = sticker['grid_pos']
                    defect_marker = " ✗" if sticker.get('is_defect', False) else ""
                    self.sticker_listbox.insert(tk.END, f"Row {r}, Col {c}{defect_marker}")
                else:
                    self.sticker_listbox.insert(tk.END, f"Sticker at {sticker['position']}")
            
            # Display summary
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', f"GRID ANALYSIS\n")
            self.info_text.insert(tk.END, f"{'='*30}\n")
            self.info_text.insert(tk.END, f"Rows: {grid_info['rows']}\n")
            self.info_text.insert(tk.END, f"Columns: {grid_info['cols']}\n")
            self.info_text.insert(tk.END, f"Expected: {grid_info['total_expected']}\n")
            self.info_text.insert(tk.END, f"Found: {grid_info['total_found']}\n")
            self.info_text.insert(tk.END, f"Missing: {grid_info['missing']}\n\n")
            
            # Defect summary
            defects = [s for s in self.stickers if s.get('is_defect', False)]
            self.info_text.insert(tk.END, f"DEFECT SUMMARY\n")
            self.info_text.insert(tk.END, f"{'='*30}\n")
            self.info_text.insert(tk.END, f"Total Defects: {len(defects)}\n")
            self.info_text.insert(tk.END, f"Defect Rate: {len(defects)/len(self.stickers)*100:.1f}%\n")
            
            self.status_label.config(text=f"Detected {len(self.stickers)} stickers, {len(defects)} defects")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error during detection")
            import traceback
            traceback.print_exc()
    
    def on_sticker_select(self, event):
        """Handle sticker selection"""
        selection = self.sticker_listbox.curselection()
        if selection and self.stickers:
            idx = selection[0]
            sticker = self.stickers[idx]
            
            # Show preview
            if 'roi' in sticker:
                self._display_preview(sticker['roi'])
            
            # Show details
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', f"STICKER DETAILS\n")
            self.info_text.insert(tk.END, f"{'='*30}\n")
            
            if 'grid_pos' in sticker:
                self.info_text.insert(tk.END, f"Grid Position: Row {sticker['grid_pos'][0]}, Col {sticker['grid_pos'][1]}\n")
            
            x, y = sticker['position']
            self.info_text.insert(tk.END, f"Position: ({x}, {y})\n")
            
            if 'score' in sticker:
                self.info_text.insert(tk.END, f"Match Score: {sticker['score']:.3f}\n")
            
            self.info_text.insert(tk.END, f"Red Ratio: {sticker.get('red_ratio', 0):.3f}\n")
            self.info_text.insert(tk.END, f"Status: {'DEFECT' if sticker.get('is_defect', False) else 'OK'}\n")
            
            if sticker.get('predicted', False):
                self.info_text.insert(tk.END, f"Note: Predicted position (missing sticker)\n")
    
    def export_report(self):
        """Export inspection report"""
        if not self.stickers:
            messagebox.showwarning("No Results", "No detection results to export")
            return
        
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if export_dir:
            import shutil
            
            # Copy output files
            if os.path.exists("detection_output"):
                for file in os.listdir("detection_output"):
                    src = os.path.join("detection_output", file)
                    dst = os.path.join(export_dir, file)
                    shutil.copy2(src, dst)
            
            messagebox.showinfo("Export Complete", f"Results exported to {export_dir}")
    
    def _show_template_preview(self):
        """Show template preview"""
        if self.template is None:
            return
        
        preview = tk.Toplevel(self.root)
        preview.title("Golden Template")
        
        rgb = cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Resize
        max_size = 300
        if pil_img.width > max_size or pil_img.height > max_size:
            scale = min(max_size / pil_img.width, max_size / pil_img.height)
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        label = ttk.Label(preview, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)
        
        ttk.Label(preview, text=f"Size: {self.template.shape[1]}x{self.template.shape[0]}").pack()
    
    def _display_image(self, image):
        """Display image in canvas"""
        if image is None:
            return
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    
    def _display_preview(self, image):
        """Display preview of selected sticker"""
        if image is None:
            return
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        max_size = 180
        if pil_img.width > max_size or pil_img.height > max_size:
            scale = min(max_size / pil_img.width, max_size / pil_img.height)
            new_w = int(pil_img.width * scale)
            new_h = int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(100, 75, image=photo)
        self.preview_canvas.image = photo
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = StickerDetectionApp()
    app.run()

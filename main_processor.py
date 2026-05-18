import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from core.sticker_processor import StickerProcessor

class StickerProcessorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Heat Transfer Sticker Processor")
        self.root.geometry("1400x900")
        
        self.current_image = None
        self.processed_image = None
        self.stickers = []
        self.processor = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Top frame
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill='x', padx=10, pady=5)
        
        # Controls
        control_frame = ttk.LabelFrame(top_frame, text="Controls")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Row 1: File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Sheet", command=self.load_sheet).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Process Sheet", command=self.process_sheet).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Export Stickers", command=self.export_stickers).pack(side='left', padx=5)
        
        # Row 2: Settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        # Red removal toggle
        self.red_remove_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Remove Red Marks", 
                       variable=self.red_remove_var).pack(side='left', padx=10)
        
        # Grid settings
        ttk.Label(settings_frame, text="Rows:").pack(side='left', padx=(20, 5))
        self.rows_var = tk.StringVar(value="auto")
        rows_combo = ttk.Combobox(settings_frame, textvariable=self.rows_var, 
                                   values=["auto", "1", "2", "3", "4", "5", "6"], width=5)
        rows_combo.pack(side='left', padx=5)
        
        ttk.Label(settings_frame, text="Cols:").pack(side='left', padx=5)
        self.cols_var = tk.StringVar(value="auto")
        cols_combo = ttk.Combobox(settings_frame, textvariable=self.cols_var,
                                   values=["auto", "1", "2", "3", "4", "5", "6"], width=5)
        cols_combo.pack(side='left', padx=5)
        
        # Row 3: Status
        self.status_label = ttk.Label(control_frame, text="Ready", relief='sunken')
        self.status_label.pack(fill='x', padx=5, pady=5)
        
        # Main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left: Image display
        left_frame = ttk.LabelFrame(main_frame, text="Image Viewer")
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_frame, bg='gray')
        self.image_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right: Sticker list
        right_frame = ttk.LabelFrame(main_frame, text="Extracted Stickers")
        right_frame.pack(side='right', fill='y', padx=5, ipadx=10, ipady=10)
        
        self.sticker_listbox = tk.Listbox(right_frame, height=15, width=30)
        self.sticker_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        self.sticker_listbox.bind('<<ListboxSelect>>', self.on_sticker_select)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(right_frame, text="Preview")
        preview_frame.pack(fill='x', padx=5, pady=5)
        
        self.preview_canvas = tk.Canvas(preview_frame, width=200, height=150, bg='gray')
        self.preview_canvas.pack(padx=5, pady=5)
        
        # Info text
        self.info_text = tk.Text(right_frame, height=8, width=30)
        self.info_text.pack(fill='x', padx=5, pady=5)
    
    def load_sheet(self):
        """Load sheet image"""
        filepath = filedialog.askopenfilename(
            title="Select Sheet Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if filepath:
            self.current_image = cv2.imread(filepath)
            self._display_image(self.current_image)
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
    
    def process_sheet(self):
        """Process sheet - remove red marks and extract stickers"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load a sheet first")
            return
        
        self.status_label.config(text="Processing...")
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', "Processing sheet...\n")
        self.root.update()
        
        try:
            # Parse grid settings
            rows = None if self.rows_var.get() == "auto" else int(self.rows_var.get())
            cols = None if self.cols_var.get() == "auto" else int(self.cols_var.get())
            
            # Create processor
            self.processor = StickerProcessor(
                red_remove=self.red_remove_var.get(),
                min_sticker_width=100,
                min_sticker_height=100,
                use_fixed_grid=(rows is not None and cols is not None)
            )
            
            # Process sheet
            output_dir = "output_processing"
            self.stickers, self.sticker_data = self.processor.process_sheet(
                "temp_image.png", 
                rows, cols, 
                output_dir
            )
            
            # Save current image temporarily
            cv2.imwrite("temp_image.png", self.current_image)
            
            # Update display
            vis_path = f"{output_dir}/02_visualization.png"
            if os.path.exists(vis_path):
                vis = cv2.imread(vis_path)
                self._display_image(vis)
            
            # Update sticker list
            self.sticker_listbox.delete(0, tk.END)
            for sticker in self.stickers:
                if 'grid_pos' in sticker:
                    name = f"Row {sticker['grid_pos'][0]}, Col {sticker['grid_pos'][1]}"
                else:
                    name = f"Sticker {sticker['index']}"
                self.sticker_listbox.insert(tk.END, name)
            
            # Analyze grid
            grid_analysis = self.processor.analyze_grid(self.sticker_data)
            if grid_analysis:
                self.info_text.insert(tk.END, f"\nGrid Analysis:\n")
                self.info_text.insert(tk.END, f"  Rows: {grid_analysis['rows']}\n")
                self.info_text.insert(tk.END, f"  Columns: {grid_analysis['cols']}\n")
                self.info_text.insert(tk.END, f"  Expected: {grid_analysis['total_expected']}\n")
                self.info_text.insert(tk.END, f"  Found: {grid_analysis['total_found']}\n")
                
                if grid_analysis['missing_positions']:
                    self.info_text.insert(tk.END, f"\n⚠️ Missing stickers at:\n")
                    for r, c in grid_analysis['missing_positions']:
                        self.info_text.insert(tk.END, f"  Row {r}, Col {c}\n")
                else:
                    self.info_text.insert(tk.END, f"\n✓ All stickers detected!\n")
            
            self.status_label.config(text=f"Processed: {len(self.stickers)} stickers extracted")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error during processing")
            import traceback
            traceback.print_exc()
    
    def on_sticker_select(self, event):
        """Handle sticker selection"""
        selection = self.sticker_listbox.curselection()
        if selection and self.stickers:
            idx = selection[0]
            sticker = self.stickers[idx]
            
            # Load sticker image
            if 'filepath' in sticker:
                img = cv2.imread(sticker['filepath'])
                if img is not None:
                    self._display_preview(img)
                    
                    # Show details
                    self.info_text.delete('1.0', tk.END)
                    self.info_text.insert('1.0', f"Sticker Details:\n")
                    self.info_text.insert(tk.END, f"  Index: {sticker['index']}\n")
                    if 'grid_pos' in sticker:
                        self.info_text.insert(tk.END, f"  Grid: Row {sticker['grid_pos'][0]}, Col {sticker['grid_pos'][1]}\n")
                    x, y, w, h = sticker['bbox']
                    self.info_text.insert(tk.END, f"  Position: ({x}, {y})\n")
                    self.info_text.insert(tk.END, f"  Size: {w}x{h}\n")
                    self.info_text.insert(tk.END, f"  Area: {sticker['area']} pixels\n")
    
    def export_stickers(self):
        """Export all extracted stickers"""
        if not self.stickers:
            messagebox.showwarning("No Stickers", "Process a sheet first")
            return
        
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if export_dir:
            import shutil
            import os
            
            # Copy all sticker files
            for sticker in self.stickers:
                if 'filepath' in sticker and os.path.exists(sticker['filepath']):
                    dest = os.path.join(export_dir, os.path.basename(sticker['filepath']))
                    shutil.copy2(sticker['filepath'], dest)
            
            messagebox.showinfo("Export Complete", f"Exported {len(self.stickers)} stickers to {export_dir}")
    
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
        
        # Resize to fit preview
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
    app = StickerProcessorApp()
    app.run()

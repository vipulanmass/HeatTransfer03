# gui/setup_wizard.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime

from core.character_detector import CharacterDetector
from data.models import Product, Sticker, Character
from data.database import ProductDatabase
from core.alignment import SheetAligner

class ProductSetupWizard:
    """
    GUI wizard for setting up a new product with manual text entry
    """
    
    def __init__(self, parent, db: ProductDatabase):
        self.parent = parent
        self.db = db
        self.detector = CharacterDetector()
        self.aligner = SheetAligner()
        
        self.golden_image = None
        self.product_id = None
        self.product_name = None
        self.stickers = []
        self.current_sticker_idx = 0
        self.detected_characters = []
        
        self._create_window()
    
    def _create_window(self):
        """Create wizard window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Product Setup Wizard")
        self.window.geometry("1200x800")
        
        # Create notebook for steps
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Step 1: Product info
        self.step1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step1_frame, text="1. Product Info")
        self._create_step1()
        
        # Step 2: Load golden image
        self.step2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step2_frame, text="2. Load Golden")
        self._create_step2()
        
        # Step 3: Sticker grid
        self.step3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step3_frame, text="3. Define Stickers")
        self._create_step3()
        
        # Step 4: Character detection
        self.step4_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step4_frame, text="4. Detect Characters")
        self._create_step4()
        
        # Step 5: Manual text entry (critical)
        self.step5_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step5_frame, text="5. Enter Text")
        self._create_step5()
        
        # Step 6: Review and save
        self.step6_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.step6_frame, text="6. Review & Save")
        self._create_step6()
        
        # Navigation buttons
        self.nav_frame = ttk.Frame(self.window)
        self.nav_frame.pack(fill='x', padx=10, pady=10)
        
        self.prev_btn = ttk.Button(self.nav_frame, text="Previous", 
                                    command=self._prev_step)
        self.prev_btn.pack(side='left', padx=5)
        
        self.next_btn = ttk.Button(self.nav_frame, text="Next", 
                                    command=self._next_step)
        self.next_btn.pack(side='right', padx=5)
        
        self.cancel_btn = ttk.Button(self.nav_frame, text="Cancel", 
                                      command=self.window.destroy)
        self.cancel_btn.pack(side='right', padx=5)
        
        self.current_step = 0
        self._update_navigation()
    
    def _create_step1(self):
        """Step 1: Basic product information"""
        # Product ID
        ttk.Label(self.step1_frame, text="Product ID:", font=('Arial', 12)).pack(anchor='w', padx=20, pady=(20, 5))
        self.product_id_entry = ttk.Entry(self.step1_frame, width=30, font=('Arial', 11))
        self.product_id_entry.pack(anchor='w', padx=20, pady=5)
        
        ttk.Label(self.step1_frame, text="Example: P001, BATCH_2025, CUSTOMER_ABC", 
                  font=('Arial', 9), foreground='gray').pack(anchor='w', padx=20)
        
        # Product Name
        ttk.Label(self.step1_frame, text="Product Name:", font=('Arial', 12)).pack(anchor='w', padx=20, pady=(20, 5))
        self.product_name_entry = ttk.Entry(self.step1_frame, width=30, font=('Arial', 11))
        self.product_name_entry.pack(anchor='w', padx=20, pady=5)
        
        # Instructions
        instructions = tk.Text(self.step1_frame, height=10, width=80, bg='#f0f0f0')
        instructions.pack(pady=40)
        instructions.insert('1.0', """
        Setup Instructions:
        
        1. Enter a unique Product ID (letters, numbers, underscores only)
        2. Enter a descriptive Product Name
        3. Click Next to load the golden sample image
        
        The wizard will guide you through:
        - Loading a perfect reference sheet
        - Defining sticker locations
        - Detecting characters automatically
        - Entering character text manually
        - Saving the configuration
        """)
        instructions.config(state='disabled')
    
    def _create_step2(self):
        """Step 2: Load golden image"""
        # Image display area
        self.golden_canvas = tk.Canvas(self.step2_frame, width=800, height=500, bg='gray')
        self.golden_canvas.pack(pady=20)
        
        # Load button
        self.load_btn = ttk.Button(self.step2_frame, text="Load Golden Image", 
                                    command=self._load_golden_image)
        self.load_btn.pack(pady=10)
        
        # Status label
        self.golden_status = ttk.Label(self.step2_frame, text="No image loaded", font=('Arial', 10))
        self.golden_status.pack()
    
    def _create_step3(self):
        """Step 3: Define sticker grid"""
        # Instructions
        ttk.Label(self.step3_frame, text="Define Sticker Locations", 
                  font=('Arial', 14, 'bold')).pack(pady=10)
        
        ttk.Label(self.step3_frame, text="Draw rectangles around each sticker on the sheet", 
                  font=('Arial', 10)).pack()
        
        # Image canvas with drawing capability
        self.sticker_canvas = tk.Canvas(self.step3_frame, width=800, height=500, bg='gray')
        self.sticker_canvas.pack(pady=10)
        
        # Drawing tools frame
        tool_frame = ttk.Frame(self.step3_frame)
        tool_frame.pack(pady=5)
        
        self.add_sticker_btn = ttk.Button(tool_frame, text="Add Sticker (Draw Rectangle)", 
                                          command=self._enable_drawing)
        self.add_sticker_btn.pack(side='left', padx=5)
        
        self.clear_stickers_btn = ttk.Button(tool_frame, text="Clear All", 
                                              command=self._clear_stickers)
        self.clear_stickers_btn.pack(side='left', padx=5)
        
        # Sticker list
        self.sticker_listbox = tk.Listbox(self.step3_frame, height=5, width=50)
        self.sticker_listbox.pack(pady=10)
        
        self.drawing_enabled = False
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        self.sticker_canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.sticker_canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.sticker_canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
    
    def _create_step4(self):
        """Step 4: Automatic character detection"""
        # Instructions
        ttk.Label(self.step4_frame, text="Automatic Character Detection", 
                  font=('Arial', 14, 'bold')).pack(pady=10)
        
        ttk.Label(self.step4_frame, text="Click 'Detect Characters' to automatically find all characters", 
                  font=('Arial', 10)).pack()
        
        # Split view for sticker selection and character display
        main_frame = ttk.Frame(self.step4_frame)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Sticker selector
        left_frame = ttk.LabelFrame(main_frame, text="Stickers")
        left_frame.pack(side='left', fill='y', padx=5)
        
        self.sticker_selector = ttk.Combobox(left_frame, state='readonly', width=20)
        self.sticker_selector.pack(padx=10, pady=10)
        self.sticker_selector.bind('<<ComboboxSelected>>', self._on_sticker_select)
        
        self.detect_btn = ttk.Button(left_frame, text="Detect Characters", 
                                      command=self._detect_characters)
        self.detect_btn.pack(padx=10, pady=10)
        
        # Right: Character display
        right_frame = ttk.LabelFrame(main_frame, text="Detected Characters")
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.char_canvas = tk.Canvas(right_frame, width=600, height=400, bg='gray')
        self.char_canvas.pack(padx=10, pady=10)
        
        self.char_count_label = ttk.Label(right_frame, text="Characters detected: 0")
        self.char_count_label.pack()
    
    def _create_step5(self):
        """Step 5: Manual text entry (CRITICAL)"""
        # Instructions
        ttk.Label(self.step5_frame, text="Manual Character Text Entry", 
                  font=('Arial', 14, 'bold')).pack(pady=10)
        
        instructions = tk.Text(self.step5_frame, height=4, width=80, bg='#ffffcc')
        instructions.pack(pady=5)
        instructions.insert('1.0', """
        IMPORTANT: Enter the correct text for each detected character.
        This ensures accurate defect detection (e.g., missing 'R' vs missing 'B').
        Click on each character to edit its text.
        """)
        instructions.config(state='disabled')
        
        # Character grid view
        self.text_entry_frame = ttk.LabelFrame(self.step5_frame, text="Characters")
        self.text_entry_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.char_grid_canvas = tk.Canvas(self.text_entry_frame, bg='white')
        self.char_grid_canvas.pack(fill='both', expand=True)
        
        self.char_entries = []  # List of (canvas_id, character_id, text_var)
        
        # Save button for this sticker
        self.save_text_btn = ttk.Button(self.step5_frame, text="Save Text for Current Sticker",
                                        command=self._save_character_texts)
        self.save_text_btn.pack(pady=10)
    
    def _create_step6(self):
        """Step 6: Review and save"""
        # Summary
        ttk.Label(self.step6_frame, text="Product Configuration Summary", 
                  font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.summary_text = tk.Text(self.step6_frame, height=15, width=80)
        self.summary_text.pack(pady=20)
        
        # Save button
        self.save_btn = ttk.Button(self.step6_frame, text="Save Product Configuration", 
                                    command=self._save_product)
        self.save_btn.pack(pady=10)
    
    def _load_golden_image(self):
        """Load golden image from file"""
        filepath = filedialog.askopenfilename(
            title="Select Golden Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if filepath:
            self.golden_image = cv2.imread(filepath)
            self.golden_image_path = filepath
            
            # Display in canvas
            self._display_image(self.golden_image, self.golden_canvas)
            self.golden_status.config(text=f"Loaded: {os.path.basename(filepath)}")
            
            # Also display in sticker canvas
            self._display_image(self.golden_image, self.sticker_canvas)
    
    def _enable_drawing(self):
        """Enable rectangle drawing mode"""
        self.drawing_enabled = True
        self.add_sticker_btn.config(relief='sunken')
    
    def _clear_stickers(self):
        """Clear all defined stickers"""
        self.stickers = []
        self.sticker_listbox.delete(0, tk.END)
        # Redraw image without rectangles
        self._display_image(self.golden_image, self.sticker_canvas)
    
    def _on_mouse_down(self, event):
        """Start drawing rectangle"""
        if not self.drawing_enabled:
            return
        self.start_x = self.sticker_canvas.canvasx(event.x)
        self.start_y = self.sticker_canvas.canvasy(event.y)
        if self.rect_id:
            self.sticker_canvas.delete(self.rect_id)
    
    def _on_mouse_move(self, event):
        """Update rectangle during drag"""
        if not self.drawing_enabled or self.start_x is None:
            return
        
        current_x = self.sticker_canvas.canvasx(event.x)
        current_y = self.sticker_canvas.canvasy(event.y)
        
        if self.rect_id:
            self.sticker_canvas.delete(self.rect_id)
        
        self.rect_id = self.sticker_canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline='red', width=2
        )
    
    def _on_mouse_up(self, event):
        """Finish drawing rectangle and add sticker"""
        if not self.drawing_enabled or self.start_x is None:
            return
        
        end_x = self.sticker_canvas.canvasx(event.x)
        end_y = self.sticker_canvas.canvasy(event.y)
        
        # Convert canvas coordinates to image coordinates
        x1, y1 = int(self.start_x), int(self.start_y)
        x2, y2 = int(end_x), int(end_y)
        
        # Ensure proper order
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        # Add sticker
        sticker_id = len(self.stickers)
        self.stickers.append(Sticker(id=sticker_id, bbox=[x, y, w, h], characters=[]))
        
        # Update listbox
        self.sticker_listbox.insert(tk.END, f"Sticker {sticker_id}: ({x}, {y}, {w}x{h})")
        
        # Reset drawing mode
        self.drawing_enabled = False
        self.add_sticker_btn.config(relief='raised')
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        # Update sticker selector
        self._update_sticker_selector()
    
    def _update_sticker_selector(self):
        """Update the sticker selector combobox"""
        sticker_names = [f"Sticker {i}" for i in range(len(self.stickers))]
        self.sticker_selector['values'] = sticker_names
        if sticker_names:
            self.sticker_selector.current(0)
    
    def _on_sticker_select(self, event):
        """Handle sticker selection change"""
        self.current_sticker_idx = self.sticker_selector.current()
        self._display_sticker_characters()
    
    def _detect_characters(self):
        """Run character detection on current sticker"""
        if not self.stickers:
            messagebox.showwarning("No Stickers", "Please define stickers first")
            return
        
        sticker = self.stickers[self.current_sticker_idx]
        x, y, w, h = sticker.bbox
        
        # Extract sticker ROI from golden image
        sticker_roi = self.golden_image[y:y+h, x:x+w]
        
        # Detect characters
        detected = self.detector.detect(sticker_roi)
        
        # Store in sticker object
        self.detected_characters = detected
        self.char_count_label.config(text=f"Characters detected: {len(detected)}")
        
        # Display detected characters
        self._display_detected_characters(detected)
        
        # Update text entry grid
        self._update_text_entry_grid(detected)
    
    def _display_detected_characters(self, characters):
        """Display detected characters on canvas"""
        if not characters:
            return
        
        # Clear canvas
        self.char_canvas.delete("all")
        
        # Get sticker ROI
        sticker = self.stickers[self.current_sticker_idx]
        x, y, w, h = sticker.bbox
        sticker_roi = self.golden_image[y:y+h, x:x+w]
        
        # Convert to PIL for display
        roi_rgb = cv2.cvtColor(sticker_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        # Resize to fit canvas
        canvas_width = self.char_canvas.winfo_width()
        canvas_height = self.char_canvas.winfo_height()
        
        if canvas_width > 10:
            scale = min(canvas_width / w, canvas_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Display image
            self.char_photo = ImageTk.PhotoImage(pil_img)
            self.char_canvas.create_image(canvas_width//2, canvas_height//2, 
                                          image=self.char_photo)
            
            # Draw bounding boxes
            for char in characters:
                cx, cy, cw, ch = char.bbox
                # Scale coordinates
                sx = canvas_width//2 - new_w//2 + cx * scale
                sy = canvas_height//2 - new_h//2 + cy * scale
                ex = sx + cw * scale
                ey = sy + ch * scale
                
                self.char_canvas.create_rectangle(sx, sy, ex, ey, 
                                                  outline='green', width=2)
                self.char_canvas.create_text(sx + 5, sy + 5, 
                                             text=str(char.id), 
                                             fill='green', anchor='nw')
    
    def _update_text_entry_grid(self, characters):
        """Create grid for manual text entry"""
        # Clear existing
        self.char_grid_canvas.delete("all")
        self.char_entries.clear()
        
        if not characters:
            return
        
        # Get sticker ROI for display
        sticker = self.stickers[self.current_sticker_idx]
        x, y, w, h = sticker.bbox
        sticker_roi = self.golden_image[y:y+h, x:x+w]
        
        # Convert to RGB for display
        roi_rgb = cv2.cvtColor(sticker_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        # Calculate grid layout
        canvas_width = self.char_grid_canvas.winfo_width()
        canvas_height = self.char_grid_canvas.winfo_height()
        
        if canvas_width > 10:
            scale = min(canvas_width / w, canvas_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Display image
            self.grid_photo = ImageTk.PhotoImage(pil_img)
            self.char_grid_canvas.create_image(canvas_width//2, canvas_height//2, 
                                               image=self.grid_photo)
            
            # Create text entry fields for each character
            for char in characters:
                cx, cy, cw, ch = char.bbox
                # Scale coordinates
                sx = canvas_width//2 - new_w//2 + cx * scale
                sy = canvas_height//2 - new_h//2 + cy * scale
                ex = sx + cw * scale
                ey = sy + ch * scale
                
                # Draw bounding box
                self.char_grid_canvas.create_rectangle(sx, sy, ex, ey, 
                                                       outline='blue', width=2)
                
                # Create text entry (stored as variable)
                text_var = tk.StringVar()
                text_entry = tk.Entry(self.char_grid_canvas, textvariable=text_var, 
                                      width=3, font=('Arial', 12), justify='center')
                
                # Position entry in the center of bounding box
                entry_window = self.char_grid_canvas.create_window(
                    (sx + ex) // 2, (sy + ey) // 2,
                    window=text_entry, width=40, height=30
                )
                
                self.char_entries.append({
                    'char_id': char.id,
                    'canvas_id': entry_window,
                    'var': text_var,
                    'bbox': (sx, sy, ex, ey)
                })
    
    def _save_character_texts(self):
        """Save manually entered text for current sticker"""
        if not self.detected_characters:
            messagebox.showwarning("No Characters", "No characters detected yet")
            return
        
        # Create character objects with entered text
        characters = []
        for entry in self.char_entries:
            text = entry['var'].get().strip()
            if not text:
                text = "?"  # Placeholder for empty
            
            # Find corresponding detected character
            detected = next((c for c in self.detected_characters 
                           if c.id == entry['char_id']), None)
            if detected:
                # Precompute Hu moments for this character
                hu_moments = self._compute_hu_moments(detected.mask)
                
                # Save character mask
                mask_path = f"data/golden/{self.product_id}_sticker{self.current_sticker_idx}_char{detected.id}.png"
                cv2.imwrite(mask_path, detected.mask)
                
                char = Character(
                    id=detected.id,
                    sticker_id=self.current_sticker_idx,
                    text=text.upper(),
                    bbox=detected.bbox,
                    mask_path=mask_path,
                    components=detected.components,
                    hu_moments=hu_moments,
                    area=detected.area
                )
                characters.append(char)
        
        # Update sticker
        self.stickers[self.current_sticker_idx].characters = characters
        
        messagebox.showinfo("Success", f"Saved {len(characters)} characters for sticker {self.current_sticker_idx}")
    
    def _compute_hu_moments(self, mask: np.ndarray) -> list:
        """Compute Hu moments for shape comparison"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            hu = cv2.HuMoments(moments).flatten()
            # Log transform
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            return hu.tolist()
        return [0] * 7
    
    def _save_product(self):
        """Save complete product configuration"""
        if not self.product_id:
            messagebox.showerror("Error", "Product ID not set")
            return
        
        # Create product object
        product = Product(
            product_id=self.product_id,
            name=self.product_name,
            golden_image_path=self.golden_image_path,
            stickers=self.stickers,
            defect_thresholds={
                'missing_threshold': 0.30,
                'smear_threshold': 0.10,
                'shape_threshold': 0.30
            },
            created_date=datetime.now().isoformat(),
            modified_date=datetime.now().isoformat()
        )
        
        # Save to database
        self.db.save_product(product)
        
        messagebox.showinfo("Success", f"Product {self.product_id} saved successfully!")
        self.window.destroy()
    
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
    
    def _prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.notebook.select(self.current_step)
            self._update_navigation()
    
    def _next_step(self):
        """Go to next step"""
        # Validate current step
        if not self._validate_step():
            return
        
        if self.current_step < 5:
            self.current_step += 1
            self.notebook.select(self.current_step)
            self._update_navigation()
            
            # Update step content
            if self.current_step == 2:
                self._display_image(self.golden_image, self.sticker_canvas)
            elif self.current_step == 5:
                self._update_summary()
    
    def _validate_step(self) -> bool:
        """Validate current step before proceeding"""
        if self.current_step == 0:
            # Validate product info
            self.product_id = self.product_id_entry.get().strip()
            self.product_name = self.product_name_entry.get().strip()
            
            if not self.product_id:
                messagebox.showwarning("Missing Info", "Please enter Product ID")
                return False
            if not self.product_name:
                messagebox.showwarning("Missing Info", "Please enter Product Name")
                return False
            
            # Check for duplicate ID
            existing = self.db.load_product(self.product_id)
            if existing:
                if not messagebox.askyesno("Duplicate", 
                    f"Product {self.product_id} already exists. Overwrite?"):
                    return False
            
            return True
        
        elif self.current_step == 1:
            # Validate golden image loaded
            if self.golden_image is None:
                messagebox.showwarning("No Image", "Please load a golden image")
                return False
            return True
        
        elif self.current_step == 2:
            # Validate stickers defined
            if not self.stickers:
                messagebox.showwarning("No Stickers", 
                    "Please define at least one sticker by drawing rectangles")
                return False
            return True
        
        elif self.current_step == 3:
            # Validate characters detected
            for sticker in self.stickers:
                if not sticker.characters:
                    messagebox.showwarning("Missing Characters", 
                        f"Sticker {sticker.id} has no characters detected.\n"
                        "Please detect characters and enter text")
                    return False
            return True
        
        return True
    
    def _update_navigation(self):
        """Update navigation buttons"""
        self.prev_btn.config(state='normal' if self.current_step > 0 else 'disabled')
        self.next_btn.config(text='Finish' if self.current_step == 5 else 'Next')
    
    def _update_summary(self):
        """Update summary text"""
        summary = f"""
Product ID: {self.product_id}
Product Name: {self.product_name}
Golden Image: {os.path.basename(self.golden_image_path)}

Stickers: {len(self.stickers)}

"""
        for sticker in self.stickers:
            summary += f"\nSticker {sticker.id}:\n"
            summary += f"  Position: ({sticker.bbox[0]}, {sticker.bbox[1]}) {sticker.bbox[2]}x{sticker.bbox[3]}\n"
            summary += f"  Characters: {len(sticker.characters)}\n"
            summary += f"  Text: {''.join([c.text for c in sticker.characters])}\n"
        
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', summary)
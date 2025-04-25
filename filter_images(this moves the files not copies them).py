import tkinter as tk
from PIL import Image, ImageTk
import os
import shutil
import argparse
from tkinter import ttk

class ImageMoverApp:
    def __init__(self, root, source_dir, dest_dir):
        self.root = root
        self.root.title("Image Mover")
        
        # Validate directories
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory {source_dir} does not exist")
        if not os.path.exists(dest_dir):
            raise ValueError(f"Destination directory {dest_dir} does not exist")
            
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Gather all image files
        self.image_paths = sorted([
            os.path.join(self.source_dir, f)
            for f in os.listdir(self.source_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ])

        self.current_index = 0

        # Create a label to display the images
        self.label = ttk.Label(self.main_frame)
        self.label.pack(expand=True, fill=tk.BOTH)

        # Create status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Bind keys
        self.root.bind("<Up>", self.prev_image)      # Up arrow -> previous image
        self.root.bind("<Down>", self.next_image)    # Down arrow -> next image
        self.root.bind("<Right>", self.move_image)   # Right arrow -> move current image
        self.root.bind("<Configure>", self.on_resize) # Window resize event
        
        # Store the current displayed image
        self.current_pil_image = None
        self.current_tk_image = None
        
        # Display the first image, if any
        self.display_image()
        
        # Update window title with counts
        self.update_title()

    def update_title(self):
        """Update window title with current image count"""
        total = len(self.image_paths)
        current = self.current_index + 1 if self.image_paths else 0
        self.root.title(f"Image Mover ({current}/{total})")

    def fit_image_to_screen(self, img):
        """Fit image to current window size while maintaining aspect ratio"""
        # Get window dimensions
        window_width = self.label.winfo_width()
        window_height = self.label.winfo_height()
        
        if window_width <= 1 or window_height <= 1:  # Window not properly initialized yet
            window_width = 800
            window_height = 600

        # Get image dimensions
        img_width, img_height = img.size

        # Calculate scaling factors
        width_ratio = window_width / img_width
        height_ratio = window_height / img_height
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new dimensionsx
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def on_resize(self, event=None):
        """Handle window resize events"""
        if self.current_pil_image:
            self.display_image(force_reload=True)

    def display_image(self, force_reload=False):
        """Displays the image at the current index (if it exists)."""
        try:
            if 0 <= self.current_index < len(self.image_paths):
                img_path = self.image_paths[self.current_index]
                
                # Only reload the image if necessary
                if force_reload or self.current_pil_image is None:
                    self.current_pil_image = Image.open(img_path)
                
                # Fit to screen
                resized_img = self.fit_image_to_screen(self.current_pil_image)
                
                # Convert to Tkinter-friendly image
                self.current_tk_image = ImageTk.PhotoImage(resized_img)
                self.label.config(image=self.current_tk_image)
                
                # Update status bar
                filename = os.path.basename(img_path)
                img_size = os.path.getsize(img_path) / 1024  # Size in KB
                self.status_var.set(f"Image: {filename} | Size: {img_size:.1f}KB | {self.current_index + 1}/{len(self.image_paths)}")
            else:
                self.label.config(image="")
                self.status_var.set("No more images.")
                self.current_pil_image = None
                self.current_tk_image = None
            
            self.update_title()
            
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")

    def next_image(self, event=None):
        """Go to the next image (Down arrow)."""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.current_pil_image = None  # Force reload of new image
            self.display_image()
            return 'break'  # Prevent default event handling

    def prev_image(self, event=None):
        """Go to the previous image (Up arrow)."""
        if self.current_index > 0:
            self.current_index -= 1
            self.current_pil_image = None  # Force reload of new image
            self.display_image()
            return 'break'  # Prevent default event handling

    def move_image(self, event=None):
        """Move the current image from source to destination (Right arrow)."""
        try:
            if 0 <= self.current_index < len(self.image_paths):
                current_path = self.image_paths[self.current_index]
                filename = os.path.basename(current_path)
                dest_path = os.path.join(self.dest_dir, filename)

                # Move the file
                shutil.move(current_path, dest_path)
                self.status_var.set(f"Moved {filename} to destination directory")

                # Remove the file from the list
                del self.image_paths[self.current_index]

                # Don't adjust index since we want to see the next image
                # Only adjust if we're at the end of the list
                if self.current_index >= len(self.image_paths):
                    self.current_index = len(self.image_paths) - 1

                # Force reload of next image
                self.current_pil_image = None
                self.display_image()
                return 'break'
                
        except Exception as e:
            self.status_var.set(f"Error moving image: {str(e)}")

def main(source_dir,dest_dir):
    # Set up argument parser
   
    root = tk.Tk()
    root.geometry("800x600")  # Set initial window size
    
    try:
        app = ImageMoverApp(root, source_dir, dest_dir)
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    source_dir ='/Users/dhirajdaga/Gdrive/backup/doubts_data_profiling/multi_image_analysis/all_images'
    dest_dir = '/Users/dhirajdaga/Gdrive/backup/doubts_data_profiling/multi_image_analysis/incomplete'

    main(source_dir,dest_dir)

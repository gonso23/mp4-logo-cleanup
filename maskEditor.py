import os
import tkinter as tk
from tkinter import Canvas, NW
from PIL import Image, ImageTk, ImageDraw

# Try to import the function for video generation from logo_remover_SIFT.
try:
    from logo_remover_SIFT import frames_to_video_ffmpeg
except ImportError:
    frames_to_video_ffmpeg = None

class MaskEditor:
    def __init__(self, root, frame_dir, mask_dir, cleaned_dir, scale=0.5):
        # Initialize directories and scaling factor
        self.frame_dir = frame_dir
        self.mask_dir = mask_dir
        self.cleaned_dir = cleaned_dir
        self.scale = scale

        # Get a sorted list of all PNG frame files
        self.frame_files = sorted([f for f in os.listdir(self.frame_dir) if f.endswith(".png")])
        self.total_frames = len(self.frame_files)
        self.current_index = 0
        self.undo_stack = []
        self.brush_size = 5
        self.output_video = "output_video.mp4"

        # Create mask directory if it does not exist
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

        # Set up the main window
        self.root = root
        self.root.title("Mask Editor")

        # Create the canvas for displaying images
        self.canvas = Canvas(root)
        self.canvas.pack()

        # Set up the button panel
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill=tk.X, pady=5)

        # Left panel: Undo and mask copy buttons
        self.left_frame = tk.Frame(self.button_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.undo_button = tk.Button(self.left_frame, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT)
        self.undo_all_button = tk.Button(self.left_frame, text="Undo All", command=self.undo_all)
        self.undo_all_button.pack(side=tk.LEFT)
        self.copy_prev_button = tk.Button(self.left_frame, text="Copy Prev Mask", command=self.copy_prev_mask)
        self.copy_prev_button.pack(side=tk.LEFT, padx=5)
        self.copy_next_button = tk.Button(self.left_frame, text="Copy Next Mask", command=self.copy_next_mask)
        self.copy_next_button.pack(side=tk.LEFT)

        # Center panel: Frame navigation and status
        self.center_frame = tk.Frame(self.button_frame)
        self.center_frame.pack(side=tk.LEFT, expand=True)
        self.prev_button = tk.Button(self.center_frame, text="Previous", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.status_entry = tk.Entry(self.center_frame, width=10, justify='center')
        self.status_entry.pack(side=tk.LEFT, padx=2)
        self.status_entry.bind('<Return>', self.goto_frame)
        self.status_label = tk.Label(self.center_frame, text=f"/ {self.total_frames - 1:05}")
        self.status_label.pack(side=tk.LEFT, padx=2)
        self.next_button = tk.Button(self.center_frame, text="Next", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT)
        # Button to generate the output video (currently just a callback wrapper)
        self.generate_video_button = tk.Button(
            self.center_frame, text="Generate Video", command=self.on_generate_video
        )
        self.generate_video_button.pack(side=tk.LEFT, padx=5)

        # Right panel: Brush size controls
        self.right_frame = tk.Frame(self.button_frame)
        self.right_frame.pack(side=tk.RIGHT, padx=10)
        self.brush_label = tk.Label(self.right_frame, text=f"Brush: {self.brush_size}")
        self.brush_label.pack(side=tk.LEFT, padx=5)
        self.increase_button = tk.Button(self.right_frame, text="Increase", command=self.increase_brush)
        self.increase_button.pack(side=tk.LEFT)
        self.decrease_button = tk.Button(self.right_frame, text="Decrease", command=self.decrease_brush)
        self.decrease_button.pack(side=tk.LEFT)

        # Bind mouse and keyboard events for interactive editing
        self.canvas.bind("<B1-Motion>", self.paint)  # Draw with left mouse button held
        self.canvas.bind("<Button-1>", self.paint)   # Draw with left mouse button click

        self.root.bind("<Escape>", lambda e: self.undo_all())
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Up>', lambda e: self.increase_brush())
        self.root.bind('<Down>', lambda e: self.decrease_brush())

        # Load the first frame on startup
        self.load_frame(self.current_index)

    def load_frame(self, index):
        """
        Load the frame and corresponding mask at the given index.
        If the mask does not exist, create a blank one.
        """
        self.current_index = index
        frame_path = os.path.join(self.frame_dir, self.frame_files[index])
        mask_path = os.path.join(self.mask_dir, self.frame_files[index])

        # Load the frame image (RGB)
        self.frame_image = Image.open(frame_path).convert("RGB")
        # Load or create the mask image (grayscale)
        if os.path.exists(mask_path):
            self.mask_image = Image.open(mask_path).convert("L")
        else:
            self.mask_image = Image.new("L", self.frame_image.size, 0)
            self.mask_image.save(mask_path)
        self.original_mask = self.mask_image.copy()

        # Scale images for display
        self.display_frame = self.frame_image.resize(
            (int(self.frame_image.width * self.scale), int(self.frame_image.height * self.scale))
        )
        self.display_mask = self.mask_image.resize(
            (int(self.mask_image.width * self.scale), int(self.mask_image.height * self.scale))
        )

        # Overlay the mask in red on the frame for visualization
        self.display_image = self.display_frame.copy()
        red_mask = Image.new("RGB", self.display_mask.size, (255, 0, 0))
        self.display_image.paste(red_mask, mask=self.display_mask)

        # Update the canvas with the new image
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

        self.draw = ImageDraw.Draw(self.mask_image)
        self.undo_stack.clear()
        self.update_status_entry()

    def save_mask(self):
        """
        Save the current mask to disk and optionally run inpainting if available.
        """
        mask_path = os.path.join(self.mask_dir, self.frame_files[self.current_index])
        self.mask_image.save(mask_path)
        try:
            from logo_remover_SIFT import inpaint_frame
            frame_path = os.path.join(self.frame_dir, self.frame_files[self.current_index])
            inpaint_frame(frame_path, mask_path, self.cleaned_dir)
        except ImportError:
            print("logo_remover_SIFT.py or inpaint_frame not found.")
        except Exception as e:
            print(f"Error during inpainting: {e}")

    def next_frame(self):
        """
        Save the current mask and move to the next frame.
        """
        self.save_mask()
        if self.current_index + 1 < self.total_frames:
            self.load_frame(self.current_index + 1)
        self.update_status_entry()

    def prev_frame(self):
        """
        Save the current mask and move to the previous frame.
        """
        self.save_mask()
        if self.current_index - 1 >= 0:
            self.load_frame(self.current_index - 1)
        self.update_status_entry()

    def paint(self, event):
        """
        Draw a filled ellipse (brush stroke) at the mouse position on the mask.
        """
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        bbox = [x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size]
        self.undo_stack.append(self.mask_image.copy())
        self.draw.ellipse(bbox, fill=255)
        self.update_display()

    def undo(self):
        """
        Undo the last paint operation.
        """
        if self.undo_stack:
            self.mask_image = self.undo_stack.pop()
            self.draw = ImageDraw.Draw(self.mask_image)
            self.update_display()

    def undo_all(self):
        """
        Reset the mask to its original state for the current frame.
        """
        self.mask_image = self.original_mask.copy()
        self.draw = ImageDraw.Draw(self.mask_image)
        self.undo_stack.clear()
        self.update_display()

    def update_display(self):
        """
        Update the displayed image on the canvas to reflect the current mask.
        """
        self.display_mask = self.mask_image.resize(
            (int(self.mask_image.width * self.scale), int(self.mask_image.height * self.scale))
        )
        self.display_image = self.display_frame.copy()
        red_mask = Image.new("RGB", self.display_mask.size, (255, 0, 0))
        self.display_image.paste(red_mask, mask=self.display_mask)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def update_status_entry(self):
        """
        Update the frame number entry widget to show the current index.
        """
        self.status_entry.delete(0, tk.END)
        self.status_entry.insert(0, str(self.current_index))

    def goto_frame(self, event=None):
        """
        Jump to a specific frame based on the number entered by the user.
        """
        try:
            frame_num = int(self.status_entry.get())
            if 0 <= frame_num < self.total_frames:
                self.save_mask()
                self.load_frame(frame_num)
                self.update_status_entry()
        except ValueError:
            pass

    def increase_brush(self):
        """
        Increase the brush size for painting masks.
        """
        self.brush_size += 1
        self.brush_label.config(text=f"Brush: {self.brush_size}")

    def decrease_brush(self):
        """
        Decrease the brush size, ensuring it doesn't go below 1.
        """
        if self.brush_size > 1:
            self.brush_size -= 1
            self.brush_label.config(text=f"Brush: {self.brush_size}")

    def copy_prev_mask(self):
        """
        Copy the mask from the previous frame to the current frame.
        """
        if self.current_index > 0:
            prev_mask_path = os.path.join(self.mask_dir, self.frame_files[self.current_index - 1])
            if os.path.exists(prev_mask_path):
                prev_mask = Image.open(prev_mask_path).convert("L")
                self.mask_image = prev_mask.copy()
                self.draw = ImageDraw.Draw(self.mask_image)
                self.update_display()

    def copy_next_mask(self):
        """
        Copy the mask from the next frame to the current frame.
        """
        if self.current_index + 1 < self.total_frames:
            next_mask_path = os.path.join(self.mask_dir, self.frame_files[self.current_index + 1])
            if os.path.exists(next_mask_path):
                next_mask = Image.open(next_mask_path).convert("L")
                self.mask_image = next_mask.copy()
                self.draw = ImageDraw.Draw(self.mask_image)
                self.update_display()

    def on_generate_video(self):
        """
        Save the current mask, close the editor window, and trigger video generation.
        """
        self.save_mask()
        self.root.destroy()
        self.post_generate_video()

    def post_generate_video(self):
        """
        Called after the Tkinter window is closed to generate the output video.
        """
        if frames_to_video_ffmpeg is None:
            print("frames_to_video_ffmpeg function not found.")
            return
        print(f"Generating video to {self.output_video}...")
        try:
            frames_to_video_ffmpeg(self.cleaned_dir, self.output_video)
            print("Video generated successfully.")
        except Exception as e:
            print(f"Error generating video: {e}")

if __name__ == "__main__":
    import atexit
    frame_dir = "./frames"
    mask_dir = "./masks"
    cleaned_dir = "./cleaned_frames"
    root = tk.Tk()
    app = MaskEditor(root, frame_dir, mask_dir, cleaned_dir, scale=0.5)
    # Register video generation to run after the GUI closes
    atexit.register(app.post_generate_video)
    root.mainloop()

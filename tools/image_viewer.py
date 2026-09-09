import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

import ctypes

# import pprint
# pprint.pprint(tk.EventType.__members__)

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception as e:
    print(f"DPI設定エラー: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple image viewer")
    parser.add_argument("--path", type=str, help="Image path")
    return parser.parse_args()
    
class ImageViewerApp:
    def __init__(self, root, path):
        self.root = root
        self.root.title("PIL + Tkinter Image Viewer")
        self.root.geometry("1024x1024")
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))

        self.root.bind("<KeyPress>", self.on_keypress)

        # Canvas for image display
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Store image references
        self.image = None
        self.image_id = None
        self.tk_image = None
        self.sub_window = None

        # Bind resize event
        self.root.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Visibility>", self.on_visibility)

        # ドラッグ用変数
        self.drag_data = {"x": 0, "y": 0}

        self.image_path = path

    def on_visibility(self, event):
        if self.image_path is not None and self.image is None:
            self.load_image(self.image_path)
        
    def on_press(self, event):
        """マウス押下時の処理"""
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag(self, event):
        """ドラッグ中の処理"""
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        # self.canvas.move(self.image_id, dx, dy)
        self.move_shape(self.image_id, dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def move_shape(self, shape, dx, dy):
        """Move the shape by dx, dy but keep it inside the canvas."""
        x1, y1, x2, y2 = self.canvas.bbox(shape)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = self.image.width
        image_height = self.image.height

        if x1 + dx > 0:
            dx = -x1
        if y1 + dy > 0 and canvas_height < image_height:
            dy = -y1
        if x2 + dx < canvas_width:
            dx = canvas_width - x2
        if y2 + dy < canvas_height:
            dy = canvas_height - y2

        if canvas_width > image_width:
            dx = 0
        if canvas_height > image_height:
            dy = 0

        self.canvas.move(shape, dx, dy)

    def on_release(self, event):
        """マウスボタンを離したときの処理"""
        self.drag_data = {"x": 0, "y": 0}

    def on_keypress(self, e):
        move_distance = 128
        if e.keysym == "o":
            self.get_sub_window()
            self.sub_window.lift(self.root)
        elif e.keysym in ["q", "Escape"]:
            exit()
        elif e.keysym == "f":
            self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
        elif e.keysym == "r":
            img = self.resize_image()
            self.display_image(img)
        elif e.keysym in ["Down", "j"]:
            self.move_shape(self.image_id, 0, -move_distance)
        elif e.keysym in ["Up", "k"]:
            self.move_shape(self.image_id, 0, move_distance)
        elif e.keysym in ["Left", "h"]:
            self.move_shape(self.image_id, move_distance, 0)
        elif e.keysym in ["Right", "l"]:
            self.move_shape(self.image_id, -move_distance, 0)
        elif "a" <= e.keysym <= "z":
            exit()

    def get_sub_window(self):
        # メインWindowに紐づくサブWindowを作成する。
        self.sub_window = tk.Toplevel()
        # サブWindowへタイトルをつける。
        self.sub_window.title('subWindow')

        window_width = 1200
        window_height = 55
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.sub_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.sub_window.overrideredirect(True)

        self.label = tk.Label(self.sub_window, text="Path: ", font=("", 20))
        self.label.pack(padx=5, pady=1, side=tk.LEFT)
        
        self.text = tk.Text(
            self.sub_window,
            font=("", 20),
            width=1200,
            height=1,
            undo=True
        )
        self.text.pack(padx=1, pady=1, side=tk.LEFT)
        self.text.focus_set()
        self.text.bind("<KeyPress>", self.on_textbox_key)
        
    def on_textbox_key(self, e):
        # if e.keycode == 13: #enter
        if e.keysym == "Return": #enter
            self.load_image(self.text.get('1.0', 'end - 1 chars'))
        elif e.keysym == "Escape":
            self.sub_window.destroy()
        
    def load_image(self, file_path):
        try:
            img = Image.open(file_path)
            self.image = img
            self.display_image(self.image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

        if self.sub_window is not None:
            self.sub_window.destroy()

    def resize_image(self):
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        x1, y1, x2, y2 = self.canvas.bbox(self.image_id)
        if canvas_width == x2 - x1 or canvas_height == y2 - y1:
            return self.image

        # Resize image to fit canvas while keeping aspect ratio
        img_copy = self.image.copy()
        img_copy.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        return img_copy
        
    def display_image(self, image):
        """Display the image on the canvas, scaled to fit."""
        if self.image is None:
            return

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")  # Clear previous image
        self.image_id = self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.tk_image, anchor=tk.CENTER)
        # self.image_id = self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.CENTER)
        # self.image_id = self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        # イベントバインド
        self.canvas.tag_bind(self.image_id, "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind(self.image_id, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(self.image_id, "<ButtonRelease-1>", self.on_release)

    def on_resize(self, event):
        """Redraw image when window is resized."""
        if self.image is None:
            return

        x1, y1, x2, y2 = self.canvas.bbox(self.image_id)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = self.image.width
        image_height = self.image.height
        if x1 > 0:
            x1 = 0
        if y1 > 0:
            y1 = 0
        if x2 < canvas_width:
            x1 = canvas_width - image_width
        if y2 < canvas_height:
            y1 = canvas_height - image_height

        if image_width < canvas_width:
            x1 = (canvas_width - image_width) // 2
        if image_height < canvas_height:
            y1 = (canvas_height - image_height) // 2
        self.canvas.moveto(self.image_id, x1, y1)
        

if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = ImageViewerApp(root, args.path)
    root.mainloop()

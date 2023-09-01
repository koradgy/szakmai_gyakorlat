import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from torchvision import transforms
import torch
from emnistnet import emnistnet
import matplotlib.pyplot as plt

# Modell betöltése
model = emnistnet(printtoggle=False)
model.load_state_dict(torch.load('Data/Model/torch.pth'))
model.eval()

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        # Rajzterület létrehozása
        self.canvas = tk.Canvas(self.root, bg="white", width=384, height=96)
        self.canvas.pack()

        # Rajzterület törlése gomb
        clear_button = Button(self.root, text="Clear", command=self.clear_canvas)
        clear_button.pack()

        # Értelmezés gomb
        predict_button = Button(self.root, text="Predict", command=self.predict)
        predict_button.pack()
        
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack()

        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.image = Image.new("L", (384, 96), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Eseménykezelők hozzárendelése a vászonhoz
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=5)
            self.draw.line([(self.last_x, self.last_y), (x, y)], fill="black", width=5)
            self.last_x = x
            self.last_y = y

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (384, 96), "white")
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_image(self, image):
        inverted_image = ImageOps.invert(image)
        enhanced_image = ImageEnhance.Contrast(inverted_image).enhance(16.0)
        threshold = 200
        thresholded_image = enhanced_image.point(lambda p: p > threshold and 255)
        edge_image = thresholded_image.filter(ImageFilter.FIND_EDGES)
        edge_image = edge_image.filter(ImageFilter.MaxFilter(15))
        image = image.resize((28, 28), Image.ANTIALIAS).convert("L")
        image_array = np.array(image)
        image_array = image_array / 255.0  
        image_array = 1.0 - image_array  
        image_array = np.expand_dims(image_array, axis=0) 
        return image_array

    def decode(self, output):
        character_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        _, predicted = torch.max(output, 1)
        predicted = predicted.view(-1)
        decoded = [character_set[p] for p in predicted]
        return "".join(decoded)

    def predict(self):
        preprocessed_image = self.preprocess_image(self.image)
        
        input_image = torch.tensor(preprocessed_image, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_image)
        
       #input_image_np = input_image.squeeze().numpy()
        #plt.imshow(input_image_np, cmap='gray')
        #plt.show()

        predicted_text = self.decode(output)
        self.result_label.config(text=f"Predicted Text: {predicted_text}")

# GUI indítása
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()

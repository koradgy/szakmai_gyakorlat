from tensorflow.keras.layers import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageEnhance, ImageFilter, Image, ImageDraw, ImageOps
from tkinter import messagebox
import cv2

# Model betöltése
model = tf.keras.models.load_model("Data/model/")

class HandwritingRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognizer")

        # Rajzterület létrehozása
        self.canvas = tk.Canvas(self.root, bg="white", width=160, height=640)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Gomb létrehozása a predikció indításához
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack()

        # Gomb létrehozása a rajz törléséhez
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # RajzoSlás változók
        self.drawing = False
        self.image = Image.new("L", (160, 640), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Eredmény megjelenítése
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack()

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=5)
            self.draw.line((self.last_x, self.last_y, x, y), fill="black", width=5)
            self.last_x = x
            self.last_y = y

    def stop_draw(self, event):
        self.drawing = False

    def preprocess_image(self, image):
        # Invertálás
        inverted_image = ImageOps.invert(image)
        enhanced_image = ImageEnhance.Contrast(inverted_image).enhance(10.0)
        threshold = 200
        thresholded_image = enhanced_image.point(lambda p: p > threshold and 255)
        edge_image = thresholded_image.filter(ImageFilter.FIND_EDGES)
        edge_image = edge_image.filter(ImageFilter.MaxFilter(3))

        return edge_image

    def predict(self):
        preprocessed_image = self.preprocess_image(self.image)
        image = preprocessed_image.resize((32, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)

        label_seq = np.zeros((1, 1))

        predictions = model.predict([image_array, label_seq])

        decoded_sequence, _ = keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1])
        output = keras.backend.get_value(decoded_sequence)[0][0]
        predicted_text = ''.join([chr(x) for x in output if x != -1])
        messagebox.showinfo("Prediction Result", f"Predicted Text: {predicted_text}")
        
        image_array = image_array.squeeze() 
        plt.imshow(image_array, cmap='gray')
        plt.title("Értelmezett Kép")
        plt.axis('off')  
        plt.show()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (160, 640), "white")
        self.draw = ImageDraw.Draw(self.image)

# GUI indítása
root = tk.Tk()
app = HandwritingRecognizerApp(root)
root.mainloop()

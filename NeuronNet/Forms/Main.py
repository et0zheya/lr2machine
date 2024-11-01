import csv
import tkinter as tk

import numpy as np

from Service import recognition
from Service.CreateDataset import transformation_array
from .NeuroNet import NeuralNetworkWindow
import os
from dotenv import load_dotenv
from PIL import Image
import io

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing Application")
        self.master.geometry("400x300")



        # Canvas setup
        frame = tk.Frame(self.master)
        frame.pack(expand=True)

        self.canvas = tk.Canvas(frame, width=200, height=200, bg='white')
        self.canvas.pack(padx=16, pady=16)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        self.recognition_button = tk.Button(frame, text="Recognize", command=self.picture_recognition)
        self.recognition_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.neural_network_button = tk.Button(frame, text="Neural Network", command=self.open_neural_network_window)
        self.neural_network_button.pack(side=tk.LEFT, padx=5)

        # Text box для вывода
        self.output_text = tk.StringVar()
        self.output_text.set("")  # Initial empty string
        font = ('Helvetica', 10)  # Пример размера шрифта
        self.output_field = tk.Entry(frame, textvariable=self.output_text, state='disabled', width=40, font=font)
        self.output_field.pack(side=tk.BOTTOM, padx=5)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    def picture_recognition(self):
        # Get the postscript data from the canvas
        ps = self.canvas.postscript(colormode='color')

        # Use PIL to open and save the image
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.convert("RGB")  # Convert to RGB if necessary

        load_dotenv()
        const_width = int(os.getenv('const_width'))

        img = img.resize((const_width, const_width))
        # Save as BMP
        path_name = "Files/drawing"
        img.save(path_name + ".bmp", "BMP")

        img_gray = img.convert('L')

        result = []
        result.append(transformation_array(np.array(img_gray)))

        with open(path_name + ".csv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            writer.writerow(result[0])

        result = recognition(path_name + ".csv")

        self.output_text.set(f"Result:{result}!")

    def clear_canvas(self):
        self.canvas.delete("all")

    def open_neural_network_window(self):
        NeuralNetworkWindow(self.master)

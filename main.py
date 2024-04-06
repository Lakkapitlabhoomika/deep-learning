from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tensorflow as tf

import os
main_data_dir = 'dataset'
label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}
label_mapping

# Load the trained model
model = tf.keras.models.load_model('plant_identification_model2.h5')
def open_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        image = Image.open(file_path)
        image = image.resize((256, 256))  # Resize the image
       
        photo = ImageTk.PhotoImage(image)
        
        # Display the image
        image_label.config(image=photo)
        image_label.image = photo
        
        # Display image information
        info_text.delete(1.0, tk.END)
        # info_text.insert(tk.END, f"File: {file_path}\n")
        # info_text.insert(tk.END, f"Size: {image.size}\n")
        # info_text.insert(tk.END, f"Format: {image.format}\n")
        # info_text.insert(tk.END, f"Mode: {image.mode}")
        image = load_img(file_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        preprocessed_image = preprocess_input(image_array)
        predictions = model.predict(preprocessed_image)
    
        # Map model's numeric predictions to labels
        predicted_label_index = np.argmax(predictions)
        # print(predicted_label_index)
        predicted_label = label_mapping[predicted_label_index]
        confidence = predictions[0][predicted_label_index]
        def read_file(predicted_label_index):
            file_mapping = {
                0: 'datainfo/Alpinia Galanga (Rasna).txt',
                1: 'datainfo/Amaranthus Viridis (Arive-Dantu).txt',
            }
        
            file_path = file_mapping.get(predicted_label_index)
        
            
            with open(file_path, 'r') as file:
                info = file.read()
                info_text.insert(tk.END, f"Predicted Label: {predicted_label} ")
                info_text.insert(tk.END, f"Confidence: {confidence:.2f} ")
                info_text.insert(tk.END, info)
                # print(f"Content of {file_path}:\n")
                # print(info)
                
    
        # Example usage:
        
        read_file(predicted_label_index)
        
        
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")
        
        
# Create the main window
root = tk.Tk()
root.title("Image Viewer")

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a text widget to display image information
info_text = tk.Text(root, height=5, width=100)
info_text.pack()

# Run the Tkinter event loop
root.mainloop()

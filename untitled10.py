from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tensorflow as tf
app = Flask(__name__)
# Load the trained model
main_data_dir = 'dataset'
label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}
model = tf.keras.models.load_model('malaria_model_10epochs.h5')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def classify_image(file_path):
 image = Image.open(file_path)
 image = image.resize((256, 256))
 image = load_img(file_path, target_size=(224, 224))
 image_array = img_to_array(image)
 image_array = np.expand_dims(image_array, axis=0)
 preprocessed_image = preprocess_input(image_array)
 predictions = model.predict(preprocessed_image)
 predicted_label_index = np.argmax(predictions)
 
 if file_path:
 with open('D:/cell_images', 'r') as file:
 return file.read()
 else:
 return "Reference information not available."
@app.route('/', methods=['GET', 'POST'])
def index():
 if not os.path.exists(app.config['UPLOAD_FOLDER']):
 os.makedirs(app.config['UPLOAD_FOLDER'])
 if request.method == 'POST':
 if 'file' not in request.files:
 return render_template('index.html', error='No file part')
 file = request.files['file']
 if file.filename == '':
 return render_template('index.html', error='No selected file')
 if file and allowed_file(file.filename):
 filename = secure_filename(file.filename)
 file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
 file.save(file_path)
 predicted_label, confidence, predicted_label_index = classify_image(file_path)
 reference_info = read_reference_file(predicted_label_index)
 return render_template('index.html', file_name=filename,
 predicted_label=predicted_label, confidence=confidence,
 reference_info=reference_info)
 return render_template('index.html')
@app.route('/uploads/<filename>')
def uploaded_file(filename):
 return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if _name_ == '_main_':
 app.run(debug=True)
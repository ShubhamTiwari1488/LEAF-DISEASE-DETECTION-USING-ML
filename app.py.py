from flask import Flask, request, render_template, jsonify,url_for
#from keras import *
import tensorflow as tf
#from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
#import tensorflow.python
#from tensorflow.python.keras.saving.save import load_model

from keras.models import load_model
#from keras.optimizers import Adam

#custom_objects = {'Custom>Adam': Adam}

app = Flask(__name__)

model_cnn = load_model('C:/Users/KIIT/Desktop/FLASK/model1.h5')  # Replace with the path to your pre-trained model
#loaded_model = tf.saved_model.load('model1.h5')
classes = ['Bacterial_spot',
           'Early_blight',
           'Late_blight',
           'Leaf_Mold',
           'Septoria_leaf_spot',
           'Spider_mites Two-spotted_spider_mite',
           'Target_Spot',
           'Tomato_Yellow_Leaf_Curl_Virus',
           'Tomato_mosaic_virus',
           'healthy',
           'powdery_mildew']

@app.route('/')
def index():
     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #if request.method == 'POST':
        # Get the image file from the request
    image_file = request.files['image']

        # Load the image using PIL
    image = Image.open(image_file.stream)

        # Preprocess the image
    image = image.resize((256, 256))  # Resize the image to the input size of the model
    image = np.asarray(image)  # Convert the PIL image to a numpy array
    image = image.astype('float32') / 255.0  # Normalize the pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

        # Make a prediction using the model
        #prediction = model_cnn.predict(image)
    pred = model_cnn.predict(image)
    label = np.argmax(pred)

        # Return the result as a JSON response
    return render_template('result.html',prediction=classes[label]) 

    # If the request method is GET, show the upload form
    #return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import pickle
from flask import Flask, request, render_template_string
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the model
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
with open('cnn_tuned_model.pkl', 'rb') as file:
    model = pickle.load(file)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>CNN Image Classifier</title>
</head>
<body>
    <h1>Upload an Image (32x32)</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Classify">
    </form>
    {% if prediction %}
        <h2>Prediction: {{ prediction | safe }}</h2>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        if file:
            img = image.load_img(file, target_size=(32, 32))
            img_array = image.img_to_array(img) / 255.0
            img_array = img_array.reshape(1, 32, 32, 3)
            pred = model.predict(img_array)
            prediction = class_names[np.argmax(pred)]
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

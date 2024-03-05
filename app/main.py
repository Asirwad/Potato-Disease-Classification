from tkinter import filedialog
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
import numpy as np
from PIL import Image

class_names = ['early blight', 'late blight', 'healthy']


def predict(model, img):
    img_array = img_to_array(img=img)
    img_array = expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


print("Select the  image ")
filename = filedialog.askopenfilename(title="Select Image", filetypes=(
("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")))
print("The image selected is : ", filename)

image = Image.open(filename)
model = load_model("models/potato.h5")
predicted_class, confidence = predict(model=model, img=image)
print("Potato is in", predicted_class, "state.")
print("Confidence : ", confidence)


import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("ingredient_model.h5")

class_names = ['onion', 'potato', 'tomato']

img_path = input("dataset\tomato\tomato1.jpg ")

img = tf.keras.preprocessing.image.load_img(
    "dataset\tomato\tomato1.jpg",
    target_size=(224, 224)
)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = predictions[0]

predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print("Prediction:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))

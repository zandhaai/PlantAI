import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. Load the trained model
model = tf.keras.models.load_model("C:\\Users\\evert\\PycharmProjects\\PlantAI\\PlantAITrainer\\plant_recognition_model.keras")

# 2. Preprocess the new image
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)  # Load image and resize
    img_array = image.img_to_array(img)                      # Convert to array
    img_array = np.expand_dims(img_array, axis=0)            # Add batch dimension
    img_array /= 255.0                                       # Normalize the image
    return img_array

# 3. Predict the class of a new image
def predict_image_class(model, img_path, class_indices):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)               # Get the model's predictions
    predicted_class = np.argmax(predictions, axis=1)         # Get the index of the highest probability
    predicted_label = class_indices[predicted_class[0]]       # Get the class label
    return predicted_label

# Example usage:
if __name__ == "__main__":
    # Path to the new image you want to classify
    new_image_path = "C:\\Users\\evert\\PycharmProjects\\PlantAI\\PlantAIPredictor\\dataset\\watermelon20.jpg"

    # Assuming you used flow_from_directory, the class indices can be found from your training generator
    class_indices = {0: 'waterapple', 1: 'watermelon'}  # Replace this with the actual class indices from training

    # Predict the class
    predicted_label = predict_image_class(model, new_image_path, class_indices)

    print(f'The predicted plant class is: {predicted_label}')

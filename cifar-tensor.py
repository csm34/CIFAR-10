import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('cifar10_cnn_model.keras')

# Define the CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define a function to preprocess the input image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to 32x32
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction function
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return f"Prediction: {predicted_class} (Confidence: {confidence:.2f})"

# Create the Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="CIFAR-10 Image Classifier",
    description="Upload an image of a CIFAR-10 category (e.g., airplane, cat, dog), and the model will classify it."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

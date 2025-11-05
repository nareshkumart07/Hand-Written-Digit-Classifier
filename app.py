import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io

# --- Model Definition ---
# I've defined a network architecture that matches the layers
# found in your .pth file (conv1, conv2, fc1, fc2, fc3).
# This is a classic LeNet-style architecture.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        # 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Input features: 16 channels * 4x4 feature map size
        # (28x28 -> conv1(5x5) -> 24x24 -> pool1(2x2) -> 12x12 -> conv2(5x5) -> 8x8 -> pool2(2x2) -> 4x4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# --- Utility Functions ---

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_model(model_path):
    """
    Loads the trained PyTorch model from the specified path.
    """
    try:
        model = Net()
        # Load the saved weights. We use map_location='cpu' to ensure it
        # runs on any machine, even if it was trained on a GPU.
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'.")
        st.error(f"Please make sure '{model_path}' is in the same directory as app.py.")
        st.error("You can generate this file by running 'train.py'.")
        return None
    except RuntimeError as e:
        st.error("Error loading model weights.")
        st.error("This usually means the defined model architecture (class Net) does not")
        st.error("exactly match the architecture used to save the weights.")
        st.error(f"Details: {e}")
        return None

def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image to be compatible with the MNIST model.
    - Converts to grayscale
    - Resizes to 28x28
    - Inverts colors (model expects white digit on black background)
    - Normalizes
    """
    try:
        # Open the image
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to grayscale

        # --- Handle Color Inversion ---
        # MNIST is white digit on black bg. User images are usually black digit on white bg.
        # We check the average pixel value. If it's high (>128), it's likely
        # a bright background (white paper), so we invert it.
        img_array = np.array(img)
        if np.mean(img_array) > 128:
            img = ImageOps.invert(img)

        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Define the same transformations as the MNIST dataset
        # Normalization values (mean, std) for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Apply transform and add batch dimension (1, 1, 28, 28)
        tensor = transform(img).unsqueeze(0)
        
        # Convert tensor back to PIL Image for display
        # We need to un-normalize to show it correctly
        display_img = transforms.ToPILImage()(tensor.squeeze(0) * 0.3081 + 0.1307)

        return tensor, display_img
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# --- Streamlit App ---

st.set_page_config(page_title="Hand Written Digit Classifier", layout="wide")
st.title("🧠 Hand Written Digit Classifier")

# Load the model
model = load_model('mnist_cnn_model.pth')

if model:
    st.success("Model 'mnist_cnn_model.pth' loaded successfully!")
    st.info("""
        **Welcome!** Try to predict a handwritten digit.
        
        1.  **Upload** an image file.
        2.  Or, use your **Camera** to take a picture of a digit.
        
        **For best results:** Use a clear, centered, single digit (like '7') 
        on a plain background. The app will process it into a 28x28 image 
        to match the model's training data.
    """)

    image_data = None
    
    tab1, tab2 = st.tabs(["📁 Upload an Image", "📸 Use Camera"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_data = uploaded_file.getvalue()

    with tab2:
        camera_input = st.camera_input("Take a picture of a single digit")
        if camera_input:
            image_data = camera_input.getvalue()

    if image_data:
        # --- Process and Predict ---
        st.divider()
        st.header("Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image_data, caption="Your Image", use_column_width=True)
        
        tensor, display_img = preprocess_image(image_data)
        
        if tensor is not None:
            with col2:
                st.image(display_img, caption="Preprocessed (28x28)", use_column_width=True)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                probabilities = F.softmax(output, dim=1)
                top_prob, top_class = probabilities.topk(1)
                
                pred_digit = top_class.item()
                confidence = top_prob.item()

            st.subheader(f"Predicted Digit: {pred_digit}")
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
            
            if confidence < 0.75:
                st.warning("""
                    **Low Confidence:** The model is not very sure.
                    This can happen if:
                    - The image is not a digit.
                    - The digit is unclear or poorly written.
                    - There are multiple digits or other objects in the image.
                """)
            
            # --- Show Probability Chart ---
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame(
                probabilities.numpy().flatten(),
                index=[str(i) for i in range(10)],
                columns=["Probability"]
            )
            st.bar_chart(prob_df)
else:
    st.error("Model could not be loaded. The application cannot continue.")
    st.info("Please run `python train.py` to create the 'mnist_cnn_model.pth' file.")

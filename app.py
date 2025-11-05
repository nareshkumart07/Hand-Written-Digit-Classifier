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
# This class MUST EXACTLY match the one in your 'model_training.py'
class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2) # 28x28 -> 14x14
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1 ,padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) # 14x14 -> 7x7
    self.fc1 = nn.Linear(32*7*7,128) # 32 channels * 7 * 7 image size
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)
    self.relu = nn.ReLU()
    # self.sigmoid = nn.Sigmoid() # Not used for 10-class output

  def forward(self ,x):
    x = self.pool1(self.relu(self.conv1(x)))
    x = self.pool2(self.relu(self.conv2(x)))
    x = x.view(x.size(0),-1)  # flatten
    
    # --- CRITICAL ---
    # This forward pass must match how the model was TRAINED.
    # The 'model_training.py' script uses ReLU on fc1 and fc2.
    # If you trained *without* these, comment them out and uncomment the lines below.
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    
    # Uncomment these ONLY if your training script had no activations on fc1/fc2
    # x = self.fc1(x) 
    # x = self.fc2(x) 
    
    x = self.fc3(x) # Raw logits output
    return x

# --- Utility Functions ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the trained PyTorch model from the specified path.
    """
    try
        # Use the *exact* class name from your training script
        model = CNNModel() 
        # Load the saved weights.
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
        st.error("This usually means the defined model architecture (class CNNModel) in app.py")
        st.error("does not exactly match the architecture used to save the weights.")
        st.error(f"Details: {e}")
        return None

def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image to be compatible with the MNIST model.
    """
    try:
        # Open the image
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to grayscale

        # --- Handle Color Inversion ---
        img_array = np.array(img)
        if np.mean(img_array) > 128:
            img = ImageOps.invert(img)

        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # --- CRITICAL ---
        # Define transformations to match your training script
        # Your 'model_training.py' only uses ToTensor() and not Normalize()
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)) # <-- DO NOT USE: Your model was not trained with this.
        ])
        
        # Apply transform and add batch dimension (1, 1, 28, 28)
        tensor = transform(img).unsqueeze(0)
        
        # Convert tensor back to PIL Image for display
        # We don't need to un-normalize since we never normalized
        display_img = transforms.ToPILImage()(tensor.squeeze(0))

        return tensor, display_img
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# --- Streamlit App ---

st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")
st.title("🧠 MNIST Digit Recognizer")

# Load the model
model = load_model('mnist_cnn_model.pth')

if model:
    st.success("Model 'mnist_cnn_model.pth' loaded successfully!")
    st.info("""
        **Welcome!** This app uses the `CNNModel` architecture.
        
        1.  **Upload** an image file.
        2.  Or, use your **Camera** to take a picture of a digit.
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
        
        tensor, display_img = preprocess_image(image_bytes)
        
        if tensor is not None:
            with col2:
                st.image(display_img, caption="Preprocessed (28x28)", use_column_width=True)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor) # Model returns raw logits
                # Apply softmax to logits to get probabilities
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

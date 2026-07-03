import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

MODEL_PATH = "simple_cnn_model.pth"
IMAGE_SIZE = (250, 250)

st.set_page_config(
    page_title="Tree vs Plant Classifier",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
}
.title-box {
    background: linear-gradient(90deg, #1b5e20, #43a047);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}
.result-box {
    background-color: #dff5e1;
    border: 2px solid #2e7d32;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    color: #1b5e20;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🌳 Tree vs Plant Image Classifier 🌿</h1>
    <p>CNN Deep Learning Model using PyTorch + Streamlit</p>
</div>
""", unsafe_allow_html=True)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    class_names = checkpoint["classes"]

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names

def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def predict(image, model, class_names):
    transform = get_transform()

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return class_names[predicted.item()], confidence.item()

st.write("")
st.write("Upload image and model will predict whether it is **Tree** or **Plant**.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying image..."):
        model, class_names = load_model()
        predicted_class, confidence = predict(image, model, class_names)

    st.markdown(f"""
    <div class="result-box">
        Prediction: {predicted_class.upper()} <br>
        Confidence: {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🛠 Tech Stack")
st.write("Python | PyTorch | CNN | Streamlit | Computer Vision")
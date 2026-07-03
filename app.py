import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import qrcode
import io
from datetime import datetime

MODEL_PATH = "simple_cnn_model.pth"
IMAGE_SIZE = (250, 250)

st.set_page_config(
    page_title="Plant Health Assistant",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
}
.title-box {
    background: linear-gradient(90deg, #1b5e20, #43a047);
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}
.result-box {
    background-color: #dff5e1;
    border: 2px solid #2e7d32;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    color: #1b5e20;
    font-weight: bold;
}
.info-box {
    background-color: #ffffff;
    border-left: 6px solid #2e7d32;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.12);
}
.card {
    background:#f1f8e9;
    padding:20px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 4px 10px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🌿 Plant Health Assistant</h1>
    <p>AI Powered Tree vs Plant Classification + Health Report System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Project Info")
st.sidebar.write("Model: CNN")
st.sidebar.write("Classes: Tree, Plant")
st.sidebar.write("Dataset Images: 111")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Deployment: Streamlit Cloud")
st.sidebar.write("Features: PDF, QR, Health Score, Treatment")

if "history" not in st.session_state:
    st.session_state.history = []


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

    return class_names[predicted.item()], confidence.item(), probabilities[0]


def get_health_info(predicted_class, confidence):
    if predicted_class.lower() == "plant":
        if confidence >= 0.90:
            severity = "Low Risk"
            medicine = "No chemical medicine required"
            treatment = [
                "Provide proper sunlight.",
                "Water regularly but avoid overwatering.",
                "Check leaves weekly for spots or yellowing.",
                "Use organic compost for better growth."
            ]
        elif confidence >= 0.70:
            severity = "Medium Risk"
            medicine = "Organic neem oil spray"
            treatment = [
                "Inspect leaves carefully.",
                "Remove damaged or infected leaves.",
                "Spray diluted neem oil once a week.",
                "Keep plant in ventilated sunlight."
            ]
        else:
            severity = "High Risk"
            medicine = "Consult a plant specialist"
            treatment = [
                "Upload a clearer leaf image.",
                "Check for yellowing, holes, or black spots.",
                "Separate the plant from others if infection is visible.",
                "Consult an agriculture expert."
            ]
    else:
        severity = "Not a leaf disease case"
        medicine = "No medicine required"
        treatment = [
            "This image is predicted as Tree.",
            "For disease detection, upload a clear leaf image.",
            "Use close-up leaf images for better analysis."
        ]

    return severity, medicine, treatment


def create_pdf(predicted_class, confidence, severity, medicine, treatment, health_score):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Plant Health Prediction Report", ln=True, align="C")

    pdf.ln(8)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Prediction: {predicted_class}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence*100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Health Score: {health_score}/100", ln=True)
    pdf.cell(0, 10, f"Severity Level: {severity}", ln=True)
    pdf.cell(0, 10, f"Medicine Recommendation: {medicine}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Treatment Steps:", ln=True)

    pdf.set_font("Arial", "", 12)
    for step in treatment:
        pdf.multi_cell(0, 8, f"- {step}")

    pdf.ln(5)
    pdf.multi_cell(
        0,
        8,
        "Note: This app currently uses a Tree vs Plant model. "
        "For real disease prediction, train a plant disease dataset model."
    )

    return pdf.output(dest="S").encode("latin-1")


def create_qr(text):
    qr = qrcode.make(text)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


st.write("")
st.write("Upload leaf/plant image or use camera. The system will predict and generate health guidance.")

option = st.radio("Choose input method:", ["Upload Image", "Use Camera"], horizontal=True)

uploaded_file = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        model, class_names = load_model()
        predicted_class, confidence, probabilities = predict(image, model, class_names)

    severity, medicine, treatment = get_health_info(predicted_class, confidence)
    health_score = int(confidence * 100)

    with col_result:
        st.markdown(f"""
        <div class="card">
            <h2>Prediction</h2>
            <h1>{predicted_class.upper()}</h1>
            <h3>{confidence*100:.2f}% Confidence</h3>
            <h3>Health Score: {health_score}/100</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if severity == "Low Risk":
            st.success("🟢 Low Risk")
        elif severity == "Medium Risk":
            st.warning("🟡 Medium Risk")
        elif severity == "High Risk":
            st.error("🔴 High Risk")
        else:
            st.info("ℹ️ Not a leaf disease case")

    if confidence > 0.95:
        st.balloons()

    st.markdown(f"""
    <div class="result-box">
        Prediction: {predicted_class.upper()} <br>
        Confidence: {confidence*100:.2f}% <br>
        Severity: {severity}
    </div>
    """, unsafe_allow_html=True)

    st.write("### 🌿 Plant Health Score")
    st.progress(health_score / 100)
    st.write(f"Health Score: **{health_score}/100**")

    st.write("### 📊 Prediction Probabilities")

    prob_values = [p.item() for p in probabilities]

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#66BB6A", "#1B5E20"]

    ax.pie(
        prob_values,
        labels=class_names,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[:len(class_names)],
        wedgeprops=dict(width=0.42)
    )

    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.set_title("Prediction Confidence Donut Chart")
    st.pyplot(fig)

    st.write("### 📌 Class Wise Probability")
    for i, cls in enumerate(class_names):
        st.write(f"**{cls}:** {probabilities[i].item()*100:.2f}%")
        st.progress(float(probabilities[i].item()))

    st.markdown("### 🌱 Plant Health Guidance")
    st.markdown(f"""
    <div class="info-box">
        <b>Severity Level:</b> {severity}<br>
        <b>Medicine Recommendation:</b> {medicine}
    </div>
    """, unsafe_allow_html=True)

    st.write("### 🧪 Treatment Steps")
    for step in treatment:
        st.write(f"✅ {step}")

    st.session_state.history.append(
        f"{predicted_class} - {confidence*100:.2f}% - {severity}"
    )

    st.write("### 🕘 Prediction History")
    for item in st.session_state.history[-5:]:
        st.write(f"🌿 {item}")

    report_text = f"""
Plant Health Assistant Report

Prediction: {predicted_class}
Confidence: {confidence*100:.2f}%
Health Score: {health_score}/100
Severity: {severity}
Medicine: {medicine}
"""

    qr_buffer = create_qr(report_text)
    st.write("### 📱 QR Code Report")
    st.image(qr_buffer, width=180)

    pdf_data = create_pdf(
        predicted_class,
        confidence,
        severity,
        medicine,
        treatment,
        health_score
    )

    col_pdf, col_qr = st.columns(2)

    with col_pdf:
        st.download_button(
            "📄 Download PDF Report",
            pdf_data,
            file_name="plant_health_report.pdf",
            mime="application/pdf"
        )

    with col_qr:
        st.download_button(
            "📱 Download QR Code",
            qr_buffer.getvalue(),
            file_name="plant_report_qr.png",
            mime="image/png"
        )

st.markdown("---")
st.markdown("### 🛠 Tech Stack")
st.write("Python | PyTorch | CNN | Streamlit | Computer Vision | PDF | QR Code | Donut Chart")
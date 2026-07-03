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
    <p>AI Powered Tree vs Plant Classification + Plant Report System</p>
</div>
""", unsafe_allow_html=True)

PLANT_DATABASE = {
    "Unknown": {
        "local_name": "Unknown",
        "scientific_name": "Unknown",
        "summary": "Plant species not selected. Basic plant health guidance is provided.",
        "advantages": ["General plant care guidance available"],
        "disadvantages": ["Exact species-level advice unavailable"],
        "uses": ["Basic gardening and plant care"]
    },
    "Neem": {
        "local_name": "Neem / नीम",
        "scientific_name": "Azadirachta indica",
        "summary": "Neem is a medicinal plant known for antibacterial, antifungal and natural pesticide properties.",
        "advantages": ["Natural pesticide", "Medicinal leaves", "Improves soil health"],
        "disadvantages": ["Very bitter taste", "Excess use may irritate skin", "Needs warm climate"],
        "uses": ["Skin care", "Organic farming", "Herbal medicine"]
    },
    "Tulsi": {
        "local_name": "Tulsi / तुलसी",
        "scientific_name": "Ocimum tenuiflorum",
        "summary": "Tulsi is a sacred medicinal plant commonly used for immunity, cough and respiratory health.",
        "advantages": ["Boosts immunity", "Easy to grow", "Useful in herbal tea"],
        "disadvantages": ["Sensitive to cold", "Needs regular watering", "Can dry in harsh sunlight"],
        "uses": ["Tea", "Ayurvedic medicine", "Home remedy"]
    },
    "Mango": {
        "local_name": "Aam / आम",
        "scientific_name": "Mangifera indica",
        "summary": "Mango is a fruit tree widely grown in India for fruits, shade and long-term plantation.",
        "advantages": ["Fruit production", "Provides shade", "Long lifespan"],
        "disadvantages": ["Needs large space", "Seasonal diseases possible", "Requires regular care"],
        "uses": ["Fruits", "Pickles", "Wood and shade"]
    },
    "Aloe Vera": {
        "local_name": "Ghritkumari / एलोवेरा",
        "scientific_name": "Aloe barbadensis miller",
        "summary": "Aloe Vera is a succulent medicinal plant used for skin care and minor burns.",
        "advantages": ["Useful for skin", "Low maintenance", "Needs less water"],
        "disadvantages": ["Overwatering can damage roots", "Not frost tolerant"],
        "uses": ["Skin gel", "Cosmetics", "Home remedies"]
    }
}

st.sidebar.title("📊 Project Info")
st.sidebar.write("Model: CNN")
st.sidebar.write("Classes: Tree, Plant")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Features: PDF, QR, Health Score, Plant Details")

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


def create_pdf(predicted_class, confidence, severity, medicine, treatment, health_score, plant_name, plant_details):
    def safe_text(text):
        return str(text).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, safe_text("Plant Health Prediction Report"), ln=True, align="C")

    pdf.ln(8)
    pdf.set_font("Arial", "", 12)

    lines = [
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        f"Prediction: {predicted_class}",
        f"Confidence: {confidence*100:.2f}%",
        f"Health Score: {health_score}/100",
        f"Severity Level: {severity}",
        f"Medicine Recommendation: {medicine}",
        "",
        "Plant Details",
        f"Plant Name: {plant_name}",
        f"Local Name: {plant_details['local_name']}",
        f"Scientific Name: {plant_details['scientific_name']}",
        f"Short Summary: {plant_details['summary']}",
        "",
        "Advantages:"
    ]

    for line in lines:
        pdf.multi_cell(0, 8, safe_text(line))

    for item in plant_details["advantages"]:
        pdf.multi_cell(0, 8, safe_text("- " + item))

    pdf.multi_cell(0, 8, safe_text(""))
    pdf.multi_cell(0, 8, safe_text("Disadvantages:"))
    for item in plant_details["disadvantages"]:
        pdf.multi_cell(0, 8, safe_text("- " + item))

    pdf.multi_cell(0, 8, safe_text(""))
    pdf.multi_cell(0, 8, safe_text("Uses:"))
    for item in plant_details["uses"]:
        pdf.multi_cell(0, 8, safe_text("- " + item))

    pdf.multi_cell(0, 8, safe_text(""))
    pdf.multi_cell(0, 8, safe_text("Treatment Steps:"))
    for step in treatment:
        clean_step = step.replace("✅", "").replace("🌿", "").replace("📄", "")
        pdf.multi_cell(0, 8, safe_text("- " + clean_step))

    pdf.multi_cell(
        0,
        8,
        safe_text("Note: This app currently uses a Tree vs Plant model. For accurate disease and species prediction, train a plant disease/species dataset model.")
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

    selected_plant = st.selectbox(
        "Select Plant Name for Detailed Report",
        list(PLANT_DATABASE.keys())
    )
    plant_details = PLANT_DATABASE[selected_plant]

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

        if severity == "Low Risk":
            st.success("🟢 Low Risk")
        elif severity == "Medium Risk":
            st.warning("🟡 Medium Risk")
        elif severity == "High Risk":
            st.error("🔴 High Risk")
        else:
            st.info("ℹ️ Not a leaf disease case")

    st.markdown(f"""
    <div class="result-box">
        Prediction: {predicted_class.upper()} <br>
        Confidence: {confidence*100:.2f}% <br>
        Severity: {severity}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌿 Plant Name & Short Details")
    st.markdown(f"""
    <div class="info-box">
        <b>Selected Plant:</b> {selected_plant}<br>
        <b>Local Name:</b> {plant_details['local_name']}<br>
        <b>Scientific Name:</b> <i>{plant_details['scientific_name']}</i><br>
        <b>Short Summary:</b> {plant_details['summary']}
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.write("### ✅ Advantages")
        for item in plant_details["advantages"]:
            st.write(f"- {item}")

    with col_b:
        st.write("### ⚠️ Disadvantages")
        for item in plant_details["disadvantages"]:
            st.write(f"- {item}")

    with col_c:
        st.write("### 🌱 Uses")
        for item in plant_details["uses"]:
            st.write(f"- {item}")

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
        f"{predicted_class} - {confidence*100:.2f}% - {severity} - {selected_plant}"
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
Plant Name: {selected_plant}
Local Name: {plant_details['local_name']}
Scientific Name: {plant_details['scientific_name']}
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
        health_score,
        selected_plant,
        plant_details
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
st.write("Python | PyTorch | CNN | Streamlit | Computer Vision | PDF | QR Code | Plant Info")
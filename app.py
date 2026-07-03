import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import qrcode
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MODEL_PATH = "simple_cnn_model.pth"
IMAGE_SIZE = (250, 250)

st.set_page_config(
    page_title="Plant Health Assistant",
    page_icon="🌿",
    layout="wide"
)

# ================= LOGIN SYSTEM =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""

def login_page():
    st.markdown("""
    <style>
    .login-box {
        background:white;
        padding:35px;
        border-radius:25px;
        box-shadow:0px 6px 25px rgba(0,0,0,0.15);
        max-width:500px;
        margin:auto;
        text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-box">
        <h1>🌿 Plant Health Assistant</h1>
        <p>Login to continue</p>
    </div>
    """, unsafe_allow_html=True)

    login_type = st.radio(
        "Choose Login Method",
        ["Email Login", "Mobile OTP", "Google Login"],
        horizontal=True
    )

    if login_type == "Email Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if email and password:
                st.session_state.logged_in = True
                st.session_state.user = email
                st.rerun()
            else:
                st.error("Enter email and password")

    elif login_type == "Mobile OTP":
        mobile = st.text_input("Mobile Number")
        otp = st.text_input("OTP")

        if st.button("Verify OTP"):
            if mobile and otp:
                st.session_state.logged_in = True
                st.session_state.user = mobile
                st.rerun()
            else:
                st.error("Enter mobile number and OTP")

    else:
        st.info("Real Google Login needs Firebase setup. This button is demo.")
        if st.button("Continue with Google Demo"):
            st.session_state.logged_in = True
            st.session_state.user = "google_user@gmail.com"
            st.rerun()

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ================= CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef8ef, #ffffff);
}

.title-box {
    background: linear-gradient(90deg, #1b5e20, #43a047);
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    color: white;
    box-shadow: 0px 6px 22px rgba(0,0,0,0.25);
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(0,0,0,0.10);
    border: 1px solid #dcedc8;
}

.info-box {
    background-color: #ffffff;
    border-left: 7px solid #2e7d32;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.12);
}

.section-card {
    background:white;
    padding:22px;
    border-radius:20px;
    box-shadow:0 4px 14px rgba(0,0,0,0.08);
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🌿 Plant Health Assistant</h1>
    <p>AI Powered Tree vs Plant Classification + Plant Report System</p>
</div>
""", unsafe_allow_html=True)

# ================= PLANT DATABASE =================
PLANT_DATABASE = {
    "Unknown": {
        "local_name": "Not Selected",
        "scientific_name": "Not Available",
        "summary": "Please select a plant name to generate a detailed report.",
        "advantages": ["General plant guidance available"],
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
        "summary": "Tulsi is a sacred medicinal plant used for immunity and respiratory health.",
        "advantages": ["Boosts immunity", "Easy to grow", "Useful in herbal tea"],
        "disadvantages": ["Sensitive to cold", "Needs regular watering", "Can dry in harsh sunlight"],
        "uses": ["Tea", "Ayurvedic medicine", "Home remedy"]
    },
    "Mango": {
        "local_name": "Aam / आम",
        "scientific_name": "Mangifera indica",
        "summary": "Mango is a fruit tree grown for fruits, shade and long-term plantation.",
        "advantages": ["Fruit production", "Provides shade", "Long lifespan"],
        "disadvantages": ["Needs large space", "Seasonal diseases possible", "Requires regular care"],
        "uses": ["Fruits", "Pickles", "Wood and shade"]
    },
    "Aloe Vera": {
        "local_name": "Ghritkumari / एलोवेरा",
        "scientific_name": "Aloe barbadensis miller",
        "summary": "Aloe Vera is a medicinal succulent plant used for skin care and minor burns.",
        "advantages": ["Useful for skin", "Low maintenance", "Needs less water"],
        "disadvantages": ["Overwatering can damage roots", "Not frost tolerant"],
        "uses": ["Skin gel", "Cosmetics", "Home remedies"]
    }
}

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= MODEL =================
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
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
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

# ================= HEALTH INFO =================
def get_health_info(predicted_class, confidence):
    if predicted_class.lower() == "plant":
        if confidence >= 0.90:
            return (
                "Low Risk",
                "No chemical medicine required",
                [
                    "Provide proper sunlight.",
                    "Water regularly but avoid overwatering.",
                    "Check leaves weekly for spots or yellowing.",
                    "Use organic compost for better growth."
                ]
            )
        elif confidence >= 0.70:
            return (
                "Medium Risk",
                "Organic neem oil spray",
                [
                    "Inspect leaves carefully.",
                    "Remove damaged or infected leaves.",
                    "Spray diluted neem oil once a week.",
                    "Keep plant in ventilated sunlight."
                ]
            )
        else:
            return (
                "High Risk",
                "Consult a plant specialist",
                [
                    "Upload a clearer leaf image.",
                    "Check for yellowing, holes, or black spots.",
                    "Separate the plant from others if infection is visible.",
                    "Consult an agriculture expert."
                ]
            )
    else:
        return (
            "Not a leaf disease case",
            "No medicine required",
            [
                "This image is predicted as Tree.",
                "For disease detection, upload a clear leaf image.",
                "Use close-up leaf images for better analysis."
            ]
        )

# ================= PDF =================
def create_pdf(predicted_class, confidence, severity, medicine, treatment, health_score, plant_name, plant_details):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def write_line(text, font="Helvetica", size=11, gap=18):
        nonlocal y
        c.setFont(font, size)

        text = str(text)
        text = text.replace("✅", "").replace("🌿", "").replace("📄", "").replace("📱", "")

        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont(font, size)

        c.drawString(50, y, text[:95])
        y -= gap

    def section(title):
        write_line(title, "Helvetica-Bold", 13, 22)

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, "Plant Health Prediction Report")
    y -= 35

    section("Prediction Details")
    write_line(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    write_line(f"Prediction: {predicted_class}")
    write_line(f"Confidence: {confidence*100:.2f}%")
    write_line(f"Health Score: {health_score}/100")
    write_line(f"Severity Level: {severity}")
    write_line(f"Medicine Recommendation: {medicine}")

    y -= 8
    section("Plant Details")
    write_line(f"Plant Name: {plant_name}")
    write_line(f"Local Name: {plant_details['local_name']}")
    write_line(f"Scientific Name: {plant_details['scientific_name']}")
    write_line(f"Summary: {plant_details['summary']}")

    y -= 8
    section("Advantages")
    for item in plant_details["advantages"]:
        write_line(f"- {item}")

    section("Disadvantages")
    for item in plant_details["disadvantages"]:
        write_line(f"- {item}")

    section("Uses")
    for item in plant_details["uses"]:
        write_line(f"- {item}")

    section("Treatment Steps")
    for step in treatment:
        write_line(f"- {step}")

    y -= 8
    write_line(
        "Note: Current model predicts Tree vs Plant only. Species/disease model needs separate dataset.",
        size=9
    )

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def create_qr(text):
    qr = qrcode.make(text)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# ================= SIDEBAR =================
st.sidebar.success(f"👋 Welcome {st.session_state.user}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Predictions", len(st.session_state.history))
st.sidebar.metric("Model", "CNN")
st.sidebar.metric("Classes", "2")
st.sidebar.markdown("---")
st.sidebar.write("Features:")
st.sidebar.write("✅ Prediction")
st.sidebar.write("✅ PDF Report")
st.sidebar.write("✅ QR Code")
st.sidebar.write("✅ Plant Info")
st.sidebar.write("✅ Login UI")

# ================= MAIN APP =================
st.write("")
st.write("Upload leaf/plant image or use camera. The system will predict and generate health guidance.")

option = st.radio(
    "Choose input method:",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

uploaded_file = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    selected_plant = st.selectbox(
        "Select Plant Name for Detailed Report",
        list(PLANT_DATABASE.keys())
    )

    plant_details = PLANT_DATABASE[selected_plant]

    with st.spinner("Analyzing image..."):
        model, class_names = load_model()
        predicted_class, confidence, probabilities = predict(
            image,
            model,
            class_names
        )

    severity, medicine, treatment = get_health_info(
        predicted_class,
        confidence
    )

    health_score = int(confidence * 100)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Prediction", predicted_class)

    with m2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with m3:
        st.metric("Health Score", f"{health_score}/100")

    with m4:
        st.metric("Selected Plant", selected_plant)

    # Image and chart
    left, right = st.columns([1.25, 1])

    with left:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with right:
        st.subheader("📊 AI Confidence Analysis")

        prob_values = [p.item() for p in probabilities]

        fig, ax = plt.subplots(figsize=(3.2, 3.2))

        colors = ["#66BB6A", "#1B5E20", "#81C784", "#AED581"]

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

        ax.set_title("Confidence", fontsize=10)

        st.pyplot(fig)

        for i, cls in enumerate(class_names):
            p = probabilities[i].item()
            st.write(f"**{cls}: {p*100:.2f}%**")
            st.progress(float(p))

    # Status
    if severity == "Low Risk":
        st.success("🟢 Low Risk")
    elif severity == "Medium Risk":
        st.warning("🟡 Medium Risk")
    elif severity == "High Risk":
        st.error("🔴 High Risk")
    else:
        st.info("ℹ️ Not a leaf disease case")

    # Plant details
    st.markdown("### 🌿 Plant Name & Short Details")
    st.markdown(f"""
    <div class="info-box">
        <b>Selected Plant:</b> {selected_plant}<br>
        <b>Local Name:</b> {plant_details['local_name']}<br>
        <b>Scientific Name:</b> <i>{plant_details['scientific_name']}</i><br>
        <b>Short Summary:</b> {plant_details['summary']}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("✅ Advantages")
        for item in plant_details["advantages"]:
            st.write("•", item)

    with c2:
        st.warning("⚠️ Disadvantages")
        for item in plant_details["disadvantages"]:
            st.write("•", item)

    with c3:
        st.info("🌱 Uses")
        for item in plant_details["uses"]:
            st.write("•", item)

    # Treatment
    st.subheader("🩺 Treatment Recommendation")
    st.markdown(f"""
    <div class="info-box">
        <b>Severity Level:</b> {severity}<br>
        <b>Medicine Recommendation:</b> {medicine}
    </div>
    """, unsafe_allow_html=True)

    for i, step in enumerate(treatment, start=1):
        st.write(f"{i}. {step}")

    # History
    st.session_state.history.append(
        f"{predicted_class} | {confidence*100:.2f}% | {severity} | {selected_plant}"
    )

    st.subheader("🕘 Prediction History")
    st.dataframe(
        st.session_state.history[-10:],
        use_container_width=True
    )

    # QR + PDF
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

    st.subheader("📄 Reports & Downloads")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "📄 Download PDF Report",
            pdf_data,
            file_name="plant_health_report.pdf",
            mime="application/pdf"
        )

    with d2:
        st.download_button(
            "📱 Download QR Code",
            qr_buffer.getvalue(),
            file_name="plant_report_qr.png",
            mime="image/png"
        )

    with d3:
        st.image(qr_buffer, width=140)

st.markdown("---")
st.caption(
    "🌿 Plant Health Assistant • Powered by PyTorch • Streamlit • Computer Vision • AI"
)
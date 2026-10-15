"""Streamlit UI for the Multi-Class Body Type Classifier."""
from pathlib import Path
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from model import create_model

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "resnet18_cars.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(checkpoint_path: str, modified_time: float):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    
    model, _ = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Match the validation transform used in training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, idx_to_class, transform

st.set_page_config(page_title="Vehicle Body Type Classifier", page_icon="🚘")
st.title("🚘 Vehicle Body Type Classifier")
st.write("Upload an image of a car to predict its general body type (e.g., SUV, Sedan, Coupe, etc.).")

if not CHECKPOINT_PATH.exists():
    st.warning("The trained model checkpoint has not been found yet.")
    st.stop()

uploaded_file = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as a valid image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model, idx_to_class, transform = load_model(
            str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime
        )
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get Top 3 Predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        st.subheader("Top Prediction")
        best_class = idx_to_class[top_indices[0].item()].replace("_", " ")
        st.success(f"**{best_class}**")
        st.metric("Confidence", f"{top_probs[0].item() * 100:.2f}%")

        st.caption("Top 3 Possibilities")
        for i in range(3):
            class_name = idx_to_class[top_indices[i].item()].replace("_", " ")
            prob = top_probs[i].item()
            st.write(f"{i+1}. **{class_name}** ({prob * 100:.2f}%)")
            st.progress(float(prob))

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
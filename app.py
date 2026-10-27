"""Streamlit UI with 'No Car' Prediction Logic."""
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
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    
    model, _ = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, idx_to_class, transform

st.title("🚘 Vehicle Classifier")

if not CHECKPOINT_PATH.exists():
    st.error("Model not found. Please train the model first.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        
        model, idx_to_class, transform = load_model()
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, 3)
        confidence = top_probs[0].item() * 100
        best_class = idx_to_class[top_indices[0].item()].replace("_", " ")

        st.subheader("Prediction")
        
        # Threshold logic for "No Car"
        if confidence < 60.0:
            st.error("🚫 **No Car Present** (Image not recognized as a vehicle)")
            st.caption(f"Highest similarity was {best_class} at {confidence:.2f}%")
        elif best_class.lower() == "no car":
            st.error("🚫 **No Car Present**")
        else:
            st.success(f"**{best_class}** ({confidence:.2f}% Match)")
            
            st.write("---")
            st.caption("Top 3 Possibilities:")
            for i in range(3):
                cls_name = idx_to_class[top_indices[i].item()].replace("_", " ")
                prob = top_probs[i].item() * 100
                st.write(f"{i+1}. {cls_name} ({prob:.2f}%)")
                st.progress(prob / 100.0)

    except Exception as e:
        st.error(f"Error processing image: {e}")
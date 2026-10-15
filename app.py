"""Streamlit UI for the Supercar vs SUV classifier."""
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import CarCNN

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "car_cnn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(checkpoint_path: str, modified_time: float):
    """Cache model loading; modified_time invalidates cache after retraining."""
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model = CarCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_to_idx = checkpoint.get(
        "class_to_idx", {"supercars": 0, "suv_cars": 1}
    )
    idx_to_class = {int(index): name for name, index in class_to_idx.items()}
    image_size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("normalization_mean", [0.5, 0.5, 0.5])
    std = checkpoint.get("normalization_std", [0.5, 0.5, 0.5])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, idx_to_class, transform


st.set_page_config(
    page_title="Supercar vs SUV Classifier",
    page_icon="🚘",
    layout="centered",
)

st.title("🚘 Supercar vs SUV Classifier")
st.write(
    "Upload a car image and the trained CNN will predict whether it is "
    "a **supercar** or an **SUV**."
)

if not CHECKPOINT_PATH.exists():
    st.warning("The trained model checkpoint has not been found yet.")
    st.code("python prepare_data.py\npython train.py\nstreamlit run app.py")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a car image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Use a clear image containing one main vehicle.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as a valid image. Try another JPG or PNG.")
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

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = idx_to_class[predicted_index]
        confidence = float(probabilities[predicted_index].item()) * 100.0

        display_names = {
            "supercars": "Supercar",
            "suv_cars": "SUV",
        }
        st.subheader("Prediction")
        st.success(display_names.get(predicted_class, predicted_class))
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption("Class probabilities")
        for index, probability in enumerate(probabilities.tolist()):
            class_name = idx_to_class.get(index, str(index))
            label = display_names.get(class_name, class_name)
            st.write(f"**{label}:** {probability * 100:.2f}%")
            st.progress(float(probability))

        st.caption(
            "Confidence is the model's estimated probability, not a guarantee "
            "that the prediction is correct. Results depend on training data quality."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("If you recently changed model.py, retrain with `python train.py`.")
else:
    st.info("Upload a JPG or PNG image to get a prediction.")

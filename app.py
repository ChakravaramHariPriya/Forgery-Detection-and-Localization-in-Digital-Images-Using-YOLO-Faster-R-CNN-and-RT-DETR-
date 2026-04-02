import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
import os

# ----------------------------
# Load Models
# ----------------------------

@st.cache_resource
def load_yolo_model():
    model = YOLO("best1l.pt")
    return model

@st.cache_resource
def load_rtdetr_model():
    model = YOLO("rtdetr.pt")
    return model


@st.cache_resource
def load_fasterrcnn_model():
    num_classes = 2  # background + forgery
    model = fasterrcnn_resnet50_fpn(pretrained=False)

    # Replace the box predictor to match your trained model's output
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load your trained weights safely
    state_dict = torch.load("forgery_fasterrcnn.pth", map_location=torch.device("cpu"))

    # Load only matching keys (ignore mismatch in case)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

st.title("Forgery Detection Application")
st.write("Upload an image and select the model to detect forgery.")

# Model selection dropdown
model_choice = st.selectbox(
    "Select a detection model:",
    ("YOLO", "Faster R-CNN","RTDETR")
)

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    file_extension = uploaded_file.name.split('.')[-1]
    input_image_path = f"uploaded_image.{file_extension}"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Uploaded Image")
    uploaded_image = Image.open(input_image_path).convert("RGB")
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Prediction")

    if model_choice == "YOLO":
        model = load_yolo_model()
        results = model.predict(source=input_image_path, save=True, conf=0.1)
        prediction_dir = results[0].save_dir
        predicted_image_path = os.path.join(prediction_dir, os.path.basename(input_image_path))

    elif model_choice == "RTDETR":
        model = load_rtdetr_model()
        results = model.predict(source=input_image_path, save=True, conf=0.1)
        prediction_dir = results[0].save_dir
        predicted_image_path = os.path.join(prediction_dir, os.path.basename(input_image_path))
        
        
    elif model_choice == "Faster R-CNN":
        model = load_fasterrcnn_model()
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(uploaded_image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']

        threshold = 0.1
        selected_indices = [i for i, s in enumerate(scores) if s >= threshold]

        if len(selected_indices) == 0:
            predicted_image_path = input_image_path
        else:
            from torchvision.utils import draw_bounding_boxes
            import torchvision.transforms.functional as F

            image_tensor = F.pil_to_tensor(uploaded_image)

            selected_boxes = boxes[selected_indices]
            selected_labels = [f"{labels[i].item()}:{scores[i]:.2f}" for i in selected_indices]

            orig_w, orig_h = uploaded_image.size
            resized_h, resized_w = img_tensor.shape[2:]
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h
            selected_boxes = selected_boxes.clone()
            selected_boxes[:, [0, 2]] *= scale_x
            selected_boxes[:, [1, 3]] *= scale_y

            # Clamp boxes to image boundaries (prevent overflow)
            selected_boxes[:, 0] = torch.clamp(selected_boxes[:, 0], 0, orig_w)
            selected_boxes[:, 1] = torch.clamp(selected_boxes[:, 1], 0, orig_h)
            selected_boxes[:, 2] = torch.clamp(selected_boxes[:, 2], 0, orig_w)
            selected_boxes[:, 3] = torch.clamp(selected_boxes[:, 3], 0, orig_h)

            # --- Draw bounding boxes ---
            drawn = draw_bounding_boxes(
                image_tensor,
                boxes=selected_boxes,
                labels=selected_labels,
                colors=["red"] * len(selected_boxes),
                width=3
            )

            # Convert tensor to PIL and save
            predicted_image = F.to_pil_image(drawn)
            predicted_image_path = "fasterrcnn_prediction_fixed.jpg"
            predicted_image.save(predicted_image_path)

            # st.image(predicted_image, caption="Forgery Detection Result", use_column_width=True)
   
   
   
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(predicted_image_path, caption=f"{model_choice} Prediction", use_container_width=True)

    st.success(f"{model_choice} prediction complete!")


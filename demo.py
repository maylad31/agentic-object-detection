import gradio as gr
from ultralytics import FastSAM
from PIL import Image, ImageDraw
import torch
from sentence_transformers import SentenceTransformer

# Load CLIP model
model = SentenceTransformer("clip-ViT-B-16")
# Load FastSAM model
model_sam = FastSAM("FastSAM-x.pt")

def predict(image, object_name):
    image = Image.open(image)
    output = image.copy()
    draw = ImageDraw.Draw(output)
    
    # Run FastSAM segmentation
    results = model_sam(image, device="cpu", retina_masks=True, imgsz=832, conf=0.7, iou=0.6)
    
    # Check if FastSAM returned valid results
    if not results[0].boxes:
        return "No objects detected.", output
    
    for box in results[0].boxes.xyxy:
        x_min, y_min, x_max, y_max = map(int, box.tolist())  # Convert to integers
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Define labels for similarity comparison
        labels = [f"picture of a {object_name}"]
        # Encode an image:
        img_emb = model.encode(cropped_image)
        # Encode text descriptions
        text_emb = model.encode(labels)
        # Compute similarities
        similarity_scores = model.similarity(img_emb, text_emb)
        
        if similarity_scores[0].item() > 0.27:         
            draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
    
    return "Detection complete.", output

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="filepath"), gr.Textbox(label="Object Name")],
    outputs=[gr.Textbox(label="Status"), gr.Image(type="pil")],
    title="FastSAM Object Detection with CLIP",
    description="Upload an image, enter the object name, and get the output image with detected objects."
)

demo.launch()

import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

st.title("Image Caption Generator")

st.write(
    "Upload an image, and the model will generate a caption based on the image content.")   

image_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption"):
            caption = generate_caption(image)
            st.subheader("Generated Caption:")
            st.write(caption)
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    st.write("App is running...")


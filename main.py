import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import os

st.set_page_config(
        page_title="AIze Img: Make your images AI",
        page_icon="https://i.ibb.co/kcfZcsW/new-circle.png"
)

hide_St = """
	<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_link__qRIco {display:none;}
    </style>
"""

st.markdown(hide_St, unsafe_allow_html=True)

# Function to generate a caption for the given image file
def generate_caption(image_path):
    client_caption = Client("gokaygokay/SD3-Long-Captioner")
    result_caption = client_caption.predict(
        image=handle_file(image_path),
        api_name="/create_captions_rich"
    )
    return result_caption[0] if isinstance(result_caption, tuple) else result_caption

# Function to generate an AI image from the given caption
def generate_image_from_caption(caption):
    client_image = Client("stabilityai/stable-diffusion-3-medium")
    result_image = client_image.predict(
        prompt=caption,
        negative_prompt="",
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1024,
        guidance_scale=5,
        num_inference_steps=28,
        api_name="/infer"
    )
    return result_image[0] if isinstance(result_image, tuple) else result_image

# Streamlit app
st.title("Make your image AI")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Ensure the temp directory exists
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # Save uploaded image to a temporary file
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Generate caption
    with st.spinner('Generating caption...'):
        caption = generate_caption(temp_file_path)

    st.write("Generated Caption:", caption)

    # Generate AI image from caption
    with st.spinner('Generating AI image...'):
        generated_image_path = generate_image_from_caption(caption)

    # Display generated image
    generated_image = Image.open(generated_image_path)
    st.image(generated_image, caption='Generated Image.', use_column_width=True)
    st.page_link("https://twitter.com/not_gallium", label="🔵Follow me on 𝕏", icon="❎")
    st.page_link("https://youtube.com/@GAllium14", label="🔴Subscribe to my YT channel", icon="📺")
    st.page_link("https://github.com/GokulAnand14/AIze-img", label="🌟Open-Source on GitHub", icon="🔓")
    st.write("made with ❤ in INDIA🇮")

    # Clean up temporary file
    os.remove(temp_file_path)

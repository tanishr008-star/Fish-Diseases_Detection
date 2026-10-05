import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from utils import TinyCNN,data, fish_classes

st.set_page_config(
    page_title="Fish Disease Detection",
    page_icon="🐟",
    layout="wide"
)

model = TinyCNN(3,32, 7)
model.load_state_dict(torch.load("fish_model.pth",map_location = torch.device('cpu')))

col1 , col2  = st.columns(2)
predicted_data = None
predicted_probs = None
with col1:
    st.header("FISH DISEASES DETECTION")
    file_upload = st.file_uploader("Upload a fish image",type =['jpeg', 'jpg','png'])
    if file_upload is not None:
        img = Image.open(file_upload).convert("RGB")
        st.image(img)
        print(img)
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])
        transformed_img = transform(img).unsqueeze(dim=0)
        print(transformed_img.shape)
        model.eval()
        with torch.inference_mode():
            output = model(transformed_img)
        
        predicted_probs = torch.softmax(output,dim=1)
        predicted_label = torch.argmax(predicted_probs, dim= 1)
        predicted_data = data[predicted_label]

with col2: 
    if predicted_data:
        st.markdown(
            f"""
            <div style="
                background-color:#000;
                padding:15px;
                border-radius:10px;
                border:2px f5f5f5;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            ">
                <h4 style="color:#fff;">Predicted Disease: 
                    <span style="color:#fff;">{predicted_data['disease']}</span>
                </h4>
                <p style="color:#fff;">{predicted_data['description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        df  = pd.DataFrame(data)
        df.drop(columns=["description"], inplace= True)
        df["probs"] = predicted_probs.squeeze().tolist()
        print(df)
        st.markdown("""<h4 style="padding:15px;">Predicted probabilities graph:</h4>""", unsafe_allow_html=True)
        st.bar_chart(df.set_index('disease'))

        




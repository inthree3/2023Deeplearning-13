import streamlit as st
from PIL import Image, ImageOps
# from tensorflow.keras.applications.imagenet_utils import decode_predictions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

st.set_page_config(page_title="Look, Coco!", page_icon="🐈")

st.title('2023 DeepLearning Project-13')
file=st.file_uploader("반려견의 눈을 촬영해주세요.", type=['jpg', 'png'])
st.text("Hello Streamlit!")

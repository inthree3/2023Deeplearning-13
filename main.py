import streamlit as st
from PIL import Image, ImageOps
import torch
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.imagenet_utils import decode_predictions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

#model file 경로
#MODEL_PATH = 'detection_deepLearning.pt'
MODEL_PATH = './efficient_b0_binary.pt'

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.num_class = 2
        self.weights = models.EfficientNet_B0_Weights.DEFAULT
        self.network = models.efficientnet_b0(weights=self.weights)
        self.network.classifier = torch.nn.Sequential(
                nn.Dropout(0.2, inplace=True), 
                nn.Linear(in_features=1280,out_features=1280)
                )
#         self.fc1 = nn.Linear(1280, 1280) 1024 linear 한 번 더 진행
        self.fc = nn.Linear(1280, self.num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.network(input)
#         x = self.fc1(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

model =MyNet()
# 모델 로드
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=True)



def preprocess_image(image, target_size):
    image = image.convert("RGB")

    # 이미지 전처리 및 증강을 위한 변환 함수 생성
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지 size 변경 및 정규화
    image = transform(image)

    # 이미지 rescale
    # image = image * rescale_factor
    
    return image
# model = state_dict['model']
# model.eval()

# def diagnose(image):
#     datagen=ImageDataGenerator(rescale=1./255)
#     image=datagen.flow_from_directory(directory=image, target_size=(224, 224), batch_size=256, shuffle=False)
#     image = torch.utils.data.DataLoader(image, batch_size=256, shuffle=False)
#     output = model(image)
#     return output


st.set_page_config(page_title="Look, Coco!", page_icon="🐕")

st.title('2023 DeepLearning Project-13')
file=st.file_uploader("반려견의 눈을 촬영해주세요. (눈 외에 다른 것이 나오지 않도록 해주세요.)", type=['jpg', 'png', 'jpeg'])

if file is None:
    st.text("사진을 업로드해주세요.")
else:
    image=Image.open(file)
    st.image(image, use_column_width=True)
    image=preprocess_image(image, target_size=(224, 224))
    # st.write(f'{image.size()}')
    
    model.eval()
    with torch.no_grad():
        output=model(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
    
    predicted_label = predicted.item()

    if predicted_label == 0:
        output_result = "진단 결과: 정상입니다 💚"
    else:
        output_result = "진단 결과: 백내장 증세가 있습니다 ❤️‍🩹"
    st.write(f"{output_result}")

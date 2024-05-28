import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load your trained model
num_classes = 3  # Adjust based on your number of classes
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the last layer
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define class labels
class_labels = ['Healthy', 'Early Blight', 'Late Blight']  # Adjust based on your labels

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    st.title("Potato Disease Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Predict the class
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]

        st.write(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main()

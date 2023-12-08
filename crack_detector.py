import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from codes.model.deepcrack import DeepCrack
import sys
sys.path.insert(0, "..")

# Load the pre-trained model
model_path = 'C:/Users/stige/Downloads/DeepCrack-master/DeepCrack-master/codes/checkpoints/DeepCrack_CT260_FT1.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Instantiate the DeepCrack model
model = DeepCrack()  # Replace with the actual model class if it's different

# Load the state_dict
model.load_state_dict(checkpoint)

# Set the model to evaluation mode
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_crack(image_path, threshold=0.5):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output_tuple = model(input_tensor)

    # Assuming the first element of the tuple is the relevant tensor
    output = output_tuple[0]

    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(output)

    # Apply thresholding to identify crack regions
    binary_mask = probabilities > threshold

    # Check if any pixel is classified as a crack
    prediction = binary_mask.any().item()

    return prediction


def main(image_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_directory, filename)
            prediction = predict_crack(image_path)

            if prediction:
                print(f"{filename}: Crack detected")
            else:
                print(f"{filename}: No crack")

if __name__ == "__main__":
    # image_directory = 'C:/Users/stige/Downloads/DeepSegmentor-master/DeepSegmentor-master/datasets/DeepCrack/test_img'
    image_directory = 'C:/Users/stige/Downloads/crack_test_images/Negative'
    main(image_directory)

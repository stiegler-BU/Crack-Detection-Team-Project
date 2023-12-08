import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from codes.model.deepcrack import DeepCrack
import sys
sys.path.insert(0, "..")
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime

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
    # prediction = binary_mask.any().item()

    # Flatten the tensors for sklearn metrics
    y_true = binary_mask.view(-1).cpu().numpy()
    y_scores = probabilities.view(-1).cpu().numpy()

    return y_true, y_scores


def main(image_directory):
    all_true = []
    all_scores = []

    for filename in os.listdir(image_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_directory, filename)
            prediction = predict_crack(image_path)
            y_true, y_scores = predict_crack(image_path)
            all_true.extend(y_true)
            all_scores.extend(y_scores)

        if prediction:
            print(f"{filename}: Crack detected")
        else:
            print(f"{filename}: No crack")

    print("before computer PR curve")
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_true, all_scores)
    area = auc(recall, precision)

    print("before plot PR curve")
    # Plot precision-recall curve
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {area:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.title('Precision-Recall Curve')
    plt.show()

    print("before saving PR curve to directory")
    # Save the plot with a datetime stamp to the specified directory
    output_directory = 'C:/Users/stige/Downloads/DeepCrack-master/DeepCrack-master/codes/deepcrack_results'
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'precision_recall_curve_{timestamp}.png'
    filepath = os.path.join(output_directory, filename)
    print(f"Full path: {filepath}")
    plt.savefig(filepath)
    print(f"Precision-Recall curve saved as {filepath}")

if __name__ == "__main__":
    # image_directory = 'C:/Users/stige/Downloads/DeepSegmentor-master/DeepSegmentor-master/datasets/DeepCrack/test_img'
    image_directory = 'C:/Users/stige/Downloads/crack_test_images/Positive'
    main(image_directory)
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import numpy as np

# Load the trained model
model = mobilenet_v3_small(weights=None, num_classes=2)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the class names
class_names = ['Teammate A', 'Teammate B']  # Replace with your actual class names

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize([0.485, 0.456, 0.406],  # Mean values
                         [0.229, 0.224, 0.225])  # Std values
])

# Initialize the video stream
video_stream = cv2.VideoCapture(0)  # Use 0 for default webcam, or replace with video file path

if not video_stream.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error loading Haar Cascade XML file.")
    exit()

print("Press 'q' to quit.")
while True:
    # Read a frame from the video stream
    ret, frame = video_stream.read()
    if not ret:
        print("End of video stream or cannot read the frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face (bounding box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame
        face = frame[y:y + h, x:x + w]

        # Preprocess the face image
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (64, 64))
        face_tensor = transform(face_resized)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_idx = predicted.item()
            class_name = class_names[class_idx]
            confidence = confidence.item() * 100  # Convert to percentage

        # Prepare the label with class name and confidence
        label = f'{class_name}: {confidence:.1f}%'

        # Calculate position for the label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = y - 10 if y - 10 > 10 else y + 10

        # Draw background rectangle for the label
        cv2.rectangle(frame, (x, label_y - label_size[1] - 5), (x + label_size[0], label_y + 5), (0, 255, 0), cv2.FILLED)

        # Put the label text above the bounding box
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close windows
video_stream.release()
cv2.destroyAllWindows()

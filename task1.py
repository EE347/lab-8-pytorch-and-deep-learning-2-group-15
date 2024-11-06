import cv2
import os
import time
from picamera2 import Picamera2

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verify that the face cascade has been loaded
if face_cascade.empty():
    raise IOError("Failed to load face cascade file.")

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Directories to store captured images
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Create the necessary directories for each person
for dataset in [train_dir, test_dir]:
    for person_label in ["0", "1"]:
        os.makedirs(os.path.join(dataset, person_label), exist_ok=True)

# Parameters
face_size = (64, 64)
images_per_person = 60
train_images = 50
test_images = 10

def generate_filename(base_dir, person_label, image_number):
    """Generates a filename for the saved image."""
    return os.path.join(base_dir, person_label, f"face_{image_number}.jpg")

def capture_faces_for_person(person_label):
    """Captures face images for a given person label."""
    print(f"\nStarting image capture for person {person_label}")
    total_captured = 0
    train_count = 0
    test_count = 0
    while total_captured < images_per_person:
        # Capture frame from the camera
        frame = picam2.capture_array()

        # Convert the frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        if len(faces) > 0:
            # Assuming we want to capture the first detected face
            (x, y, w, h) = faces[0]

            # Add padding to the face rectangle
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame_bgr.shape[1] - x, w + 2 * padding)
            h = min(frame_bgr.shape[0] - y, h + 2 * padding)

            # Crop the face
            face_crop = frame_bgr[y:y+h, x:x+w]

            # Resize the face crop to 64x64
            face_resized = cv2.resize(face_crop, face_size)

            # Display the face for verification
            cv2.imshow('Captured Face', face_resized)
            print("Press 'y' to accept the face or 'n' to retake.")

            # Wait for user input
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                # Determine the save directory
                if train_count < train_images:
                    save_dir = train_dir
                    image_number = train_count
                    train_count += 1
                else:
                    save_dir = test_dir
                    image_number = test_count
                    test_count += 1

                # Generate filename and save the image
                filename = generate_filename(save_dir, person_label, image_number)
                cv2.imwrite(filename, face_resized)
                print(f"Saved image {filename}")
                total_captured += 1
            elif key == ord('n'):
                print("Image discarded. Retaking...")
            else:
                print("Invalid key pressed. Retaking image.")
        else:
            print("No face detected. Please adjust your position.")
            # Optional: Show the frame to help the user adjust
            cv2.imshow('Camera Preview', frame_bgr)
            cv2.waitKey(1)
            time.sleep(0.5)  # Brief pause to adjust

    print(f"Finished capturing images for person {person_label}")
    cv2.destroyAllWindows()

def main():
    print("Instructions:")
    print("1. Position person 0 in front of the camera.")
    print("2. The program will detect faces and ask for your confirmation.")
    print("3. Press 'y' to accept an image, 'n' to retake.")
    print("4. After 60 images, the program will proceed to person 1.")
    print("5. Repeat the process for person 1.")
    print("Press 'q' at any time to quit.")

    try:
        # Capture images for person 0
        input("\nPress Enter to start capturing images for person 0...")
        capture_faces_for_person("0")

        # Capture images for person 1
        input("\nPosition person 1 in front of the camera and press Enter to continue...")
        capture_faces_for_person("1")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        # Cleanup
        picam2.close()
        cv2.destroyAllWindows()
        print("Program terminated.")

if __name__ == "__main__":
    main()

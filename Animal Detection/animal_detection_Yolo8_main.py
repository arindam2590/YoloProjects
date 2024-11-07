import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def detect_animals(image_path):
    # Load the YOLOv8 model pretrained on the COCO dataset
    model = YOLO('yolov8s.pt')

    # List of COCO classes related to animals for filtering detections
    animal_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display in Matplotlib

    # Run YOLOv8 model on the image
    results = model(image_rgb)

    # Draw bounding boxes around detected animals
    for result in results:
        for box in result.boxes:
            # Get class ID and confidence score
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            # Only process if the detected class is an animal
            if class_name in animal_classes:
                # Extract bounding box coordinates and convert them to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Draw bounding box and label on the image
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detections
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


def main():
    # Path to the image for detection
    image_path = "Images/IMG7.jpg"

    # Run the animal detection function
    detect_animals(image_path)


if __name__ == "__main__":
    main()

import os
import cv2

# Define the data directory
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define your classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E',
           'Love You', 'All the Best']

dataset_size = 100  # Number of images per class

# Initialize webcam
cap = cv2.VideoCapture(0)

for class_name in classes:
    # Create directory for each class
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for class: {class_name}')

    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f'Class: {class_name} | Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_name = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(img_name, frame)
        counter += 1

    print(f'Finished collecting for class: {class_name}')

cap.release()
cv2.destroyAllWindows()

import os
import pickle
import cv2
import mediapipe as mp

# Initialize
data = []
labels = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Label mapping (optional but good practice)
label_map = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "A": 10, "B": 11, "C": 12, "D": 13, "E": 14,
    "Love You": 15, "All the Best": 16
}

# Your dataset path
data_dir = './dataset'

for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)

    if not os.path.isdir(folder_path):
        continue

    label = label_map.get(folder_name)
    if label is None:
        print(f"⚠️ Unknown label: {folder_name}")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append(landmarks)
                labels.append(label)

print(f"✅ Total samples collected: {len(data)}")

# Save the data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(" data.pickle created successfully!")

import onnxruntime as ort
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print("======================================")
print("Phase-2 Final Evaluation")
print("======================================")

# ---------------------------
# Load ONNX Model
# ---------------------------
session = ort.InferenceSession("defect_model (2).onnx")
input_name = session.get_inputs()[0].name
print("Model loaded successfully!")

# ---------------------------
# Exact Training Class Order
# ---------------------------
classes = [
    "Bridge",
    "Clean",
    "CMP Scratch",
    "Crack",
    "LER",
    "Malformed Via",
    "Open",
    "Others"
]

dataset_path = "hackathon_test_dataset"

y_true = []
y_pred = []

# ---------------------------
# Inference Loop
# ---------------------------
for class_name in os.listdir(dataset_path):

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):

        image_path = os.path.join(class_folder, filename)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (224, 224))

        # Same scaling as training
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        outputs = session.run(None, {input_name: img})
        pred_class = np.argmax(outputs[0])

        # True label mapping
        if class_name in classes:
            true_class = classes.index(class_name)
        else:
            true_class = classes.index("Others")

        y_true.append(true_class)
        y_pred.append(pred_class)

print("\nInference completed!")

# ---------------------------
# Metrics
# ---------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print("\n======================================")
print("Evaluation Results")
print("======================================")
print("Total Images:", len(y_true))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Macro Precision:", round(precision * 100, 2), "%")
print("Macro Recall:", round(recall * 100, 2), "%")

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

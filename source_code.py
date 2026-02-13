# ============================================================
# Semiconductor Defect Classification – Full Training Pipeline
# Transfer Learning using MobileNetV2
# ============================================================

# -------------------------------
# 1. Upload Dataset ZIP
# -------------------------------
from google.colab import files
uploaded = files.upload()   # Upload: IESA_Final.zip


# -------------------------------
# 2. Extract Dataset
# -------------------------------
import zipfile
import os

zip_path = "/content/IESA_Final.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content")

print("Dataset Extracted!")
print("Folders:", os.listdir("/content/IESA"))


# -------------------------------
# 3. Define Paths
# -------------------------------
train_dir = "/content/IESA/Train"
val_dir   = "/content/IESA/Validation"
test_dir  = "/content/IESA/Test"


# -------------------------------
# 4. Import Libraries
# -------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------
# 5. Data Generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


# -------------------------------
# 6. Load Pretrained Model
# -------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze Base Layers
for layer in base_model.layers:
    layer.trainable = False


# -------------------------------
# 7. Custom Classification Head
# -------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(
    train_data.num_classes,
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)


# -------------------------------
# 8. Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# -------------------------------
# 9. Train Model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)


# -------------------------------
# 10. Accuracy & Loss Graphs
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train","Validation"])

plt.show()


# -------------------------------
# 11. Evaluate Model
# -------------------------------
test_loss, test_acc = model.evaluate(test_data)

print("\nFinal Test Accuracy:", test_acc)


# -------------------------------
# 12. Confusion Matrix
# -------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="viridis", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()


# -------------------------------
# 13. Classification Report
# -------------------------------
from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names
))


# -------------------------------
# 14. Save Model
# -------------------------------
model.save("Semiconductor_Defect_Model.h5")

print("\nModel Saved!")


# -------------------------------
# 15. Download Model
# -------------------------------
files.download("Semiconductor_Defect_Model.h5")

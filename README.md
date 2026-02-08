# 🧠 Edge-AI Semiconductor Defect Classification

### IESA Hackathon Project — Wafer & Die Inspection using Transfer Learning

---

## 📌 Background

Semiconductor fabrication involves **hundreds of tightly controlled manufacturing steps**, where even microscopic defects can degrade performance or cause catastrophic device failure.

Modern semiconductor fabs generate **terabytes of inspection images daily** using:

* Optical Microscopes
* Scanning Electron Microscopes (SEM)
* Atomic Force Microscopes (AFM)
* Automated Defect Review Stations

Traditional centralized/manual inspection systems face major challenges:

* ⏱️ High latency in defect review
* 💰 Expensive compute infrastructure
* 🌐 Network & bandwidth bottlenecks
* 📉 Limited scalability for real-time throughput

To address these limitations, **Edge-AI enables on-device defect analysis**, reducing latency and supporting Industry 4.0 smart manufacturing ecosystems.

---

## 🧩 Problem Description

Semiconductor manufacturing produces massive volumes of wafer and die inspection images.

Undetected defects can:

* Reduce manufacturing yield
* Cause electrical failures
* Impact reliability & lifecycle

Centralized cloud inspection pipelines introduce:

* Data transfer delays
* Storage overhead
* Infrastructure cost

Hence, there is a need for a **portable, lightweight, real-time defect classification system** deployable directly on edge devices.

---

## 🎯 Objective

Design and implement an **Edge-AI powered semiconductor defect classification system** capable of:

* Detecting defects in wafer/die inspection images
* Classifying defects into predefined categories
* Achieving strong classification accuracy
* Operating under lightweight compute constraints
* Supporting real-time inspection workflows
* Enabling portability to Edge deployment frameworks (e.g., NXP eIQ)

---

## 🗂️ Dataset Structure

```
IESA/
│
├── Train/
├── Validation/
└── Test/
```

Each subset contains the following defect classes:

### 🔎 Defect Categories

* Bridge
* Crack
* CMP Scratch
* LER (Line Edge Roughness)
* Malformed Via
* Open
* Clean
* Others

---

## 🏗️ Model Architecture

This project uses **Transfer Learning** for efficient training on limited datasets.

**Backbone:** MobileNetV2 (ImageNet pretrained)

**Custom Head:**

* Global Average Pooling
* Dense Layer (ReLU)
* Dropout Regularization
* Softmax Output Layer

**Input Resolution:** 224 × 224 × 3

---

## ⚙️ Edge-AI Considerations

To align with edge deployment constraints:

* Lightweight CNN backbone (MobileNetV2)
* Reduced parameter count
* Efficient inference latency
* Compatibility with model compression workflows
* Portable to embedded AI runtimes

---

## 🚀 Training Configuration

| Parameter     | Value                       |
| ------------- | --------------------------- |
| Optimizer     | Adam                        |
| Loss Function | Categorical Crossentropy    |
| Epochs        | 15–25                       |
| Batch Size    | 32                          |
| Image Size    | 224×224                     |
| Augmentation  | Rotation, Shift, Zoom, Flip |

---

## 📊 Results

| Metric              | Performance |
| ------------------- | ----------- |
| Training Accuracy   | ~90–97%     |
| Validation Accuracy | ~78–82%     |
| Test Accuracy       | ~70–75%     |

> Accuracy is constrained by dataset size and class similarity.

---

## 📉 Observations

* Slight overfitting observed
* Class imbalance impacts minority defects
* Visually similar defects (e.g., scratches vs cracks) create confusion

---

## 🔮 Future Enhancements

* Larger industrial dataset integration
* EfficientNet / ResNet backbone upgrade
* Quantization & pruning for edge deployment
* Real-time inference benchmarking
* Deployment on NXP Edge platforms (eIQ SDK)

---

## 💾 Model Artifact

Saved trained model:

```
IESA_Defect_Model.h5
```

Can be converted to:

* TensorFlow Lite
* ONNX
* Edge TPU format

---

## ▶️ Execution Steps

1. Upload dataset to Google Colab
2. Extract dataset folders
3. Run training notebook
4. Evaluate performance
5. Export trained model

---

## 🧪 Use Cases

* Semiconductor wafer inspection
* Yield optimization
* Automated defect review
* Smart fab analytics
* Industry 4.0 manufacturing

---

## 👩‍💻 Authors

Akshaya Sreleka P.S
Varsha S
Aberlin karunya J.S
Rosiny C A
IESA Hackathon — Edge-AI Semiconductor Inspection

---

## 📜 License

This project is developed for academic and hackathon demonstration purposes.

---

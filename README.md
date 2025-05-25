# Drowsiness Detection System

This project implements a deep learning-based system for detecting drowsiness by analyzing eye and mouth features from images. It is designed to enhance safety in scenarios like driving or operating machinery where drowsiness is a critical hazard.

## Features
- **Custom CNN Models**: Utilizes custom convolutional neural networks (CNNs) to detect eye closure and yawning.
- **Ensemble Learning**: Combines predictions from multiple CNN models for better accuracy.
- **Data Augmentation**: Improves model robustness by artificially increasing the size and diversity of the training dataset.
- **MobileNetV2 Integration**: Incorporates MobileNetV2 for efficient and lightweight feature extraction.
- **Real-time Processing**: Can be extended for real-time video feed analysis.

---

## Getting Started

### Prerequisites
Ensure the following tools and libraries are installed:
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Google Colab (optional, for execution in the cloud)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Methodology

### CNN Architectures
1. **Eye Detection Model**:
   - Input shape: `(50, 50, 1)` (grayscale images).
   - Architecture: 
     - **Conv2D** layers for feature extraction.
     - **MaxPooling2D** layers for dimensionality reduction.
     - Fully connected layers for classification (open vs. closed).

2. **Mouth Detection Model**:
   - Input shape: `(50, 50, 1)` (grayscale images).
   - Similar architecture to the eye detection model but tuned for mouth feature detection.

3. **MobileNetV2**:
   - Lightweight pre-trained model used for feature extraction to improve performance.
   - Fine-tuned on the drowsiness dataset to enhance detection accuracy.

### Ensemble Modeling
The ensemble combines predictions from the custom CNN models and MobileNetV2. This approach leverages the strengths of each model, improving the overall detection accuracy.

### Data Augmentation
Data augmentation techniques are applied to the training dataset to prevent overfitting and improve generalization. Techniques include:
- **Flipping**: Horizontal flips.
- **Rotation**: Random rotations.
- **Zoom**: Random zoom-in/out.
- **Brightness Adjustments**: Varying brightness levels.

---

## Usage

### Running the Code
1. **Upload Image**: Use the `files.upload()` function in Colab or a similar file upload mechanism.
2. **Execute Detection**:
   ```python
   from google.colab import files
   
   # Upload an image
   uploaded = files.upload()
   
   for filename in uploaded.keys():
       frame = cv2.imread(filename)  # Load the image
       detect_drowsiness(frame, eye_models, mouth_models)  # Perform detection
   ```

---

### Workflow
1. **Image Preprocessing**:
   - Converts images to grayscale.
   - Resizes them to `50x50` pixels to match the CNN models’ input shape.
   - Normalizes pixel values to the range `[0, 1]`.

2. **Model Inference**:
   - Eye and mouth detection models, along with MobileNetV2, analyze the image.
   - Ensemble logic combines predictions to identify drowsiness.

3. **Output**:
   - Alerts if drowsiness is detected (e.g., eyes closed or yawning).

---

## File Structure
```
.
├── models/
│   ├── eye_model.h5        # Custom CNN model for eye detection
│   ├── mouth_model.h5      # Custom CNN model for mouth detection
│   ├── mobilenetv2_model.h5 # Fine-tuned MobileNetV2 model
├── dataset/
│   ├── yawn/               # Yawn images
│   ├── no_yawn/            # Non-yawn images
├── DrowsinessDetection.ipynb  # Model Notebook
├── README.md               # Project documentation

```

---

## Performance and Improvements
- **Baseline Accuracy**: Achieved using custom CNN models.
- **Improved Accuracy**: Ensemble learning and MobileNetV2 increased accuracy by approximately 10%.
- **Data Augmentation**: Boosted generalization on unseen data, especially under varying conditions like lighting and orientation.

### Future Work
- **Real-time Video Integration**: Extend the system to process video feeds in real-time.
- **Transfer Learning**: Experiment with other pre-trained models like EfficientNet or ResNet.
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and network architectures.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Thanks to the TensorFlow/Keras and OpenCV communities for their powerful tools.
- Inspiration drawn from various research papers on drowsiness detection and CNN architectures.

---

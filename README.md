# Multi-Class-U-Net-for-Retinal-OCT-Segmentation
This repository implements a **Multi-Class U-Net** architecture for **retinal OCT image segmentation**, focusing on segmenting critical retinal layers and pathological fluids. The model demonstrates state-of-the-art performance through the integration of dropout regularization, skip connections, and upsampling layers.  

Designed for precision and robustness, the model achieves exceptional accuracy and generalizability, making it suitable for applications in **medical image analysis** and scalable AI healthcare solutions.

---

## Model Architecture

The model follows a **U-Net-based design** with:

1. **Encoder (Contraction Path):**
   - Downsampling using Conv2D and MaxPooling layers.
   - Dropout for regularization.

2. **Bottleneck:**
   - Deeper feature extraction using high filter Conv2D layers.

3. **Decoder (Expansive Path):**
   - Transposed convolutions for upsampling.
   - Skip connections to recover spatial information.

4. **Output Layer:**
   - Final Conv2D layer with a **softmax activation** for multi-class segmentation.

---

## Key Results

The model was evaluated on **Retinal OCT Segmentation Tasks**, achieving **state-of-the-art performance** across multiple classes. Results are reported using the **Dice Coefficient** and **Intersection over Union (IoU)** metrics.

### **Dice Coefficient Results**
| **Class** | **Dice Coefficient** |
|-----------|----------------------|
| Class 1   | 0.9980               |
| Class 2   | 0.9585               |
| Class 3   | 0.9566               |
| Class 4   | 0.6566               |
| Class 5   | 0.9953               |
| Class 6   | 0.9079               |
| **Mean Dice Coefficient** | **0.9122** |

---

### **IoU (Intersection over Union) Results**
| **Class** | **IoU**       |
|-----------|---------------|
| Class 1   | 0.9973        |
| Class 2   | 0.9416        |
| Class 3   | 0.9374        |
| Class 4   | 0.5829        |
| Class 5   | 0.9936        |
| Class 6   | 0.8675        |
| **Mean IoU** | **0.8867** |

---

## Performance Summary

- **Mean Dice Coefficient**: 0.9122  
- **Mean IoU**: 0.8867  

The model delivers robust segmentation performance with near-perfect accuracy for several classes, showcasing its capability for **automated retinal disease analysis**. Class 4, exhibiting lower performance, highlights potential areas for improvement using advanced techniques like **attention mechanisms**, **hybrid transformer models**, or **data augmentation strategies**.

---

## Requirements

Install the required libraries before running the model:

```bash
pip install tensorflow keras numpy matplotlib
```

---

## Usage

### 1. Model Creation
Create and compile the U-Net model:

```python
from multi_unet_model import multi_unet_model

model = multi_unet_model(n_classes=8, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### 2. Training
Train the model using your dataset:

```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=50)
```

### 3. Inference
Predict segmentation masks for new data:

```python
predictions = model.predict(X_test)
```

---

## Applications

The U-Net model is tailored for **medical imaging applications**, specifically in **Retinal OCT Segmentation** for diagnosing:

- **Diabetic Macular Edema (DME)**  
- **Age-Related Macular Degeneration (AMD)**  
- **Retinal Layer Segmentation**  

The model's precision and scalability make it suitable for **real-world clinical applications**.

---

## Future Improvements

To further improve model performance:

1. Integrate **Attention Mechanisms** (e.g., Attention U-Net).  
2. Explore **Vision Transformer-based U-Nets** for better long-range dependency capture.  
3. Employ advanced augmentation techniques to improve generalizability for underperforming classes.

---

## References

- Olaf Ronneberger et al., *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, MICCAI 2015.  
- Research and datasets focused on **retinal disease segmentation** in OCT images.

---


## License

This project is released under the **MIT License**.

---

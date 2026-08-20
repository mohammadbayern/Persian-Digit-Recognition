# 🇮🇷 Persian Digits Recognition using Deep Learning

A Deep Learning project for **recognizing Persian handwritten digits (0–9)** using a Convolutional Neural Network (CNN).

This project covers the complete workflow from **image preprocessing and dataset preparation** to **CNN training, model evaluation, prediction, and a desktop GUI for recognizing numbers from real images**.

---

## 📌 Project Overview

The goal of this project is to build a Deep Learning model capable of recognizing Persian digits from images.

The project includes:

* 🖼️ Image preprocessing with OpenCV
* 🔄 Dataset creation and normalization
* ✂️ Train/Test dataset splitting
* 🧠 Convolutional Neural Network (CNN)
* 📊 Model evaluation
* 📈 Accuracy and Loss learning curves
* 🔲 Confusion Matrix
* 📋 Classification Report
* 📉 ROC-AUC analysis
* 🎯 Precision-Recall Curve
* 🔍 Prediction on unseen images
* 🖥️ Desktop GUI using Tkinter
* 🔢 Recognition of multi-digit numbers from images

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Persian-Digit-Recognition.git
cd Persian-Digit-Recognition

## Create a Virtual Environment: python -m venv venv

## Activate the environment: venv\Scripts\activate

## Linux / macOS: source venv/bin/activate

## Install Dependencies : pip install -r requirements.txt

##Run the Notebook

##Start Jupyter Notebook
## Then open  notebooks/Persian_Digit_Recognition.ipynb
## 🧠 Model Architecture

The model is implemented using TensorFlow/Keras.

```text
Input Image
    │
    ▼
32 × 32 × 1
    │
    ▼
Conv2D
9 Filters
3 × 3 Kernel
ReLU
    │
    ▼
MaxPooling2D
2 × 2
    │
    ▼
Flatten
    │
    ▼
Dense
32 Neurons
ReLU
    │
    ▼
Dense
16 Neurons
ReLU
    │
    ▼
Dense
10 Neurons
    │
    ▼
Digit Classification
0 ─ 9
```

### Model Definition

```python
model = Sequential()

model.add(
    Conv2D(
        filters=9,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 1)
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(32, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(10))
```

The final layer contains **10 neurons**, corresponding to the ten digit classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

# 🔄 Data Preprocessing

Before training the CNN, the input images go through several preprocessing steps.

### 1. Read Images

Images are loaded using OpenCV:

```python
im = cv2.imread(pic_all)
```

### 2. Convert to Grayscale

Color information is not required for digit recognition, so images are converted to grayscale:

```python
im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
```

This reduces the input from three color channels to a single channel.

### 3. Resize Images

All images are resized to:

```text
32 × 32
```

```python
im3 = cv2.resize(im2, (32, 32))
```

This ensures that every input has the same dimensions.

### 4. Invert Image

The image colors are inverted:

```python
im4 = cv2.bitwise_not(im3)
```

This makes the digit representation suitable for the model's training data.

### 5. Normalize Pixel Values

Pixel values are converted from:

```text
0 – 255
```

to:

```text
0 – 1
```

using:

```python
im5 = im4 / 255
```

Normalization helps the neural network train more effectively.

---

# 💾 Dataset Preparation

The dataset is organized using folders representing digit classes.

Example:

```text
dataset/
│
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
```

The folder name is used as the class label:

```python
label.append(int(folder))
```

The processed images and labels are converted into NumPy arrays:

```python
x = np.array(dimension)
y = np.array(label)
```

The dataset is then saved in compressed NumPy format:

```python
np.savez_compressed(
    "persian_digits_dataset.npz",
    x=x,
    y=y
)
```

This makes it possible to reuse the processed dataset without repeating the preprocessing step.

---

# ✂️ Train/Test Split

The dataset is divided into training and testing sets using an **80/20 split**:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    test_size=0.2
)
```

Approximately:

```text
80% → Training
20% → Testing
```

The test set is kept separate and used to evaluate the model on unseen data.

---

# 🧩 Preparing Images for CNN

Since `Conv2D` expects a channel dimension, a new axis is added:

```python
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
```

The final image shape becomes:

```text
32 × 32 × 1
```

where `1` represents the grayscale channel.

---

# ⚙️ Model Compilation

The model uses the Adam optimizer:

```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)
```

### Configuration

| Component     | Configuration                   |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Loss Function | Sparse Categorical Crossentropy |
| From Logits   | True                            |
| Metric        | Accuracy                        |
| Classes       | 10                              |

---

# 🚀 Model Training

The CNN is trained for 25 epochs:

```python
h = model.fit(
    x_train,
    y_train,
    epochs=25,
    validation_data=(x_test, y_test)
)
```

During training, both training and validation performance are monitored.

---

# 📊 Model Evaluation

After training, the model is evaluated using the test dataset:

```python
loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("Loss:", loss)
print("Accuracy:", accuracy)
```

The project also uses several additional evaluation techniques.

---

## 🎯 Accuracy

Accuracy measures the percentage of correctly classified digits.

```python
accuracy_score(y_test, out)
```

This provides an overall view of classification performance.

---

# 🔲 Confusion Matrix

A confusion matrix is used to analyze which digits are correctly classified and which digits are confused with each other.

```python
cm = confusion_matrix(y_test, out)
```

It is visualized using a heatmap.

The confusion matrix is particularly useful for identifying difficult digit pairs.

For example:

```text
True Digit → Predicted Digit
```

This allows us to see whether the model frequently confuses certain classes.

---

# 📋 Classification Report

The project generates a classification report:

```python
print(
    classification_report(
        y_test,
        out
    )
)
```

This provides metrics such as:

* Precision
* Recall
* F1-score
* Support

for each digit class.

---

# 📈 Learning Curves

Training and validation accuracy are plotted across epochs:

```python
plt.plot(
    h.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    h.history["val_accuracy"],
    label="Validation Accuracy"
)
```

These curves help analyze:

* Learning progress
* Generalization
* Possible overfitting
* Training stability

---

# 📉 Loss Curve

Training and validation loss are also visualized:

```python
plt.plot(
    h.history["loss"],
    label="Train Loss"
)

plt.plot(
    h.history["val_loss"],
    label="Validation Loss"
)
```

A decreasing training and validation loss generally indicates that the model is learning useful patterns.

A large gap between training and validation loss can indicate overfitting.

---

# 📊 ROC-AUC

The project also evaluates the multi-class classifier using ROC-AUC.

First, the model outputs logits:

```python
labels_logits = model.predict(x_test)
```

Then probabilities are calculated using Softmax:

```python
out_pro = tf.nn.softmax(
    labels_logits
).numpy()
```

The labels are binarized for multi-class ROC-AUC:

```python
test_label_bin = label_binarize(
    y_test,
    classes=np.arange(10)
)
```

Finally:

```python
auc = roc_auc_score(
    test_label_bin,
    out_pro,
    multi_class="ovr"
)
```

The project uses the **One-vs-Rest (OvR)** strategy for multi-class ROC-AUC evaluation.

---

# 📐 Precision-Recall Curve

The project also calculates a Precision-Recall curve for individual classes.

```python
precision, recall, thresholds = precision_recall_curve(
    test_label_bin[:, 0],
    out_pro[:, 0]
)
```

This provides another perspective on classification performance, especially when analyzing individual digit classes.

---

# 🔮 Making Predictions

The trained model produces logits:

```python
labels_logits = model.predict(x_test)
```

The predicted class is selected using `argmax`:

```python
out = np.argmax(
    labels_logits,
    axis=1
)
```

For example:

```text
Model Output
     ↓
10 class scores
     ↓
argmax
     ↓
Predicted Digit
```

---

# 🔢 Multi-Digit Number Recognition

One of the more interesting parts of this project is that the model is not limited to classifying a single isolated digit.

The project contains an image-processing pipeline capable of detecting individual digits inside an image.

The process is approximately:

```text
Input Image
     │
     ▼
Grayscale
     │
     ▼
Gaussian Blur
     │
     ▼
Otsu Thresholding
     │
     ▼
Contour Detection
     │
     ▼
Sort Digits from Left → Right
     │
     ▼
Crop Individual Digits
     │
     ▼
Resize to 32 × 32
     │
     ▼
Normalize
     │
     ▼
CNN Prediction
     │
     ▼
Combine Predictions
     │
     ▼
Final Number
```

The contours are sorted from left to right:

```python
contours = sorted(
    contours,
    key=lambda c: cv2.boundingRect(c)[0]
)
```

This allows the system to reconstruct the original number.

For example:

```text
Image:

  3    7    5

 ↓    ↓    ↓

CNN  CNN  CNN

 ↓    ↓    ↓

  3    7    5

      ↓

    "375"
```

---

# 🖥️ Desktop Application

The project also includes a simple desktop GUI using **Tkinter**.

The application allows the user to:

1. Load an image.
2. Process the image.
3. Detect individual digits.
4. Classify each digit using the CNN.
5. Display the predicted number.
6. Visualize the processed image.

The interface includes:

```text
┌──────────────────────────────────────┐
│       Persian Digits Recognition     │
│                                      │
│          [ Load Image ]              │
│                                      │
│          ☑ Show Process              │
│                                      │
│       The Number Is: 375             │
│                                      │
│          [ Processed Image ]         │
│                                      │
└──────────────────────────────────────┘
```

---

# 🛠️ Technologies

The project was developed using:

| Technology         | Purpose                        |
| ------------------ | ------------------------------ |
| Python             | Main programming language      |
| TensorFlow / Keras | Deep Learning                  |
| CNN                | Digit classification           |
| OpenCV             | Image processing               |
| NumPy              | Numerical computation          |
| Pandas             | Data handling                  |
| Scikit-learn       | Dataset splitting & evaluation |
| Matplotlib         | Visualization                  |
| Seaborn            | Confusion matrix visualization |
| Tkinter            | Desktop GUI                    |
| Pillow             | Image handling                 |

---

# 📁 Project Structure

A recommended GitHub structure for this project is:

```text
Persian-Digits-Recognition/
│
├── dataset/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 9/
│
├── notebooks/
│   └── Persian_Number_Recognition.ipynb
│
├── models/
│   └── model.keras
│
├── images/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── prediction_example.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Persian-Digits-Recognition.git
```

Move into the project directory:

```bash
cd Persian-Digits-Recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Main dependencies:

```text
numpy
pandas
matplotlib
seaborn
opencv-python
scikit-learn
tensorflow
keras
pillow
```

---

# ▶️ How to Run

### 1. Prepare the Dataset

Place the digit images inside:

```text
dataset/
```

with one folder for each class:

```text
dataset/
├── 0/
├── 1/
├── 2/
├── ...
└── 9/
```

### 2. Run the Notebook

Open:

```text
notebooks/Persian_Number_Recognition.ipynb
```

and execute the cells in order.

### 3. Train the CNN

Run the dataset preparation and training cells.

### 4. Evaluate the Model

Run the evaluation sections to generate:

* Accuracy
* Confusion Matrix
* Classification Report
* ROC-AUC
* Precision-Recall Curve
* Learning Curves

### 5. Run the GUI

After training the model, the Tkinter application can be used to load an image and recognize the number.

---

# 🎯 Key Features

### Deep Learning

* CNN-based image classification
* 10-class digit recognition
* ReLU activation
* Max Pooling
* Dense classification layers
* Adam optimizer

### Computer Vision

* Grayscale conversion
* Image inversion
* Gaussian Blur
* Otsu Thresholding
* Contour detection
* Digit segmentation
* Image resizing
* Pixel normalization

### Model Evaluation

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall Curve
* Training/Validation curves

### Application

* Multi-digit number recognition
* Left-to-right digit ordering
* Real image inference
* Tkinter graphical interface

---

# 📚 What I Learned

This project helped me practice the complete Deep Learning workflow:

```text
Raw Images
    ↓
Data Collection
    ↓
Image Preprocessing
    ↓
Dataset Creation
    ↓
Train/Test Split
    ↓
CNN Design
    ↓
Model Training
    ↓
Evaluation
    ↓
Computer Vision Pipeline
    ↓
Real-World Prediction
    ↓
Desktop Application
```

The project demonstrates how a trained CNN can be combined with traditional Computer Vision techniques to build a practical image-recognition application.

---

# 🔮 Future Improvements

Possible improvements for future versions:

* [ ] Increase CNN depth and model capacity
* [ ] Add Data Augmentation
* [ ] Use Batch Normalization
* [ ] Add Dropout
* [ ] Improve digit segmentation
* [ ] Handle touching/overlapping digits
* [ ] Save and load the trained model separately
* [ ] Add confidence scores to predictions
* [ ] Add support for handwritten multi-digit numbers
* [ ] Create a more modern GUI
* [ ] Deploy the model as a web application
* [ ] Compare CNN with transfer-learning approaches


## 🔢 Multi-Digit Persian Number Recognition

The project also supports recognition of **multi-digit Persian numbers** from a single image.

Instead of classifying only an isolated digit, the system detects individual digits using OpenCV and then classifies each digit using the trained CNN model.

### Pipeline

```text
Input Image
     ↓
Preprocessing
     ↓
Contour Detection
     ↓
Digit Segmentation
     ↓
Left-to-Right Sorting
     ↓
CNN Classification
     ↓
Combine Digit Predictions
     ↓
Final Number
---

# 👨‍💻 Author

**Mohammad Nahvi**

This project was developed as part of my journey in **Machine Learning, Deep Learning, Computer Vision, and Python development**.

---

## ⭐ If you found this project useful

Feel free to ⭐ star the repository and explore the code.

**Built with Python, TensorFlow, Keras and OpenCV.**

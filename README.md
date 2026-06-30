🤖 Hand Gesture Recognition using CNN

 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to recognize different hand gestures from input images. 

The trained model classifies a given hand gesture into one of five predefined categories and displays the predicted gesture along with its confidence score.


✋ Supported Hand Gestures

- ✊ Fist
- 👌 OK
- ✋ Palm
- ✌️ Peace
- 👍 Thumbs Up



🛠️ Technologies Used

- 🐍 Python 3.11
- 🧠 TensorFlow
- 🔷 Keras
- 🔢 NumPy
- 📊 Matplotlib
- 🖼️ Pillow (PIL)

📂 Project Structure

```text
Hand gesture/
│
├── fist/
├── ok/
├── palm/
├── peace/
├── thumbs up/
├── main.py
├── test.jpeg
└── hand_gesture_model.keras
```

🏗️ CNN Architecture

- 🔹 Input Layer
- 🔹 Rescaling Layer
- 🔹 Conv2D (32 Filters)
- 🔹 MaxPooling2D
- 🔹 Conv2D (64 Filters)
- 🔹 MaxPooling2D
- 🔹 Conv2D (128 Filters)
- 🔹 MaxPooling2D
- 🔹 Flatten Layer
- 🔹 Dense Layer (128 Neurons)
- 🔹 Dropout (0.5)
- 🔹 Output Layer (Softmax)

---

⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| 🖼️ Image Size | 128 × 128 |
| 📦 Batch Size | 32 |
| 🔁 Epochs | 20 |
| ⚡ Optimizer | Adam |
| 📉 Loss Function | Sparse Categorical Crossentropy |



🚀 How to Run

 1️⃣ Install Required Libraries

```bash
pip install tensorflow numpy matplotlib pillow
```

2️⃣ Place the Dataset

Copy the dataset folders into the project directory.

 3️⃣ Add a Test Image

Place the test image in the project folder.

Example:

```text
test.jpeg
```
 4️⃣ Run the Project

```bash
py -3.11 main.py
```


📈 Output

The application displays:

- 🖼️ Input Test Image
- 🎯 Predicted Gesture
- 📊 Confidence Percentage
- 📈 Training Accuracy Graph
- 📉 Training Loss Graph
- 📊 Prediction Confidence Graph

### Sample Output

```text
Predicted Gesture : Palm
Confidence : 99.86%
```

⭐ Features

- ✅ CNN-Based Image Classification
- ✅ Five Hand Gesture Recognition
- ✅ Confidence Score Prediction
- ✅ Accuracy Visualization
- ✅ Loss Visualization
- ✅ Prediction Confidence Graph
- ✅ Simple and Easy to Use


🔮 Future Enhancements

- 🎥 Real-Time Webcam Gesture Recognition
- 📱 Mobile Application Integration
- ➕ Additional Gesture Classes
- 🚀 Higher Accuracy using Transfer Learning
- 🌐 Web Application Deployment

📌 Conclusion

This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** for hand gesture recognition. The model accurately classifies five different hand gestures and provides prediction confidence along with graphical visualization of the training process.

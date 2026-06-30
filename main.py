# ============================
# Step 1: Extract Dataset
# ============================


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
print("Dataset Extracted")


# ============================
# Step 2: Load Dataset
# ============================

import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 32

dataset_path = "."

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("Classes:", class_names)


# ============================
# Step 3: CNN Model
# ============================

from tensorflow.keras import Sequential
from tensorflow.keras.layers import *

model = Sequential([

    Input(shape=(128,128,3)),
    Rescaling(1./255),

    Conv2D(32,(3,3),activation="relu"),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(),

    Flatten(),

    Dense(128,activation="relu"),
    Dropout(0.5),

    Dense(len(class_names),activation="softmax")
])


# ============================
# Step 4: Compile
# ============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================
# Step 5: Train
# ============================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    shuffle=False
)
import matplotlib.pyplot as plt

# Accuracy Graph
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss Graph
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================
# Step 6: Accuracy
# ============================

loss, accuracy = model.evaluate(val_ds)

print("Validation Accuracy:", accuracy)


# ============================
# Step 7: Save Model
# ============================

model.save("hand_gesture_model.keras")

print("Model Saved Successfully")


# ============================
# Step 8: Test Image Prediction
# ============================

from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Test image
img_path = "test.jpeg"      #image

# ---------- Display Original Image ----------
display_img = Image.open(img_path)

# ---------- Prediction ----------
img = image.load_img(
    img_path,
    target_size=(128,128)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

# Model lo already Rescaling(1./255) undi kabatti
# img_array = img_array / 255.0  <-- idi rayakandi

prediction = model.predict(
    img_array,
    verbose=0
)[0]

predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100
plt.figure(figsize=(6,4))
plt.bar(class_names, prediction)
plt.title("Prediction Probabilities")
plt.xlabel("Classes")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.show()

# ---------Display Result-----------
fig, ax = plt.subplots(figsize=(5,7))

ax.imshow(display_img)
ax.axis("off")

# Bottom text
fig.subplots_adjust(bottom=0.20)

fig.text(
    0.5,
    0.08,
    f"Predicted Gesture: {predicted_class}\nConfidence: {confidence:.2f}%",
    ha="right",
    fontsize=14,
    fontweight="bold",
    color="black"
)

plt.show()

# ---------- Terminal Output ----------
print("Predicted Gesture:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("\nAll Probabilities:")

for cls, prob in zip(class_names, prediction):
    print(f"{cls}: {prob*100:.2f}%")
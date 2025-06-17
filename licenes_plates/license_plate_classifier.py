import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


data_dir = 'license_plates_img'
img_height, img_width = 50, 100
batch_size = 32
epochs = 20

def load_all_images(filenames):
    images = []
    for filename in filenames:
        img = load_img(os.path.join(data_dir, filename), target_size=(img_height, img_width))
        img = img_to_array(img) / 255.0 
        images.append(img)
    return np.array(images)

filenames = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
labels = [filename.split('_')[0] for filename in filenames]

classes = np.unique(labels)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
num_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    filenames, 
    [class_to_idx[label] for label in labels],
    test_size=0.2,
    stratify=labels
)

X_train_images = load_all_images(X_train)
X_test_images = load_all_images(X_test)

y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_images,
    y_train_categorical,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test_images, y_test_categorical)
)


model.save('license_plate_classifier.h5')
import pickle

config = {
    'classes': classes,
    'num_classes': num_classes,
    'img_height': img_height,
    'img_width': img_width
}

with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)

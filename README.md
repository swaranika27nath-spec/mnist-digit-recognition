# mnist-digit-recognition
Deep learning project to recognize handwritten digits using convolutional neural networks (CNN) and the MNIST dataset.
# mnist_cnn.py
# Digit Recognition using CNN (TensorFlow/Keras, MNIST dataset)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape data to fit CNN input: (samples, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32,
          validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Predict one sample (optional)
import numpy as np
sample = np.expand_dims(X_test[0], axis=0)
predicted_class = model.predict(sample).argmax(axis=1)[0]
print(f"Predicted digit for first test sample: {predicted_class}")
print(f"True digit for first test sample: {y_test[0]}")

import streamlit as st
#import tensorflow as tf
#import torch
#import numpy as np
#import pandas as pd
#from tensorflow import keras
#from torch import nn
#from code_executor import code_execution_widget # In-Browser Code Executor disabled

st.title("Deep Learning Basics")

# Sidebar persistent code execution widget (temporarily disabled)
# code_execution_widget()

st.markdown("## 📌 Deep Learning Fundamentals")

# Expandable Section: Introduction to Deep Learning
with st.expander("🧠 What is Deep Learning?", expanded=False):
    st.markdown("### Understanding Neural Networks")
    st.code('''
# A simple feedforward neural network with TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# View model summary
model.summary()
''', language="python")

# Expandable Section: Implementing Neural Networks with TensorFlow/Keras
with st.expander("🔬 Implementing Neural Networks with TensorFlow/Keras", expanded=False):
    st.markdown("### Building a Basic Neural Network")
    st.code('''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(20,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
''', language="python")

    st.markdown("### Training the Model")
    st.code('''
# Train model with dummy data
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(0, 2, size=(1000,))

model.fit(X_train, y_train, epochs=10, batch_size=32)
''', language="python")

# Expandable Section: Implementing Neural Networks with PyTorch
with st.expander("🔥 Implementing Neural Networks with PyTorch", expanded=False):
    st.markdown("### Defining a Neural Network in PyTorch")
    st.code('''
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)

# Initialize model
model = NeuralNetwork()
print(model)
''', language="python")

    st.markdown("### Training the Model")
    st.code('''
# Training loop (simplified)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(torch.rand(100, 20))  # Dummy input
    loss = criterion(outputs, torch.rand(100, 1))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
''', language="python")

# Expandable Section: Transfer Learning
with st.expander("📦 Transfer Learning with Pretrained Models", expanded=False):
    st.markdown("### Using Pretrained Models in TensorFlow")
    st.code('''
from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
print(model.summary())
''', language="python")

    st.markdown("### Using Pretrained Models in PyTorch")
    st.code('''
import torchvision.models as models

# Load pretrained ResNet model
model = models.resnet50(pretrained=True)
print(model)
''', language="python")

# Expandable Section: Saving & Loading Models
with st.expander("💾 Saving & Loading Models", expanded=False):
    st.markdown("### Saving a TensorFlow Model")
    st.code('''
# Save the model
model.save("my_model.h5")

# Load the model
from tensorflow.keras.models import load_model
model = load_model("my_model.h5")
''', language="python")

    st.markdown("### Saving a PyTorch Model")
    st.code('''
# Save model state
torch.save(model.state_dict(), "model.pth")

# Load model state
model.load_state_dict(torch.load("model.pth"))
''', language="python")

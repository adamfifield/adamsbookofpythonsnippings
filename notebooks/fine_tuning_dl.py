"""
Fine-Tuning Deep Learning Models

This script covers techniques for:
- Transfer learning and fine-tuning pre-trained models
- Hyperparameter tuning using Grid Search and Bayesian Optimization
- Regularization techniques (Dropout, L1/L2 weight decay)
- Efficient model training and deployment
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import optuna

# ----------------------------
# 1. Transfer Learning & Fine-Tuning
# ----------------------------

base_model = ResNet50(weights="imagenet", include_top=False)
for layer in base_model.layers[:100]:  # Freeze first 100 layers
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(10, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ----------------------------
# 2. Hyperparameter Tuning (Bayesian Optimization with Optuna)
# ----------------------------

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.2, 0.6)

    x = Dense(256, activation="relu")(base_model.output)
    x = Dropout(dropout)(x)
    output_layer = Dense(10, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])

    return model.evaluate(x_train, y_train, verbose=0)[1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

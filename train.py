import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tf import keras
from tf.keras import layers
from tf.keras.layers import Dense, Dropout, Input, Flatten, Rescaling, Activation
from tf.keras.callbacks import ModelCheckpoint, EarlyStopping
from tf.data.experimental import AUTOTUNE
from tf.keras.preprocessing import image_dataset_from_directory
from tf.keras.applications import DenseNet121
from tf.keras.models import Model
from tf.keras.optimizers import Adam
from tf.keras.optimizers.schedules import ExponentialDecay
from tf.keras.applications.densenet import Preprocess_Input



flowers_path = './flowers'
classes = os.listdir(flowers_path)
print(classes)

# Passable Parameters

target_size = (150, 150)
input_shape = target_size + (3,)
batch_size = 64

train_ds = image_dataset_from_directory(
    flowers_path,
    validation_split=0.2,          
    subset="training",             
    seed=42,                     
    image_size=target_size,         
    batch_size=batch_size,         
    label_mode="categorical",      
    class_names=classes
)

val_ds = image_dataset_from_directory(
    flowers_path,
    validation_split=0.2,
    subset="validation",         
    seed=42,
    image_size=target_size,
    batch_size=batch_size,
    label_mode="categorical",
    class_names=classes
)

# Preprocessing for densenset models
def preprocess_image_densenet(image, label):
    image = Preprocess_Input(image)
    return image, label

# Map preprocessed images to train and val
train_ds = train_ds.map(preprocess_image_densenet, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess_image_densenet, num_parallel_calls=AUTOTUNE)

def get_model():
    # Get base model 
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape)
    # Freeze the layers in base model
    for layer in base_model.layers:
        layer.trainable = False
    
    base_model_ouput = base_model.output
    
    # Add new layers
    x = layers.Rescaling(1./255)(base_model.output)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model


def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
]

train_ds_aug = train_ds.map(lambda x, y: (data_augmentation(x), y))

model = get_model()

# Learning Rate Optimizer
scheduler = ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.90,
    staircase=True
)

optimizer = Adam(learning_rate=scheduler, weight_decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(
    'densenet121_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

history_fine=model.fit(train_ds_aug, epochs=30, verbose=1, validation_data=val_ds, callbacks=[checkpoint, early_stopping])

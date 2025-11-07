import tensorflow as tf
from keras import layers, models
import os

def build_model(img_size=96, num_classes=13, lr=5e-4):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size,img_size,3),
        include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = layers.Input((img_size,img_size,3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)

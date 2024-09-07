import tensorflow as tf

model = tf.keras.applications.ResNet50(weights="imagenet")

model.export("resnet")

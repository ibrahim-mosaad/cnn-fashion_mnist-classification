import math
import tensorflow as tf
from data import prepare_datasets
from model import build_cnn_model

BATCH_SIZE = 32
EPOCHS = 10

def train():
    train_ds, test_ds, num_train, _ = prepare_datasets(BATCH_SIZE)

    model = build_cnn_model()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(num_train / BATCH_SIZE)
    )

    model.save("saved_model/cnn_fashion_mnist")
    print("Model saved successfully!")

if __name__ == "__main__":
    train()

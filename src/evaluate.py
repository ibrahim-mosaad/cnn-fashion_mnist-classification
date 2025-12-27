import math
import tensorflow as tf
from data import prepare_datasets

BATCH_SIZE = 32

def evaluate():
    _, test_ds, _, num_test = prepare_datasets(BATCH_SIZE)

    model = tf.keras.models.load_model(
        "saved_model/cnn_fashion_mnist"
    )

    loss, accuracy = model.evaluate(
        test_ds,
        steps=math.ceil(num_test / BATCH_SIZE)
    )

    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()

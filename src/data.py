import tensorflow as tf
import tensorflow_datasets as tfds

def load_data():
    dataset, metadata = tfds.load(
        'fashion_mnist',
        as_supervised=True,
        with_info=True
    )

    train_ds = dataset['train']
    test_ds  = dataset['test']

    num_train = metadata.splits['train'].num_examples
    num_test  = metadata.splits['test'].num_examples

    return train_ds, test_ds, num_train, num_test


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label


def prepare_datasets(batch_size=32):
    train_ds, test_ds, num_train, num_test = load_data()

    train_ds = (
        train_ds
        .map(preprocess)
        .shuffle(num_train)
        .batch(batch_size)
        .repeat()
    )

    test_ds = (
        test_ds
        .map(preprocess)
        .batch(batch_size)
    )

    return train_ds, test_ds, num_train, num_test

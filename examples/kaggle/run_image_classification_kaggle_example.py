"""
Usage example of Targetran with TFDS for image classification model training.

On a Kaggle Notebook, select the "Accelerator": TPUv3-8
"""
import os
from typing import Tuple

# Needed for the Kaggle Notebook.
os.system("pip install targetran")

import matplotlib.pylab as plt
import numpy as np
import targetran.tf as tt
import tensorflow as tf
import tensorflow_datasets as tfds
from targetran.utils import image_only
from tensorflow.keras import layers, Model


def setup_accelerators_and_get_strategy() -> tf.distribute.Strategy:
    """
    References:
    https://www.kaggle.com/docs/tpu
    https://www.kaggle.com/mgornergoogle/five-flowers-with-keras-and-xception-on-tpu
    """
    try:
        # Detect and init the TPU.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        # Instantiate a distribution strategy.
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"Using TPU.")
    except ValueError:
        # Fallback to GPU, if exists. Strategy for GPU or multi-GPU machines.
        strategy = tf.distribute.MirroredStrategy()
        print(f"Not using TPU.")
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
    return strategy


def make_model(
        image_size: Tuple[int, int],
        num_classes: int
) -> Model:
    """
    A pre-trained Xception model is used.
    """
    base_model = tf.keras.applications.Xception(
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False
    )
    base_model.trainable = False

    image_input = layers.Input(shape=(image_size[0], image_size[1], 3))
    x = image_input
    x = tf.keras.applications.xception.preprocess_input(x)
    x = base_model(x)

    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(image_input, output)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["acc"]
    )
    model.summary()
    return model


def load_data() -> tf.data.Dataset:
    """
    See:
    https://www.tensorflow.org/datasets/catalog/tf_flowers
    """
    ds = tfds.load(
        "tf_flowers",
        split="train",
        shuffle_files=False,
        as_supervised=True,
        try_gcs=True
    )
    return ds.cache()


def split_ds(
        ds: tf.data.Dataset,
        num_val_samples: int,
        seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split the dataset into two for training and validation.
    """
    ds = ds.shuffle(2048, seed=seed)
    ds_val = ds.take(num_val_samples)
    ds_train = ds.skip(num_val_samples)
    return ds_train, ds_val


def transform_and_batch(
        ds_train: tf.data.Dataset,
        ds_val: tf.data.Dataset,
        image_size: Tuple[int, int],
        batch_size: int,
        seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Apply data augmentation (to training set only) and batching.
    """

    def set_image_shape(
            image: tf.Tensor,
            label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Somehow the dataset loses the shape information after transformation,
        so this step may be needed to tackle the following exception when
        running on TPU:
            Compilation failure: Dynamic Spatial Convolution is not supported
        """
        image.set_shape((image_size[0], image_size[1], 3))
        return image, label

    affine_transform = tt.TFCombineAffine(
        [tt.TFRandomFlipLeftRight(),
         tt.TFRandomRotate(),
         tt.TFRandomShear(),
         tt.TFRandomTranslate()],
        num_selected_transforms=2,
        seed=seed
    )

    auto = tf.data.AUTOTUNE

    # The `image_only` utility is used for image classification training.
    ds_train = ds_train \
        .shuffle(2048, seed=seed) \
        .map(image_only(affine_transform), num_parallel_calls=auto) \
        .map(image_only(tt.TFResize(image_size)), num_parallel_calls=auto) \
        .map(set_image_shape, num_parallel_calls=auto) \
        .batch(batch_size) \
        .prefetch(auto)

    ds_val = ds_val \
        .map(image_only(tt.TFResize(image_size)), num_parallel_calls=auto) \
        .map(set_image_shape, num_parallel_calls=auto) \
        .batch(batch_size) \
        .prefetch(auto)

    return ds_train, ds_val


def save_plot(
        ds: tf.data.Dataset,
        num_rows: int,
        num_cols: int,
        figure_size_inches: Tuple[float, float] = (10.0, 10.0)
) -> None:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size_inches)

    for n, sample in enumerate(ds.unbatch().take(num_rows * num_cols)):

        image, label = [tensor.numpy() for tensor in sample]
        image = image.astype(np.int32)

        if num_rows == 1 or num_cols == 1:
            ax = axes[n]
        else:
            ax = axes[n // num_cols][n % num_cols]

        ax.imshow(image)
        ax.set_title(str(label))
        ax.set_axis_off()

    fig.set_tight_layout(True)
    fig.savefig("figure.png")


def train_model(
        model: Model,
        ds_train: tf.data.Dataset,
        ds_val: tf.data.Dataset,
        max_num_epochs: int
) -> None:
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", patience=6,
            factor=0.1, min_lr=1e-05, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10,
            restore_best_weights=True, verbose=1
        ),
    ]
    model.fit(
        ds_train,
        epochs=max_num_epochs,
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=2
    )


def main() -> None:
    strategy = setup_accelerators_and_get_strategy()

    image_height = 331
    image_width = 331
    num_val_samples = 256
    batch_size = 16 * strategy.num_replicas_in_sync  # See TPU docs.
    max_num_epochs = 5  # With early-stopping, this can be a large number.

    image_size = (image_height, image_width)

    with strategy.scope():
        model = make_model(image_size, num_classes=5)

    ds = load_data()
    ds_train, ds_val = split_ds(ds, num_val_samples)

    ds_train, ds_val = transform_and_batch(
        ds_train, ds_val, image_size, batch_size
    )
    save_plot(ds_train, num_rows=4, num_cols=3)
    train_model(model, ds_train, ds_val, max_num_epochs)


if __name__ == "__main__":
    main()

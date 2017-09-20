import argparse
import os
from utils import build_fit_dataset
from utils import CarNDBehavioralCloningSequence as Sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
import matplotlib.pyplot as plt


def build_model(input_shape):
    """Build sequential model based on NVIDIA architecture."""
    model = Sequential()
    model.add(Cropping2D(((50,20), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x - 180.0) / 180.0))
    model.add(Conv2D(24, [5, 5], strides=(2, 2), padding="valid", activation="relu"))
    model.add(Conv2D(36, [5, 5], strides=(2, 2), padding="valid", activation="relu"))
    model.add(Conv2D(48, [5, 5], strides=(2, 2), padding="valid", activation="relu"))
    # Dropout layers for better generalization
    model.add(Dropout(.1))
    model.add(Conv2D(64, [3, 3], padding="valid", activation="relu"))
    model.add(Conv2D(64, [3, 3], padding="valid", activation="relu"))
    # Dropout layers for better generalization
    model.add(Dropout(.1))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    # Dropout layers for better generalization
    model.add(Dropout(.1))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Driver")
    parser.add_argument(
        "-d",
        type=str,
        help="Path to train data.",
        dest="data_path"
    )
    parser.add_argument(
        "-b",
        type=int,
        help="Batch size",
        default=128,
        nargs="?",
        dest="batch_size"
    )
    parser.add_argument(
        "-e",
        type=int,
        help="epochs",
        default=2,
        nargs="?",
        dest="epochs"
    )
    parser.add_argument(
        "-l",
        type=str,
        help="model_weights",
        nargs="?",
        dest="model_weights"
    )
    parser.add_argument(
        "-o",
        type=str,
        help="model_output",
        nargs="?",
        dest="model_output",
        default="model.h5",
    )
    args = parser.parse_args()

    # Buld model
    model = build_model((160, 320, 3))

    # Load weights and variables for transfer learning
    if args.model_weights is not None:
        model.load_weights(args.model_weights, by_name=True)

    # Compiles code and sumarry
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    # Build dataset, python list of tuples.
    # Each tuple is a file_path, image_trasformations, steering, steering_trasformations
    dataset = build_fit_dataset(args.data_path)
    # Split dataset in train and validation
    train_dataset, test_dataset = train_test_split(dataset, test_size=.2)
    # Create sequences for parallel access to the dataset.
    # Generator will read image from file, transform it returned for training.
    train_sequence = Sequence(train_dataset, args.batch_size)
    test_sequence = Sequence(test_dataset, args.batch_size)

    # Train the model
    model.fit_generator(
        train_sequence, len(train_sequence),
        epochs=args.epochs,
        validation_data=test_sequence, validation_steps=len(test_sequence),
        max_queue_size=10, workers=6,
        verbose=2)

    # Store final model and weights
    model.save(args.model_output)
    name, ext = os.path.splitext(args.model_output)
    model.save(name + "_weights" + ext)

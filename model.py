import argparse
from utils import build_fit_dataset
from utils import CarNDBehavioralCloningSequence as Sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
import matplotlib.pyplot as plt

def build_model(input_shape):
    model = Sequential()
    model.add(Cropping2D(((50,20), (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x - 180.0) / 180.0))
    model.add(Conv2D(24, [5, 5], strides=(2, 2), padding="valid", activation="relu"))
    model.add(Conv2D(36, [5, 5], strides=(2, 2), padding="valid", activation="relu"))
    model.add(Conv2D(48, [5, 5], strides=(2, 2), padding="valid", activation="relu"))
    model.add(Dropout(.1))
    model.add(Conv2D(64, [3, 3], padding="valid", activation="relu"))
    model.add(Conv2D(64, [3, 3], padding="valid", activation="relu"))
    model.add(Dropout(.1))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
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

    model = build_model((160, 320, 3))
    if args.model_weights is not None:
        model.load_weights(args.model_weights, by_name=True)

    model.compile(loss="mse", optimizer="adam")
    model.summary()

    dataset = build_fit_dataset(args.data_path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=.2)
    train_sequence = Sequence(train_dataset, args.batch_size)
    test_sequence = Sequence(test_dataset, args.batch_size)

    model.fit_generator(
        train_sequence, len(train_sequence),
        epochs=args.epochs,
        validation_data=test_sequence, validation_steps=len(test_sequence),
        max_queue_size=10, workers=6,
        verbose=2)
    model.save(args.model_output)

import json
from pickletools import optimize
import numpy as np
from sklearn.model_selection import train_test_split
import keras.api._v2.keras as keras
import matplotlib.pyplot as plt

INPUT_FILE = '/dev/github/year-guesser/data/output.json'
EXAMPLE_FILE = '/dev/github/year-guesser/data/example.json'
YEAR_MIN = 1960
YEAR_MAX = 2022

def load_data():
    with open(INPUT_FILE, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfccs"])
    labels = np.array(data["labels"])
    #targets = 1 - ((YEAR_MAX - labels.astype(int)) / (YEAR_MAX - YEAR_MIN))
    targets = labels.astype(int) - YEAR_MIN

    with open(EXAMPLE_FILE, "r") as fp:
        example_data = json.load(fp)

    examples = np.array(example_data["mfccs"])

    return inputs, targets, examples

def plot_history(history):

    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    inputs, targets, examples = load_data()

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)
    
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(60, activation="softmax")
        ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), batch_size=32, epochs=50)
    plot_history(history)
    
    predictions = model.predict(examples)[0]     
    sum = 0

    for i, weight in enumerate(predictions):
        print(i, weight)
        sum += i * weight
    
    guessed_year = round(sum + YEAR_MIN)
    print(guessed_year)

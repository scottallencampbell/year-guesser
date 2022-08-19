import json
from pickletools import optimize
import numpy as np
from sklearn.model_selection import train_test_split
import keras.api._v2.keras as keras
import matplotlib.pyplot as plt
from operator import itemgetter

INPUT_FILE = './data/output.json'
MODEL_FILE = './models/1.model'
YEAR_MIN = 1960
YEAR_MAX = 2020

def load_data():
    with open(INPUT_FILE, "r") as fp:
        data = json.load(fp)

    zeros = np.where(np.array(data["labels"]).astype(int) < YEAR_MIN)[0][::-1]
    
    for idx in zeros:
        data["labels"].pop(idx)
        data["mfccs"].pop(idx)

    inputs = np.array(data["mfccs"])
    labels = np.array(data["labels"])
        
    #targets = 1 - ((YEAR_MAX - labels.astype(int)) / (YEAR_MAX - YEAR_MIN))
    targets = labels.astype(int) - YEAR_MIN

    return inputs, targets

def prepare_datasets(test_size, validation_size):
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    # why???
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

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

def build_model(input_shape):
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(YEAR_MAX - YEAR_MIN, activation='softmax'))
    #model.add(keras.layers.Dense(1, activation='linear'))

    return model

def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

if __name__ == "__main__":
    # get train, validation, test splits
    
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(YEAR_MAX - YEAR_MIN, activation="softmax")
        ])
    """
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(YEAR_MAX - YEAR_MIN, activation="softmax")
        ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), batch_size=32, epochs=50)
    plot_history(history)
    
    test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print(test_accuracy)
    """

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, 
        validation_data=(X_validation, y_validation), 
        batch_size=32, 
        epochs=30)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save(MODEL_FILE)
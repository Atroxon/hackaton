import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 400
IMG_HEIGHT = 300
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = list()
    labels = list()

    dim = (IMG_WIDTH, IMG_HEIGHT)

    # Get the CWD and append the input folder, resulting in the full path; regardless of the OS
    os_data_dir = os.path.join(os.getcwd(), data_dir)

    for (root, dirs, files) in os.walk(os_data_dir, topdown=True):
        
        # Avoid processing on main data_dir
        if root == os_data_dir:
            continue
 
        label = os.path.split(root)[1] # Returns tuple with head and tail (tail = label)
        print(label)
        for f in files:
            img = cv2.imread(os.path.join(root, f), 1) # 1 = RGB
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            images.append(resized)
            labels.append(int(label))

    print("Images included in output:", str(len(images)))

    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    PERSONAL REFERENCE (atroxon): 
    https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
    """

    # Classic CNN architecture Conv->Conv->Pool->Conv->Pool->Fully connected
    model = tf.keras.models.Sequential([

        # Convolutional layer. Inputs are 30x30 RGB images with 3 channels. Filters with 3 dimensions.
        # "ReLU layers work far better because the network is able to train a lot faster.
        # It also helps to alleviate the vanishing gradient problem" Deshpande, A (2016)
        tf.keras.layers.Conv2D(
            16, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)#, padding='same'
        ),

        tf.keras.layers.Conv2D(
            32, (2, 2), activation="relu"#, padding='same'
        ),

        # Max-pooling layer, using 2x2 pool size
        # Previous layer passes one 2D array per neuron
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            64, (2, 2), activation="relu"
        ),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout "`Dense` implements the operation: `output = activation(dot(input, kernel) + bias)`"
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add output layer with outputs = NUM_CATEGORIES
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]

    )

    return model

if __name__ == "__main__":
    main()
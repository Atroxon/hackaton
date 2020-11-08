import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Total number of categories in your data
NUM_CATEGORIES=2
#Resize of your image
IMG_WIDTH = 400
IMG_HEIGHT = 400
#Identifier for your data i.e: if you want to give a more descriptive name 
class_names = ['Tanque', 'Micro']
#Test percentage to divide into train and validation
TEST_SIZE=0.5


def main():
    model = tf.keras.models.load_model('model.h5')
    images, labels = load_data("database\\validation")
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    images_test, images_train, labels_test, labels_train = train_test_split(np.array(images), np.array(labels), test_size=TEST_SIZE)
    #Predict all images in images_test
    predictions = model.predict(images_test)

    #print('La imagen predecida es: ',np.argmax(predictions[0]))
    #print('La imagen real es: ',np.argmax(labels_test[0]))

    #Get just one image from the predicions and plot its results
    #i = 4
    #plt.figure(figsize=(6,3))
    #plt.subplot(1,2,1)
    #plot_image(i, predictions[i], labels_test, images_test)
    #plt.subplot(1,2,2)
    #plot_value_array(i, predictions[i],  labels_test)
    #plt.show()

    # Plot X test images, their predicted labels, and the true labels.
    # Color correct predictions in Blue and incorrect predictions in Red, Low predictions Gray.
    #Quantity of images to show
    num_rows = 5
    num_cols = 5
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], labels_test, images_test)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], labels_test)
        plt.tight_layout()
    plt.show()

#Functions Load Data
def load_data(data_dir):
    #Initialize list to store images and labels
    images = []
    labels = []
    #Get the path of the data
    filepath = os.path.abspath(data_dir)
    #Iterate through all folder categories
    for i in range(NUM_CATEGORIES):
      #Join the path of the data with the exact category folder to iterate
      #Change to that NEW path
        os.chdir(os.path.join(filepath, str(i)))
        #Iterate through all the images inside that category
        for image in os.listdir(os.getcwd()):
            #Read that image as an array
            img = cv2.imread(image)
            #If it has data
            if img.size != 0:
                #Resize it accordingly
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            #Append information of image and category
            images.append(img)
            labels.append(i)
    #Change path to folder to sotre the model on root
    os.chdir(filepath)
    return (images, labels)

#Plot images predicted
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False) #Turn off grid
    plt.xticks(range(NUM_CATEGORIES)) #X label with 43 ticks as we have 43 possible categories
    plt.yticks([])  #Without y ticks
    thisplot = plt.bar(range(NUM_CATEGORIES), predictions_array, color="#777777") #Plot the bar
    plt.ylim([0, 1])  #Y label limit from 0 to 1 i.e: 0 no match, 1 fully identified
    predicted_label = np.argmax(predictions_array)  #Get the index of the max value
    label = np.argmax(true_label) #Get the index of the max value
    #MAtch the values accordingly if it is a value inside predicted RED, label BLUE
    thisplot[predicted_label].set_color('red')
    thisplot[label].set_color('blue')

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False) #Trun off grids
  plt.xticks([])  #Without axis
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)
  #Store de index of the max value i.e: the probable image
  predicted_label = np.argmax(predictions_array) 
  #Store de index of the max label i.e: the correct image
  label = np.argmax(true_label) 
  #If the image was identified correctly
  if predicted_label == label:
    #Plot it blue
    color = 'blue'
    #If not plot it red
  else:
    color = 'red'
  #The label for the image: The predicted image, the percentage of accuracy, the ID.
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[label]),
                                color=color)

if __name__ == "__main__":
    main()
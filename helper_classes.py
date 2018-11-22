import time
import math
import random
import pandas as pd 
import numpy as np 
import tensorflow as tf
import os, cv2
from datetime import timedelta
import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('tf')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
import itertools

from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage import transform
from skimage.filters.rank import enhance_contrast
from skimage import data, img_as_float
from skimage import exposure
from skimage import img_as_ubyte
plt.rcParams['font.size'] = 10

num_channels = 3
img_size = 128
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)

from keras.models import model_from_json
from keras.models import load_model

early_stopping = None

checkpoint_dir = 'Models/'


def save_model(model_name, model):
	# Saving and loading model and weights 
	# serialize model to JSON
	model_json = model.to_json()
	with open(checkpoint_dir+model_name+'.json', "w") as json_file: #model.json
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(checkpoint_dir+model_name+'.h5')
	print("Saved model to disk")


def load_model(model_name):
	# load json and create model - DON'T RUN, ONLY FOR LOADING
	json_file = open(checkpoint_dir+model_name+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(checkpoint_dir+model_name+'.h5')
	print("Loaded model from disk")
	return loaded_model


def save_image(path_and_type, image_array):
	cv2.imwrite(path_and_type, image_array)


#Plot images in a 3x3 Grid with 9 total images. There is a True and predicted Label at the bottom
def plot_images(images, cls_true, cls_pred=None):
    """
	Input-
	images: actual images List
	cls_true: Class label of Images List
	cls_pred: Optional if predicted class exists
    """
    if len(images) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 9))
        
        
    images, cls_true  = zip(*[(images[i], cls_true[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots in a single Notebook cell.
    plt.show()


#Helper function for plotting Example Errors
def plot_example_errors(cls_pred, correct):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def convert_multilabel_to_binary(cnf):
	#Input: Confusion Matrix
	correct = [[0,0], [0,0]]
            
	for i in range(5):
	    for j in range(5):
	        if(i in [0, 1]):
	            if(j not in [0,1]):
	                correct[0][1] += cnf[i][j]
	            else:
	                correct[0][0] += cnf[i][j]
	        else:
	            if(j not in [2,3,4]):
	                correct[1][0] += cnf[i][j]
	            else:
	                correct[1][1] += cnf[i][j]

	correct = (np.array(correct))

	np.set_printoptions(precision=2)

	plt.figure()
	# Plot non-normalized confusion matrix
	plot_confusion_matrix(cnf, classes=['class 0 (Closed)', 'class 1 (Open)'], 
		title='Confusion matrix Binary Classification')


def plot_roc_curve(clf, X_test, y_test, no_classes):
	# Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_score = model.predict_proba(X_test)

    for i in range(no_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(no_classes):
        plt.plot(fpr[i],tpr[i],label=str(i)+" auc="+str(roc_auc[i]))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(loc=4)
    plot.show()

def get_confusion_matrix(y_test, y_pred):
	print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


 #Helper function for showing the performance
def print_validation_accuracy(show_example_errors=False, show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
    # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)


        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images, y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


def preprocess_images(path, flag=0):
    if(flag==0): #Rescale the Image Only
        img1 = cv2.imread(path)[320:850, :800]
        return img1
    elif(flag==1): #Rescale and resize the image
        img1 = cv2.imread(path)
        return cv2.resize(img1[320:850, :800], (128, 128))
    else:
        img1=cv2.imread(path)
        img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        p2, p98 = np.percentile(img1, (82, 98))
        img_rescale = exposure.rescale_intensity(img1, in_range=(p2, p98))
        img2 = rgb2gray(img_rescale[320:850, :800])
        #img3 = enhance_contrast(noisy_image, disk(3))

        input_img = cv2.resize(img2,(128,128))
        return input_img

def keras_visualize_train_validation_accuracy(hist, num_epoch):
	train_acc=hist.history['acc']
	val_acc=hist.history['val_acc']
	xc=range(num_epoch)

	plt.figure(figsize=(7,5))
	plt.plot(xc,train_acc, marker='o')
	plt.plot(xc,val_acc, marker='s')
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train','val'])
	plt.show()

def keras_test_image_preprocess(test_image, flag=0):
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    test_image1 = change_img_size(test_image)
    return test_image1

def print_classification_report(y_test, y_pred, target_names):
	print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
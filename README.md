# bounding-box-estimator

Final project for SCC5830/MAI5021
Luiz Giordani

## Table of contents
  - [Abstract](#abstract)
  - [Objective](#objective)
  - [Data source](#data-source)
  - [First results](#first-results)

## Abstract

It is understandable to assume that when working on image processing tasks, and more specifically, image classification, object detection and image detection, large amounts of labeled images are necessary to build a reasonable classifier.

However producing such amount of labeled data is often expensive in both time and money, because it requires a substantial set of images to be obtained which will be labeled by a crowdsourced team, these that requires a great amount of instructions, and finally use the images on a task.

The problem is that different labeling tasks can be ambiguous, and different labelers can come up with different conclusions. And as the user of the dataset there isn't usually an easy way to assess the quality of the labeling.

Therefore, this project aims to follow [data centric AI](https://datacentricai.org/labeling-and-crowdsourcing/) principles. One which is to make better use of the data at hand rather than collecting more. One way this principle could be applied is by creating a classifier that can help check the consistency of a set of images that will be used on a downstream task. More specifically, the idea is to use outlier detection alogrithms with the objective of detecting objects in an image. In this case, we are considering anything that constrasts from the background as a potential object of interest.



For such, some processing tasks such as creating a bag of features from the image to feed the model, and if applicable, apply different filters to help produce more consistent images

Finally, two outputs are expected: A heatmap of the image where higher intensities will denote places with relevant objects. If the first task is successfull, we can move to the second which consists of defining a heuristic to join segments of the image that are considered outliers to get an approximate bounding box, which we can later compare with the original bounding box coordinates of the image.

## Objective

In summary, the task consists of deciding where in the image there are objects. However to perform the task we won't use any annonations or information about the objects. Only the information about how they stand out in the image by their differences when compared to the background.

Therefore, it seems reasonable to assume that we can use the complete image information to perform such task, and we should also assume that we do not know where the objects are. Therefore, one logical approach is to split each imagine in equal sized subsets (patches) and apply a machine learning algorithm to detect whether that patch contains an object.

To perform such task, the following steps must be completed:
1. Image preprocessing
    
    Here, it can be several steps, such as applying a filter to reduce noise or enhance the image, changing its scale, or changing from color to grayscale. However, most importantly is defining the regions to which apply the algorithm which will be non-overlapping patches of fixed size that cover the whole image.

2. Creating feature descriptors for the image.

    Since we do not intend to use algorithms that can take the full image as an input, some mode of processing will need to be applied to extract some features from the image. More specifically, we will prioritize, in order:
    
    a. Intensity Histograms
    
    b. Haralick Descriptors
    
    c. PCA     

3. Fitting the outlier detection algorithm and evaluating its results:

    Here we will train a machine learning algorithm on a set of patches of vartying images to hopefully create a model that can generalize to our whole collection. In terms of algorithm we will explore classical outlier detection models such as Isolation Forest and One Class SVM. Preference will be given to the first one given its robustness and simplicity.

4. Estimating the bounding boxes:

    If the previous task is successful, we can try to come up with an heuristic to estimate the location of the objects in the image. 

    Since the model will give scores on how likely a patch is considered an outlier, we can create masks base on the differentiation between a patch, and its surroundings, and define a threshold to check if the neighbour patch belongs to the same object of if it is different.

In summary, the process consists of feeding some images to an outlier detection algorithm with the hope that any object in the image is marked as an outlier.

## Data source

We will try this task on the [Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset). 

It is a dataset publicy available on Kaggle and for this project, specifically, we will use the validation set that fit our purpose and contains the appropriate annotations.

The images are basically grocery products scattered around a white background. There are also levels of difficulty: easy, medium and hard which relates to the amount of objects on each image.

This dataset is interesting for this project because of its consistent white background that makes hard for an object to blend in it.

Some examples taken from the source (There are more on the notebooks)

![example](/.img/image_source.png)

## First results

A first test was performed using:
* **Image size:** 1024x1024
* **Patch size:** 64x64
* **Feature descriptor:** Graylevel intensity histograms (255 bins)
* **Outlier detection algorithm:** Isolation Forest

More details can be found on the notebooks where we split a [random samples of images](01_take_samples.ipynb), [resize and create a grayscale copy of the images](02_preprocess.ipynb) and [fit the first algorithm](03_algorithm.ipynb)

Below are some examples on a set of 20 images not used for training the algorithm (Higher values denote higher chance of being an outlier):
![example_results_1](/.img/results_gray_1.png)

![example_results_2](/.img/results_gray_2.png)

The model doesn't seem to perform very well and to some point, it seems that it is learning some object placement pattern, not the changes in intensities. To remediate this, new attempts will:

1. Use the color channels to add information -> The grayscale might be helping to conceal some objects

2. Change the number of bins on the intensity histogram to better encode the information

3. Maybe try to fit a model for each image. i.e fit a model specific to an image to try and discover its objects. This should be reasonably fast as long as we keep the total number of patches fairly low.
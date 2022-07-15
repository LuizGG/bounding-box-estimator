# bounding-box-estimator

Final project for SCC5830/MAI5021
Luiz Giordani

## Table of contents
  - [Abstract](#abstract)
  - [Objective](#objective)
  - [Data source](#data-source)
  - [Results](#results)

## Abstract

It is understandable to assume that when working on image processing tasks, and more specifically, image classification, object detection and image detection, large amounts of labeled images are necessary to build a reasonable classifier.

However producing such amount of labeled data is often expensive in both time and money, because it requires a substantial set of images to be obtained which will be labeled by a crowdsourced team. These teams often requires a great amount of instructions and guidance to produce quality annotations to be used on the final task.

The problem is that different labeling tasks can be ambiguous, and different labelers can come up with different conclusions. And as the user of the dataset there isn't usually an easy way to assess the quality of the labeling.

Therefore, this project aims to follow [data centric AI](https://datacentricai.org/labeling-and-crowdsourcing/) principles. One which is to make better use of the data at hand rather than collecting more. One way this principle could be applied is by creating a classifier that can help check the consistency of a set of images that will be used on a downstream task. 

More specifically, the idea is to use outlier detection alogrithms with the objective of detecting objects in an image. In this case, we are considering anything that constrasts from the background as a potential object of interest.

For such, some processing tasks need to be performed such as creating a bag of features from the image to feed the model, and if applicable, apply different filters to help produce more consistent images.

Finally, two outputs are expected: A heatmap of the image where higher intensities will denote places with relevant objects. If the first task is successfull, we can move to the second which consists of defining a heuristic to join segments of the image that are considered outliers to get an approximate bounding box. This which can later be compared with the original bounding box coordinates of the image.

## Objective

The task consists of deciding where in the image there are objects. However to perform this task we won't use any annonations or information about the objects. Only the information about how they stand out in the image by their differences when compared to the background. Hence, by not knowing where in the image the objects are, we resort to the task of scanning the whole image.

Outlier detections algorithms work by comparing anomalies (aka outliers) with the norm. In this project we will consider the background as the norm and the objects as outliers. The task of the algorithm is to find a way to isolate one from the other and label each instance to its correct group.

Thus, we have the task of searching for objects on the image and a method that will give for an instance a label, of whether that instance is an outlier or not. Since we have to scan the whole image, we can produce patches (or blocks, subsets) of the image as inputs for the algorithm, which is tasked of detecting if on that patch there is an object or not. The blocks that will be produces will be non-overlapping ones.

To complete our objects, the following steps must be completed:
1. Image preprocessing
    
    Here, it can be several steps, such as applying a filter to reduce noise or enhance the image, changing its scale, or changing from color to grayscale. Moreover, in this step we will also create the non-overlapping patches of fixed size that cover the whole image.

2. Creating feature descriptors for the image.

    Since we do not intend to use algorithms that can take the full image as an input, some mode of processing will need to be applied to extract some features from the patches. More specifically, we will prioritize, in order:
    
    a. Intensity Histograms (Implemented)
    
    b. Haralick Descriptors (Not done)
    
    c. PCA (Not done)

3. Fitting the outlier detection algorithm and evaluating its results:

    Here we will train a machine learning algorithm on a set of patches of varying images to hopefully create a model that can generalize to our whole collection. In terms of algorithms we will explore classical outlier detection models such as Isolation Forest and One Class SVM. Preference will be given to the first one given its robustness and simplicity.

4. Estimating the bounding boxes:

    If the previous task is successful, we can try to come up with an heuristic to estimate the location of the objects in the image. 

    Since the model will give scores on how likely a patch is considered an outlier, we can apply classical image segmentation methods to link predictions that are similar. Hopefully, segmentating the objects from the background.

In summary, the process consists of feeding some images to an outlier detection algorithm with the hope that any object in the image is marked as an outlier.

## Data source

We will try this task on the [Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset). 

It is a dataset publicy available on Kaggle and for this project we will use the validation, which set that fits our purpose and contains the appropriate annotations for comparison.

The images are basically grocery products scattered around a white background. There are also levels of difficulty: easy, medium and hard which relates to the amount of objects on each image.

This dataset is interesting for this project because of its consistent white background that makes hard for an object to blend in it.

Some examples taken from the source (There are more on the notebooks)

![example](/.img/image_source.png)

## Results

First, a sample of the outcomes is shown comparing two different configurations of the isolation forest.
* **gray_255**: Isolation forest fitted on grayscale images using an intensity histogram with 255 bins.
* **color_16**: Isolation forest fitted on color images using the concatenated intensity histogram for each channel, each histogram has 16 bins.

Both configurations were adjusted using images of size 1024x1024 and a patch size of 64x64. Below are some examples of the distribution of outlier scores on some images (brighter is equal to more likely an outlier).

![sample_results_1](/.img/results_1.png)

![sample_results_2](/.img/results_2.png)

Later, we apply the Otsu segmentation algorithm and pixel aggregation on the results from the **color_16** model to create the bounding boxes. The model and heuristic are capable of finding objects, albeit, only a cluster of them. 

Still the model is much better than the naive approach, of applying the segmentation on the original, grayscale, images.

![sample_results_3](/.img/results_3.png)

![sample_results_4](/.img/results_4.png)

For more details look at the notebooks where we split a [random samples of images](01_take_samples.ipynb), [resize and create a grayscale copy of the images](02_preprocess.ipynb) and [fit the algorithms](03_algorithm.ipynb). A more concise overview is available on [demo_presentation.ipynb](demo_presentation.ipynb).
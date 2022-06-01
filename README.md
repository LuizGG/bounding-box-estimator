# bounding-box-estimator

Final project for SCC5830/MAI5021
Luiz Giordani

## Abstract

It is understandable to assume that when working on image processing tasks, and more specifically, image classification, object detection and image detection, large amounts of labeled images are necessary to build a reasonable classifier.

However producing such amount of labeled data is often expensive in both time and money, because it requires a substantial set of images to be obtained which will be labeled by a crowdsourced team, these that requires a great amount of instructions, and finally use the images on a task.

The problem is that different labeling tasks can be ambiguous, and different labelers can come up with different conclusions. And as the user of the dataset there isn't usually an easy way to assess the quality of the labeling.

Therefore, this project aims to follow [data centric AI](https://datacentricai.org/labeling-and-crowdsourcing/) principles, one which is to make better use of the data at hand rather than collect more. One way this principle could be applied is by creating a classifier that can help check the consistency of a set of images that will be used on a downstream task. More specifically, the idea is to use outlier detection alogrithms with the objective of detecting objects in an image. In this case, we are considering anything that constrasts from the background as a potential object of interest.

For such, some processing tasks such as creating a bag of features from the image to feed the model, and if applicable, apply different filters to help produce more consistent images

Finally, two outputs are expected: A heatmap of the image where higher intensities will denote places with relevant objects and also the join of the different segments of the image to get an approximate bounding box, which we can later compare with the original bounding box coordinates of the image (if available).

## Data sources

Possibly explore how the presented solution works on datasets that have more homogeneous (1) and heterogeneous (2) backgrounds

1. [Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset) -> For this, consider only the validation and test datasets, which fits the task

![example](/.img/ex_1.jpg)

2. [CGI Planes in Satellite Imagery w/ BBoxes](https://www.kaggle.com/datasets/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes)

![example](/.img/ex_2.png)
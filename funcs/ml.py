from itertools import tee
import numpy as np
import imageio
from skimage.util import view_as_blocks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

## Reading data
def read_images(images_path, color=False):
    
    n = len(images_path)
    if color:
        imgs_array = np.zeros((n, 1024, 1024, 3))

        for i in range(n):

            img = imageio.imread(images_path[i])
            imgs_array[i, :, :, :] = img
    else:    
        imgs_array = np.zeros((n, 1024, 1024))

        for i in range(n):

            img = imageio.imread(images_path[i])
            imgs_array[i, :, :] = img

    return imgs_array


def images_as_patches(imgs_array, patch_size):

    width = patch_size[1]
    channels = imgs_array.shape[-1] < 4
    # We flatten the patches since we will take the histogram and the matrix format isn't necessary
    if channels:
        img_patches = view_as_blocks(imgs_array, patch_size).copy().reshape(-1, width**2, 3)
    else: 
        img_patches = view_as_blocks(imgs_array, patch_size).copy().reshape(-1, width**2)    

    return img_patches

## Outlier detection
class IntensityHistogram(TransformerMixin, BaseEstimator):
    """ 
    Sklearn compatible method to compute the intensity histogram. It returns the normalized histogram.
    (i.e the frequencies divided by their sum).

    Parameters
    ----------
    n_bins : int, default=255
        The number of bins on the histogram.
    """

    def __init__(self, nbins=255):
        self.nbins = nbins

    def fit(self, X, y=None):

        return self.fit_transform(X)

    def transform(self, X):
       
        return self.fit_transform(X)
    
    def fit_transform(self, X, y=None):

        matrix_shape = X.shape

        if len(matrix_shape) < 3:
            return self.histogram_gray(X)
        else:
            return self.histogram_color(X)

    def histogram_gray(self, X):

        n = X.shape[0]
        new_X = np.zeros((n, self.nbins))

        for i in range(n):
            bins, _ = np.histogram(X[i, :], bins=self.nbins, range=(0, 255))
            bins = bins / bins.sum()

            new_X[i, :] = bins

        return new_X

    def histogram_color(self, X):

        n = X.shape[0]
        new_X = np.zeros((n, self.nbins*3))

        for i in range(n):
            red_bins, _ = np.histogram(X[i, :, 0], bins=self.nbins, range=(0, 255))
            green_bins, _ = np.histogram(X[i, :, 1], bins=self.nbins, range=(0, 255))
            blue_bins, _ = np.histogram(X[i, :, 2], bins=self.nbins, range=(0, 255))
            
            red_bins = red_bins / red_bins.sum()
            green_bins = green_bins / green_bins.sum()
            blue_bins = blue_bins / blue_bins.sum()

            new_X[i, :] = np.concatenate((red_bins, green_bins, blue_bins))

        return new_X

def train_isoforest(X, nbins, ntrees=100, max_samples='auto', seed=1):

    pipeline = Pipeline([('feature_descriptor', IntensityHistogram(nbins=nbins)), # Method to compute the intensity Histogram
                    ('clf', IsolationForest(n_estimators=ntrees, max_samples=max_samples, n_jobs=-1, random_state=seed))])

    pipeline.fit(X)

    return pipeline

def eval_model(X, pipeline):

    # We are getting the outliers as defined by the metric. Which is not ideal, but it is useful
    predicted_outliers = pipeline.predict(X)

    transformed_x = pipeline.steps[0][1].fit_transform(X) # Accessing the histogram method

    # Getting the silhouette score which is a metric to evaluate the homogeneity of clusters
    # While it is not measured on our ideal clusters, i.e the objects. A good anomaly detection system will split into groups that are very dissimilar
    score = silhouette_score(transformed_x, predicted_outliers)
    
    return score


## Plotting results
def prediction_intensities(model, image, patch_size):
    """
    Takes the scores from the model and creates an array of the same shape as the images to use as an overlay of intensities.
    """

    img_patches = images_as_patches(image, patch_size)
    n = img_patches.shape[0]
    width = patch_size[0]

    scores = model.score_samples(img_patches)*-1 # To make the interpretation easier, I want the value of the abnormal observations to be higher

    scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1))

    scores_grid = np.repeat(scores, width**2, axis=1).reshape(n, width, width)

    scores_image_like = reconstruct_image(scores_grid, image, patch_size)

    return scores_image_like

def reconstruct_image(image_patches, image, patch_size):

    n = image_patches.shape[0]
    img_size = image.shape[0]
    width = patch_size[0]

    step_size = int(img_size/width)
    idx = pairwise(np.arange(n+step_size, step=step_size))

    stacks = []

    for start, end in idx:
        stacks.append(np.hstack(image_patches[start:end, :, :]))

    reconstructed_image = np.vstack(stacks)

    return reconstructed_image


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    # Taken from https://docs.python.org/3/library/itertools.html#itertools.pairwise -> It is a part of the itertools package as of 3.10
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
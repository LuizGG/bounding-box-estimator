from itertools import tee
import numpy as np
import imageio
from skimage.util import view_as_blocks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

def read_images(images_path):
    
    n = len(images_path)
    imgs_array = np.zeros((n, 1024, 1024))

    for i in range(n):

        img = imageio.imread(images_path[i])
        imgs_array[i, :, :] = img

    return imgs_array


def images_as_patches(imgs_array, patch_size):

    # TODO: Check for the case when there are colors (3 channesl)
    width = patch_size[1]
     # We flatten the patches since we will take the histogram and the matrix format isn't necessary
    img_patches = view_as_blocks(imgs_array, patch_size).copy().reshape(-1, width**2)

    return img_patches

def prediction_intensities(model, image, patch_size):
    """
    Takes the scores from the model and creates an array of the same shape as the images to use as an overlay of intensities.
    """

    img_patches = images_as_patches(image, patch_size)
    n = img_patches.shape[0]
    width = patch_size[0]

    scores = model.score_samples(img_patches)*-1 # To make the interpretatione easier, I want the value of the abnormal observations to be higher

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

        # TODO: Implement for 3 channels

        n = X.shape[0]
        new_X = np.zeros((n, self.nbins))

        for i in range(n): # This might not scale with more images/channels/patches
            bins, _ = np.histogram(X[i, :], bins=self.nbins, range=(0, 255))
            bins = bins / bins.sum()

            new_X[i, :] = bins

        return new_X

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    # Taken from https://docs.python.org/3/library/itertools.html#itertools.pairwise -> It is a part of the itertools package as of 3.10
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
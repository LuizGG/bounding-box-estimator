import pandas as pd
import numpy as np
import imageio
from skimage.transform import resize
from skimage.color import rgb2gray

def read_data():

    train_val_images = pd.read_pickle('data/retail_product/train_val_images.pkl')
    train_val_annotations = pd.read_pickle('data/retail_product/train_val_annotations.pkl')
    test_images = pd.read_pickle('data/retail_product/test_images.pkl')
    test_annotations = pd.read_pickle('data/retail_product/test_annotations.pkl')

    return train_val_images, train_val_annotations, test_images, test_annotations

def resize_images(df_images, size): # Assumes image sizes are regular

    new_df_images = df_images.copy()

    base_path_name = 'data/retail_product/sampled_images/{id}_{source}.jpg'

    new_df_images['new_file_name'] = 'data/retail_product/sampled_images/' + new_df_images['id'].astype(str) + '_' + new_df_images['source'] + '.jpg'

    for i, values in new_df_images[['file_name', 'new_file_name', 'id', 'source']].iterrows():

        img = imageio.imread(values['file_name'])

        new_img = resize(img, (size, size), preserve_range=True).astype(np.uint8)

        new_img_path = base_path_name.format(id=values['id'], source=values['source'])

        imageio.imwrite(values['new_file_name'], new_img)

    # new_df_images['file_name'] = 'data/retail_product/sampled_images/' + new_df_images['id'].astype(str) + '_' + new_df_images['source'] + '.jpg'
    new_df_images['new_width'] = size
    new_df_images['new_height'] = size
    new_df_images['width_ratio'] = size/new_df_images['width']
    new_df_images['height_ratio'] = size/new_df_images['height']

    new_df_images.drop(columns=['file_name', 'width', 'height'], inplace=True)
    new_df_images.rename(columns={'new_file_name': 'file_name', 'new_width': 'width', 'new_height': 'height'}, inplace=True)

    new_df_images['n_pixels'] = size**2

    return new_df_images

def to_gray(df_images):

    new_df_images = df_images.copy()
    new_df_images['new_file_name'] = 'data/retail_product/sampled_images/gray_' + new_df_images['id'].astype(str) + '_' + new_df_images['source'] + '.jpg'

    for i, values in new_df_images[['file_name', 'new_file_name', 'id', 'source']].iterrows():

         img = imageio.imread(values['file_name'])

         gray_img = rgb2gray(img).astype(np.uint8)

         imageio.imwrite(values['new_file_name'], gray_img)
    
    new_df_images.drop(columns=['file_name'], inplace=True)
    new_df_images.rename(columns={'new_file_name': 'file_name'}, inplace=True)

    return new_df_images
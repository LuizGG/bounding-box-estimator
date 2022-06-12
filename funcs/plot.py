import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

def show_image(image_id, df_images, plot_bboxes=False, df_annotations=None, ax=None):

    img_path = df_images.loc[df_images['id']==image_id, 'file_name'].values[0]

    img = imageio.imread(img_path)

    channels = img.ndim

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(img, cmap='gray' if channels < 3 else None)
    ax.axis('off')

    if plot_bboxes:
        bboxes = df_annotations.loc[df_annotations['image_id']==image_id, 'bbox'].values.tolist()
        for bbox in bboxes:
            patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='black', facecolor='none')

            ax.add_patch(patch)
        
    return ax
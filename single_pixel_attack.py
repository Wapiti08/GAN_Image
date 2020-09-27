import numpy as np

def create_img(original_image, target_idx):

    #
    # TODO: Implement this function!
    # You probably want some prep here...
    #

    # kind of targeted attack here, using single pixel attack

    max_pixels = 1000
    min_, max_ = (0, 255)

    # get the height and width of the image
    h = original_image.shape(0)
    w = original_image.shape(1) 
    pixels = np.random.permutation(h*w)
    pixels = pixels[:max_pixels]

    for i, pixel in enumerate(pixels):
        x = pixel % w
        y = pixel // w
        location = [x, y]
        location.insert(model.channel_axis(), slice(None))
        location.tuple(location)
import numpy as np

import math

from PIL import Image


def histogram_expansion(image_path, min, max):
    # Image openning
    image = Image.open(image_path)

    # Image convert to Array
    img_array = np.asarray(image, np.float64)

    # Image and filter dimensions
    height, width, channel = img_array.shape

    for y in range(height):
        for x in range(width):
            r = img_array.item(y, x, 0)
            g = img_array.item(y, x, 1)
            b = img_array.item(y, x, 2)

            r = round((r - min) * ((255) / (max - min)))
            g = round((g - min) * ((255) / (max - min)))
            b = round((b - min) * ((255) / (max - min)))

            img_array.itemset((y, x, 0), r)
            img_array.itemset((y, x, 1), g)
            img_array.itemset((y, x, 2), b)

    expanded_image = Image.fromarray(np.uint8(img_array))
    return expanded_image


image_file = "sobel_filterDancingInWater.jpg"

expanded_image = histogram_expansion(image_file, 0, 255)
expanded_image.save('resultados/histogram' + image_file)

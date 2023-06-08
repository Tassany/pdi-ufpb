import numpy as np

from PIL import Image


def rgb_to_yiq(image_path):
    # Open image
    image = Image.open(image_path)

    # Convert image to numpy array
    image_array = np.asarray(image, np.float64)

    height, width, channel = image_array.shape

    # Calculate YIQ values
    for ly in range(0, height):
        for lx in range(0, width):

            r = image_array.item(ly, lx, 0)
            g = image_array.item(ly, lx, 1)
            b = image_array.item(ly, lx, 2)

            y = (0.299*r) + (0.587*g) + (0.114*b)

            i = (0.596*r) - (0.274*g) - (0.322*b)

            q = (0.211*r) - (0.523*g) + (0.312*b)

            image_array.itemset((ly, lx, 0), y)
            image_array.itemset((ly, lx, 1), i)
            image_array.itemset((ly, lx, 2), q)

    return image_array


def yiq_to_rgb(image_array):
    height, width, channel = image_array.shape

    # Calculate RGB values
    for ly in range(0, height):
        for lx in range(0, width):

            y = image_array.item(ly, lx, 0)
            i = image_array.item(ly, lx, 1)
            q = image_array.item(ly, lx, 2)

            r = y + (0.956*i) + (0.621*q)

            g = y - (0.272*i) - (0.647*q)

            b = y - (1.105*i) + (1.702*q)

            image_array.itemset((ly, lx, 0), r)
            image_array.itemset((ly, lx, 1), g)
            image_array.itemset((ly, lx, 2), b)

    rgb_image = Image.fromarray(np.uint8(image_array))
    return rgb_image


def negative_rgb(image_path):

    # Convert image to numpy array
    image = np.asarray(Image.open(image_path))
    image_array = image.copy()

    height, width, channel = image_array.shape

    # Calculate YIQ values
    for ly in range(0, height):
        for lx in range(0, width):

            r = image_array.item(ly, lx, 0)
            g = image_array.item(ly, lx, 1)
            b = image_array.item(ly, lx, 2)

            Nr = 255 - r

            Ng = 255 - g

            Nb = 255 - b

            image_array.itemset((ly, lx, 0), Nr)
            image_array.itemset((ly, lx, 1), Ng)
            image_array.itemset((ly, lx, 2), Nb)

    rgb_image = Image.fromarray(image_array)
    return rgb_image


def negative_yiq(image_path):
    # Open image
    image = Image.open(image_path)

    # Convert image to numpy array
    image_array = np.asarray(image, np.float64)

    height, width, channel = image_array.shape

    # Calculate YIQ values
    for ly in range(0, height):
        for lx in range(0, width):

            r = image_array.item(ly, lx, 0)
            g = image_array.item(ly, lx, 1)
            b = image_array.item(ly, lx, 2)

            y = (0.299*r) + (0.587*g) + (0.114*b)
            y = 255 - y

            i = (0.596*r) - (0.274*g) - (0.322*b)

            q = (0.211*r) - (0.523*g) + (0.312*b)

            image_array.itemset((ly, lx, 0), y)
            image_array.itemset((ly, lx, 1), i)
            image_array.itemset((ly, lx, 2), q)

    return image_array


image_file = "DancingInWater.jpg"

# Transform RGB to YIQ
yiq_array = rgb_to_yiq(image_file)

# Convert YIQ to RGB
rgb_image = yiq_to_rgb(yiq_array)

rgb_image.save('resultados/rgb_' + image_file)

# Convert Negative RGB
negative_image_rgb = negative_rgb(image_file)

negative_image_rgb.save('resultados/negative_rgb_' + image_file)

# Convert Negative YIQ

negative_image_yiq_array = negative_yiq(image_file)
negative_y = yiq_to_rgb(negative_image_yiq_array)


negative_y.save('resultados/negative_yiq_' + image_file)

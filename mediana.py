import numpy as np
from PIL import Image


def median_filter(image_path, filter_dimensions):
    # Image openning
    image = Image.open(image_path)

    image = image.resize((1500, 1000))    # Apply a resize for bigger images

    # Image convert to Array
    img_array = np.asarray(image, np.float64)

    # Image to apply the result
    filtered_image = Image.new(image.mode, image.size)

    # Image and filter dimensions
    height, width, channel = img_array.shape
    filter_half_width = filter_dimensions[1] // 2
    filter_half_height = filter_dimensions[0] // 2

    # Go through all the pixels of the image
    for y in range(height):
        for x in range(width):
            # Create a list of neighbors for each color
            r = []
            g = []
            b = []
            # Go through height neighbors
            for i in range(-filter_half_height, filter_half_height+1):
                # Go through widhth neighbors
                for j in range(-filter_half_width, filter_half_width+1):
                    # Get coordinates
                    widthNeighbor = x + j
                    heightNeighbor = y + i

                    # Make zero extension if a neighbor has a negative value or if it's bigger than the image
                    if widthNeighbor < 0 or widthNeighbor >= width or heightNeighbor < 0 or heightNeighbor >= height:
                        r.append(0)
                        g.append(0)
                        b.append(0)
                    # If the pixel is in the image, add the pixel value to the list of neighbors
                    else:
                        r.append(img_array[heightNeighbor, widthNeighbor, 0])
                        g.append(img_array[heightNeighbor, widthNeighbor, 1])
                        b.append(img_array[heightNeighbor, widthNeighbor, 2])

            # Calculate the median for each color
            median_value_r = int(np.median(r))
            median_value_g = int(np.median(g))
            median_value_b = int(np.median(b))

            # Aply the new values to the pixel
            filtered_image.putpixel(
                (x, y), (median_value_r, median_value_g, median_value_b))

    return filtered_image


image_file = "DancingInWater.jpg"


# Apply median filter

m = 5
n = 5
filtered_image = median_filter(image_file, (m, n))
filtered_image.save('resultados/median_filter_' + image_file)

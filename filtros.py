import numpy as np

import math
import time

from PIL import Image


inicio = time.time()


def ler_arquivo(txt_path):
    m = 0
    n = 0
    arquivo = open(txt_path, 'r')
    filter_values = []
    cont = 0
    filter_type = ''
    for linha in arquivo:
        valor = linha.split()
        if cont == 0:
            m = int(valor[0])
            n = int(valor[1])
        elif cont == 1:
            filter_type = valor[0]
        else:
            for item in valor:
                filter_values.append(int(item))
        cont += 1

    print(m, n, filter_values)
    arquivo.close()
    return m, n, filter_type, filter_values


def soma_filter(image_path, filter_size):

    # offset
    offset = 50

    # Image openning
    image = Image.open(image_path)

    image = image.resize((1500, 1000))    # Apply a resize for bigger images

    # Image convert to Array
    img_array = np.asarray(image, np.float64)

    # Image to apply the result
    filtered_image = Image.new(image.mode, image.size)

    # Image and filter dimensions
    height, width, channel = img_array.shape
    filter_half_width = filter_size[1] // 2
    filter_half_height = filter_size[0] // 2

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

            # Calculate the sum for each color
            soma_value_r = int(sum(r) + offset)
            if soma_value_r > 255:
                soma_value_r = 255
            elif soma_value_r < 0:
                soma_value_r = 0

            soma_value_g = int(sum(g) + offset)
            if soma_value_g > 255:
                soma_value_g = 255
            elif soma_value_g < 0:
                soma_value_g = 0

            soma_value_b = int(sum(b) + offset)
            if soma_value_b > 255:
                soma_value_b = 255
            elif soma_value_b < 0:
                soma_value_b = 0

            # Aply the new values to the pixel
            filtered_image.putpixel(
                (x, y), (soma_value_r, soma_value_g, soma_value_b))

    return filtered_image


def box_filter(image_path, filter_size):

    # offset
    offset = 10

    # Value of the filter
    divison = filter_size[0] * filter_size[1]

    # Image openning
    image = Image.open(image_path)

    image = image.resize((1500, 1000))    # Apply a resize for bigger images

    # Image convert to Array
    img_array = np.asarray(image, np.float64)

    # Image to apply the result
    filtered_image = Image.new(image.mode, image.size)

    # Image and filter dimensions
    height, width, channel = img_array.shape
    filter_half_width = filter_size[1] // 2
    filter_half_height = filter_size[0] // 2

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

            # Calculate the average for each color
            average_value_r = int((sum(r)/divison) + offset)
            if average_value_r > 255:
                average_value_r = 255
            elif average_value_r < 0:
                average_value_r = 0

            average_value_g = int((sum(g)/divison) + offset)
            if average_value_g > 255:
                average_value_g = 255
            elif average_value_g < 0:
                average_value_g = 0

            average_value_b = int((sum(b)/divison) + offset)
            if average_value_b > 255:
                average_value_b = 255
            elif average_value_b < 0:
                average_value_b = 0

            # Aply the new values to the pixel
            filtered_image.putpixel(
                (x, y), (average_value_r, average_value_g, average_value_b))
    fim = time.time()
    print(fim - inicio)
    return filtered_image


def sobel_filter(image_path):

    # offset
    offset = 0
    # Load image
    image = Image.open(image_path)

    # image = image.resize((1500, 1000))    # Apply a resize for bigger images

    # Apply Sobel filter
    x_gradient = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_gradient = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convert image to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Image and filter dimensions
    height, width, channel = img_array.shape
    filter_half_width = 3 // 2
    filter_half_height = 3 // 2

    # Create a new image to store the filtered result
    filtered_image = Image.new(image.mode, image.size)

    # Loop through all pixels in the image
    for y in range(height):
        for x in range(width):
            red = np.empty((3, 3))
            green = np.empty((3, 3))
            blue = np.empty((3, 3))
            # Go through height neighbors
            for i in range(-filter_half_height, filter_half_height+1):
                # Go through widhth neighbors
                for j in range(-filter_half_width, filter_half_width+1):
                    # Get coordinates
                    widthNeighbor = x + j
                    heightNeighbor = y + i

                    # Make zero extension if a neighbor has a negative value or if it's bigger than the image
                    if widthNeighbor < 0 or widthNeighbor >= width or heightNeighbor < 0 or heightNeighbor >= height:
                        red[i][j] = 0
                        green[i][j] = 0
                        blue[i][j] = 0
                    # If the pixel is in the image, add the pixel value to the list of neighbors
                    else:
                        red[i][j] = img_array[heightNeighbor, widthNeighbor, 0]
                        green[i][j] = img_array[heightNeighbor, widthNeighbor, 1]
                        blue[i][j] = img_array[heightNeighbor, widthNeighbor, 2]

            # Calculate the gradient in the x and y directions

            gx_r = np.sum(x_gradient * red)
            gx_g = np.sum(x_gradient * green)
            gx_b = np.sum(x_gradient * blue)
            gy_r = np.sum(y_gradient * red)
            gy_g = np.sum(y_gradient * green)
            gy_b = np.sum(y_gradient * blue)

            # Combine the x and y gradients to get the final result
            gradient_magnitude_r = int(np.sqrt(gx_r**2 + gy_r**2)) + offset
            if gradient_magnitude_r > 255:
                gradient_magnitude_r = 255
            elif gradient_magnitude_r < 0:
                gradient_magnitude_r = 0
            gradient_magnitude_g = int(np.sqrt(gx_g**2 + gy_g**2)) + offset
            if gradient_magnitude_g > 255:
                gradient_magnitude_g = 255
            elif gradient_magnitude_g < 0:
                gradient_magnitude_g = 0
            gradient_magnitude_b = int(np.sqrt(gx_b**2 + gy_b**2)) + offset
            if gradient_magnitude_b > 255:
                gradient_magnitude_b = 255
            elif gradient_magnitude_b < 0:
                gradient_magnitude_b = 0

            # Set the pixel value in the filtered image
            filtered_image.putpixel(
                (x, y), (gradient_magnitude_r, gradient_magnitude_g, gradient_magnitude_b))

    return filtered_image


def emboss_filter(image_path, filter_size, filter_values):

    # offset
    offset = 0

    # Image openning
    image = Image.open(image_path)

    image = image.resize((1500, 1000))    # Apply a resize for bigger images

    # Image convert to Array
    img_array = np.asarray(image, np.float64)

    # Image to apply the result
    filtered_image = Image.new(image.mode, image.size)

    # Image and filter dimensions
    height, width, channel = img_array.shape
    filter_half_width = filter_size[1] // 2
    filter_half_height = filter_size[0] // 2

  # Initialize variables to control loop behavior
    cont1 = 0
    cont2 = 0

    # Go through all the pixels of the image
    for y in range(height):
        for x in range(width):
            # Create a list of neighbors for each color
            r = []
            g = []
            b = []
            cont1 = 0
            # Go through height neighbors
            for i in range(-filter_half_height, filter_half_height+1):
                if filter_size[0] % 2 == 0 and i == filter_half_height:
                    if cont1 == 0:
                        cont1 = 1
                        continue
                    else:
                        break
                cont2 = 0
                # Go through width neighbors
                for j in range(-filter_half_width, filter_half_width+1):
                    if filter_size[1] % 2 == 0 and j == filter_half_width:
                        if cont2 == 0:
                            cont2 = 1
                            continue
                        else:
                            break
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

            # Calculate the product of the pixels values with the

            # Calculate the product of the pixels values with the filter_values for each color
            average_value_r = abs((sum(np.multiply(r, filter_values))))
            average_value_g = abs((sum(np.multiply(g, filter_values))))
            average_value_b = abs((sum(np.multiply(b, filter_values))))

            average_value_r = int(average_value_r + offset)
            if average_value_r > 255:
                average_value_r = 255
            elif average_value_r < 0:
                average_value_r = 0

            average_value_g = int(average_value_g + offset)
            if average_value_g > 255:
                average_value_g = 255
            elif average_value_g < 0:
                average_value_g = 0

            average_value_b = int(average_value_b + offset)
            if average_value_b > 255:
                average_value_b = 255
            elif average_value_b < 0:
                average_value_b = 0

            # Aply the new values to the pixel
            filtered_image.putpixel(
                (x, y), (average_value_r, average_value_g, average_value_b))

    return filtered_image


image_file = "DancingInWater.jpg"


# Read input with the filter
m, n, filter_type, filter_values = ler_arquivo("input.txt")


# Apply sum filter

if filter_type == 'soma':

    filtered_image = soma_filter(image_file, (m, n))
    filtered_image.save('resultados/soma_filter' + image_file)

# Apply box filter

if filter_type == 'box':

    filtered_image = box_filter(image_file, (m, n))
    filtered_image.save('resultados/box_filter' + image_file)

# Apply sobel filter
if filter_type == 'sobel':

    filtered_image = sobel_filter(image_file)
    filtered_image.save('resultados/sobel_filter' + image_file)


if filter_type == 'emboss':
    filtered_image = emboss_filter(image_file, (m, n), filter_values)
    filtered_image.save('resultados/emboss_filter' + image_file)

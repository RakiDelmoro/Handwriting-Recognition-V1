import einops
import random
import cv2 as cv
import numpy as np

def visualize_patched_image(image_array):
    grid_iamge = image_array.copy()
    # Draw vertical lines correspond to patch size
    for x in range(0, image_array.shape[-1], 16): cv.line(grid_iamge, (x, 0), (x, image_array.shape[0]), (255, 255, 255), 1)
    return  grid_iamge

def resize_image_array(image_array, rescaled_size):
    border_size = 10
    image_with_border = cv.copyMakeBorder(image_array, border_size+10, border_size+10, border_size+5, border_size+5, cv.BORDER_CONSTANT, value=255)
    resized_image_array = cv.resize(image_with_border, (rescaled_size[1], rescaled_size[0]))
    canvas = np.ones(rescaled_size, dtype=np.uint8) * 255
    canvas[:rescaled_size[0], 0:rescaled_size[1]] = resized_image_array
    return canvas

def image_array_processing(image_array, image_fixed_size, patch_width):
    average_image_pixel = np.average(image_array) - 25
    background_condition = image_array > average_image_pixel
    image_array[background_condition] = 255
    image_array = resize_image_array(image_array, image_fixed_size)

    maximum_pixel_value = image_array.max()
    center = (image_array.shape[1] // 2, image_array.shape[0] // 2)
    angle = random.randint(-1, 1)
    rotate_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image_array = cv.warpAffine(image_array, rotate_matrix, (image_array.shape[1],image_array.shape[0]), borderValue=int(maximum_pixel_value))
    inverted_image_array = cv.bitwise_not(rotated_image_array)
    normalized_image_array = cv.normalize(inverted_image_array, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    cv.imwrite('test.png', visualize_patched_image(inverted_image_array))
    return einops.rearrange(normalized_image_array, 'height (width patch_width) -> (width) height patch_width', patch_width=patch_width)

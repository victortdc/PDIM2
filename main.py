# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import random



def add_noise(img):
    # Getting the dimensions of the image
    col = img.shape[0]
    row = img.shape[1]
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number
    number_of_pixels = random.randint(0, 5000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between
    number_of_pixels = random.randint(0, 5000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread('BrainOriginal.png')
    cv2.imshow('ImgOriginal', img)
    add_noise(img)

    cv2.imshow('ImgNoise', img)
    # Load image
    img_median = cv2.medianBlur(img, 5)  # Add median filter to image

    cv2.imshow('ImgMediana', img_median)  # Display img with median filter
    cv2.waitKey(0)  # Wait for a key press to

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

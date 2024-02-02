import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def write_in_csv(file_name, data):
    with open(file_name, mode="w") as file:
        file_writer = csv.writer(file, delimiter = ";", lineterminator="\r")
        for irow in data:
            file_writer.writerow(irow)

def find_max_intensity(data):
    pix_min = 0
    current_max = pix_min
    
    for irow in data:
        for ipix in irow:
            if current_max < ipix:
                current_max = ipix
    return current_max

def find_medium_spot_intensity(data):
    pix_min = 0
    summ = 0
    count = 0

    for irow in data:
        for ipix in irow:
            if pix_min < ipix:
                count += 1
                summ += ipix

    return summ // count

def generate_noise(shape, mean, stddev):
    noise = np.random.normal(mean, stddev, shape).astype(np.uint8)
    write_in_csv('matrices/noise.csv', noise)
    return noise

def add_noise(data, mean, stddev):
    noise = generate_noise(data.shape, mean, stddev)
    image_with_noise = cv2.add(data, noise)
    return image_with_noise

if __name__ == "__main__":
    gray_image = cv2.imread('pictures/image_1.png', cv2.IMREAD_GRAYSCALE)
    write_in_csv('matrices/image.csv', gray_image)

    intensity_range = find_max_intensity(gray_image)
    medium_spot_intensity = find_medium_spot_intensity(gray_image)

    print('Spot intensity range: ', intensity_range)
    print('Medium spot intensity: ', medium_spot_intensity)

    mean = medium_spot_intensity
    stddev = medium_spot_intensity // 3

    gray_image_with_noise = add_noise(gray_image, mean, stddev)
    write_in_csv('matrices/image_with_noise.csv', gray_image_with_noise)

    cv2.imwrite('pictures/noisy_img.png', gray_image_with_noise)

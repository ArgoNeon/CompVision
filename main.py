import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def write_in_csv(file_name, data):
    with open(file_name, mode="w") as file:
        file_writer = csv.writer(file, delimiter = ";", lineterminator="\r")
        for irow in data:
            file_writer.writerow(irow)

def find_min(data):
    pix_max = 255
    current_min = pix_max
    
    for irow in data:
        for ipix in irow:
            if current_min > ipix:
                current_min = ipix
    return current_min

def find_intensity_range(data):
    pix_max = 255
    data_min = find_min(data)
    return pix_max - data_min

def find_medium_spot_intensity(data):
    pix_max = 255
    summ = 0
    count = 0

    for irow in data:
        for ipix in irow:
            if pix_max > ipix:
                count += 1
                summ += ipix

    return summ // count

if __name__ == "__main__":
    gray_image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    write_in_csv('test.csv', gray_image)

    intensity_range = find_intensity_range(gray_image)
    medium_spot_intensity = find_medium_spot_intensity(gray_image)
    
    print('Spot intensity range: ', intensity_range)
    print('Medium spot intensity: ', medium_spot_intensity)

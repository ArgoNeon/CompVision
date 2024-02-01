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

if __name__ == "__main__":
    gray_image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    write_in_csv('test.csv', gray_image)
    print('Spot intensity range: ', find_intensity_range(gray_image))

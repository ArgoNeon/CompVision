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
    #write_in_csv('matrices/noise.csv', noise)
    return noise

def add_noise(data, mean, stddev):
    noise = generate_noise(data.shape, mean, stddev)
    image_with_noise = cv2.add(data, noise)
    #write_in_csv('matrices/image_with_noise.csv', image_with_noise)
    return image_with_noise

def generate_image_with_noise(image_file, noisy_image_file):
    gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    #write_in_csv('matrices/image.csv', image)

    intensity_range = find_max_intensity(gray_image)
    medium_spot_intensity = find_medium_spot_intensity(gray_image)

    print('Spot intensity range: ', intensity_range)
    print('Medium spot intensity: ', medium_spot_intensity)

    mean = medium_spot_intensity
    stddev = medium_spot_intensity // 3

    gray_image_with_noise = add_noise(gray_image, mean, stddev)
    cv2.imwrite(noisy_image_file, gray_image_with_noise)

def find_contours(image_file):
    image = cv2.imread(image_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    aperture_size = 5
    blur_image = cv2.medianBlur(gray_image, aperture_size)
    #cv2.imwrite('pictures/blur_image.png', blur_image)

    low_threshold   = 7
    high_threshold  = 50
    canny_image = cv2.Canny(blur_image, low_threshold, high_threshold)
    #cv2.imwrite('pictures/canny_image.png', canny_image)

    anchor = (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, anchor)
    closed_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('pictures/closed_image.png', closed_image)

    contours = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours

def draw_elliptic_contours(image_file, image_file_with_contours):
    image = cv2.imread(image_file)
    contours = find_contours(image_file)
    
    for contour in contours:
        contour_perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * contour_perimeter
        contour_approximation = cv2.approxPolyDP(contour, epsilon, True)
        elliptic_contour = cv2.fitEllipse(contour_approximation)
        
        green_color     = (0,255,0)
        thickness       = 1
        cv2.ellipse(image, elliptic_contour, green_color, thickness)
    cv2.imwrite(image_file_with_contours, image)

def draw_contours(image_file, image_file_with_contours):
    image = cv2.imread(image_file)
    contours = find_contours(image_file)
    
    for contour in contours:
        contour_perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * contour_perimeter
        contour_approximation = cv2.approxPolyDP(contour, epsilon, True)

        all_contours    = -1
        green_color     = (0,255,0)
        thickness       = 1
        cv2.drawContours(image, [contour_approximation], all_contours, green_color, thickness)
    cv2.imwrite(image_file_with_contours, image)

if __name__ == "__main__":
    generate_image_with_noise('pictures/image_1.png', 'pictures/noisy_image.png')

    draw_contours('pictures/noisy_image.png', 'pictures/image_with_contours.png')
    draw_elliptic_contours('pictures/noisy_image.png', 'pictures/image_with_elliptic_contours.png')
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
    return medium_spot_intensity, intensity_range 

def find_contours(image_file, low_threshold, high_threshold):
    image = cv2.imread(image_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    aperture_size = 7
    blur_image = cv2.medianBlur(gray_image, aperture_size)
    #cv2.imwrite('pictures/blur_image.png', blur_image)

    canny_image = cv2.Canny(blur_image, low_threshold, high_threshold)
    #cv2.imwrite('pictures/canny_image.png', canny_image)

    anchor = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, anchor)
    closed_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite('pictures/closed_image.png', closed_image)

    contours = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours

def draw_elliptic_contours(image_file, image_file_with_contours, medium_spot_intensity):
    low_threshold   = medium_spot_intensity * 1.0
    high_threshold  = medium_spot_intensity * 2.0

    if high_threshold > 255:
        high_threshold = 255

    image = cv2.imread(image_file)
    contours = find_contours(image_file, low_threshold, high_threshold)
    
    for contour in contours:
        contour_perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * contour_perimeter
        contour_approximation = cv2.approxPolyDP(contour, epsilon, True)

        if (len(contour_approximation) >= 5):
            elliptic_contour_approximation = cv2.fitEllipse(contour_approximation)
        
            green_color     = (0,255,0)
            thickness       = 1
            cv2.ellipse(image, elliptic_contour_approximation, green_color, thickness)
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

    for i in range(10):
        start_image_name        = 'pictures/image_' + str(i) + '.png'
        noisy_image_name        = 'pictures/noisy_image_' + str(i) + '.png'
        contoured_image_name    = 'pictures/image_with_elliptic_contours_' + str(i) + '.png'
        #refined_image_name      = 'pictures/refined_image_with_elliptic_contours_' + str(i) + '.png'

        print('Image name: ', start_image_name)
        medium_spot_intensity, intensity_range = generate_image_with_noise( start_image_name,
                                                                            noisy_image_name)

        #draw_contours('pictures/noisy_image.png', 'pictures/image_with_contours.png')
        draw_elliptic_contours( noisy_image_name,
                                contoured_image_name,
                                medium_spot_intensity)
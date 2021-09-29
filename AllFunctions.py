# This file should contain all the functions required by Wrapper.py

from pathlib import Path

import numpy
import skimage.io as sk
from skimage import exposure
from skimage.filters import threshold_otsu
import time

# All functions which has vijay_function_name implemented to use in problem statements


def compute_hist(image_path: Path, num_bins=255) -> list:
    # bins_vec and freq_vec should contain values computed by custom function
    # bins_vec_lib and freq_vec_lib should contain values computed by python library function

    # ------------------------------------------------------------------------------------------
    # Using own function
    image_read = sk.imread(image_path).astype('float64')
    bins_vec, freq_vec = vijay_hist_comp(image_read, num_bins)
    # ----------------------------------------------------------------------------------------
    # Using Inbuilt function
    [freq_vec_lib, bins_vec_lib] = exposure.histogram(image=image_read, nbins=num_bins)

    return [bins_vec, freq_vec, bins_vec_lib, freq_vec_lib]


def otsu_threshold(gray_image_path: Path) -> list:
    image_read = sk.imread(gray_image_path)
    dummy1, dummy2 = vijay_hist_comp(image_read)
    histogram = [dummy2, dummy1]
    histogram[0] = histogram[0] / sum(histogram[0])
    # ------------------------------------------------------------------------------------------
    # Threshold using Within class variance
    start = time.time()

    threshold_wcv = vijay_within_class_variance(histogram)

    end = time.time()
    within_class_var_time = end - start

    # ------------------------------------------------------------------------------------------
    # Threshold finding using between class variance
    start = time.time()

    threshold_bcv = vijay_between_class_variance(histogram)

    end = time.time()
    between_class_var_time = end - start
    # ------------------------------------------------------------------------------------------
    #  Binary image conversion using vectorization

    bin_image = vijay_binary_conversion(image_read, threshold=threshold_bcv, check_threshold=1)

    return [threshold_wcv, threshold_bcv, within_class_var_time,
            between_class_var_time, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> numpy.ndarray:
    fro_image = sk.imread(quote_image_path)
    back_image = sk.imread(bg_image_path)
    [dummy1, dummy2] = vijay_hist_comp(fro_image)
    histogram = [dummy2, dummy1]
    histogram[0] = histogram[0] / sum(histogram[0])
    # ------------------------------------------------------------------------------------------
    # Converting binary using threshold using vectorization
    threshold = vijay_between_class_variance(histogram)
    binary_image = vijay_binary_conversion(fro_image, threshold=threshold, check_threshold=1)
    # ------------------------------------------------------------------------------------------
    # Adding binary image to background using vectorization
    modified_image = back_image.copy()
    modified_image[binary_image == 0] = 0

    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    image_read = sk.imread(gray_image_path)
    # ------------------------------------------------------------------------------------------
    # Doing connected component analysis to quote
    threshold = threshold_otsu(image_read)
    final_image = vijay_connected_components(image_read, threshold)
    # ------------------------------------------------------------------------------------------
    # finding number of components without punctuations
    histogram = exposure.histogram(final_image)
    freq = [x for x in histogram[0] if x != 0]
    freq.remove(freq[0])
    freq_mean = numpy.mean(freq)
    final_freq = [x for x in freq if x > freq_mean / 5]
    num_characters = len(final_freq)

    return num_characters


def binary_morphology(gray_image_path: Path) -> numpy.ndarray:
    image_read = sk.imread(gray_image_path)
    binary_image = vijay_binary_conversion(image_read)
    # ------------------------------------------------------------------------------------------
    # Noice removal
    dummy3 = numpy.append(binary_image[0:, 1:], numpy.transpose([binary_image[0:, 0]]), axis=1)
    dummy4 = numpy.append(numpy.transpose([binary_image[0:, -1]]), binary_image[0:, 0:-1], axis=1)
    clean_image = binary_image.copy()
    clean_image += dummy3
    clean_image += dummy4
    clean_image += numpy.append(binary_image[1:, 0:], [binary_image[0, 0:]], axis=0)
    clean_image += numpy.append([binary_image[-1, 0:]], binary_image[0:-1, 0:], axis=0)
    clean_image += numpy.append(dummy3[1:, 0:], [dummy3[0, 0:]], axis=0)
    clean_image += numpy.append([dummy3[-1, 0:]], dummy3[0:-1, 0:], axis=0)
    clean_image += numpy.append(dummy4[1:, 0:], [dummy4[0, 0:]], axis=0)
    clean_image += numpy.append([dummy4[-1, 0:]], dummy4[0:-1, 0:], axis=0)
    clean_image = clean_image / 9
    clean_image = vijay_binary_conversion(clean_image, threshold=127.5, check_threshold=1)

    return clean_image


def count_mser_components(gray_image_path: Path) -> list:
    image_read = sk.imread(gray_image_path)
    bin_num = 20
    maximum = numpy.max(image_read)
    minimum = numpy.min(image_read)
    # ------------------------------------------------------------------------------------------
    # Doing connected component analysis to all thresholds
    thresholds = [x * (maximum - minimum) / bin_num for x in range(bin_num)]
    final_image = numpy.zeros(numpy.shape(image_read))
    for threshold in thresholds:
        connected_image = vijay_connected_components(image_read, threshold)
        final_image = final_image + connected_image
    final_image = final_image / bin_num
    mser_image = vijay_connected_components(final_image, 0)
    # ------------------------------------------------------------------------------------------
    # from mser finding number of components
    mser_histogram = exposure.histogram(mser_image)
    mser_freq = [x for x in mser_histogram[0] if x != 0]
    mser_freq.remove(mser_freq[0])
    num_mser_components = len(mser_freq)
    # ------------------------------------------------------------------------------------------
    # from otsu finding number of components
    otsu_image = vijay_connected_components(image_read, threshold_otsu(image_read))
    otsu_histogram = exposure.histogram(otsu_image)
    otsu_freq = [x for x in otsu_histogram[0] if x != 0]
    otsu_freq.remove(otsu_freq[0])
    num_otsu_components = len(otsu_freq)
    # ------------------------------------------------------------------------------------------
    # binarization of both mser and otsu images
    ostu_binary_image = vijay_binary_conversion(otsu_image, threshold=0, check_threshold=1)
    mser_binary_image = vijay_binary_conversion(mser_image, threshold=0, check_threshold=1)

    return [mser_binary_image, ostu_binary_image, num_mser_components, num_otsu_components]


def vijay_within_class_variance(histogram):
    # ------------------------------------------------------------------------------------------
    # finding threshold from within class variance
    within_class_var = 0
    threshold_index_wcv = 0
    temp_within_class_var = 0
    for t in range(0, len(histogram[1])):
        w0 = 0
        w1 = 0
        temp_mean0 = 0
        temp_mean1 = 0
        temp_var0 = 0
        temp_var1 = 0
        for i in range(0, t + 1):
            w0 += histogram[0][i]
            temp_mean0 += histogram[1][i] * histogram[0][i]
            temp_var0 += histogram[1][i] * histogram[1][i] * histogram[0][i]
        for i in range(t + 1, len(histogram[1])):
            w1 += histogram[0][i]
            temp_mean1 += histogram[1][i] * histogram[0][i]
            temp_var1 += histogram[1][i] * histogram[1][i] * histogram[0][i]
        if w0 != 0 and w1 != 0:
            u0 = temp_mean0 / w0
            u1 = temp_mean1 / w1
            var0 = (temp_var0 / w0) - (u0 * u0)
            var1 = (temp_var1 / w1) - (u1 * u1)
            temp_within_class_var = w0 * var0 + w1 * var1
        if t == 1:
            within_class_var = temp_within_class_var
        if temp_within_class_var < within_class_var:
            within_class_var = temp_within_class_var
            threshold_index_wcv = t

    return histogram[1][threshold_index_wcv]


def vijay_between_class_variance(histogram):
    # ------------------------------------------------------------------------------------------
    # finding threshold from between class variance
    threshold_index_bcv = 0
    between_class_var = 0
    temp_between_class_var = 0
    for t in range(0, len(histogram[1])):
        w0 = 0
        w1 = 0
        temp_mean0 = 0
        temp_mean1 = 0
        for i in range(0, t + 1):
            w0 += histogram[0][i]
            temp_mean0 += histogram[1][i] * histogram[0][i]
        for i in range(t + 1, len(histogram[1])):
            w1 += histogram[0][i]
            temp_mean1 += histogram[1][i] * histogram[0][i]
        if w0 != 0 and w1 != 0:
            u0 = temp_mean0 / w0
            u1 = temp_mean1 / w1
            ut = (w0 * u0) + (w1 * u1)
            temp_between_class_var = (w0 * (u0 - ut) * (u0 - ut)) + (w1 * (u1 - ut) * (u1 - ut))
        if t == 1:
            between_class_var = temp_between_class_var
        if between_class_var < temp_between_class_var:
            between_class_var = temp_between_class_var
            threshold_index_bcv = t

    return histogram[1][threshold_index_bcv]


def vijay_connected_components(input_image, input_threshold):
    # ------------------------------------------------------------------------------------------
    # Doing connected component analysis to given image and returning image with each component has unique intensities
    binary_image = vijay_binary_conversion(input_image, threshold=input_threshold, check_threshold=1)
    binary_histogram = exposure.histogram(binary_image)
    if binary_histogram[0][0] > binary_histogram[0][-1]:
        binary_image = 255-binary_image
    size = numpy.shape(binary_image)
    k = 0
    region_array = numpy.zeros(size).astype('int')
    region_array[0][0] = k
    equivalent_elements = []
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if binary_image[i][j] == 0:
                if i == 0 and j != 0:
                    if binary_image[i][j] == binary_image[i][j - 1]:
                        region_array[i][j] = region_array[i][j - 1]
                    else:
                        k += 1
                        region_array[i][j] = k
                if i != 0 and j == 0:
                    if binary_image[i][j] == binary_image[i - 1][j]:
                        region_array[i][j] = region_array[i - 1][j]
                    else:
                        k += 1
                        region_array[i][j] = k
                if i != 0 and j != 0:
                    if binary_image[i][j] == binary_image[i - 1][j] and binary_image[i][j] != binary_image[i][j - 1]:
                        region_array[i][j] = region_array[i - 1][j]
                    elif binary_image[i][j] != binary_image[i - 1][j] and binary_image[i][j] == binary_image[i][j - 1]:
                        region_array[i][j] = region_array[i][j - 1]
                    elif binary_image[i][j] == binary_image[i - 1][j] and binary_image[i][j] == binary_image[i][j - 1]:
                        region_array[i][j] = region_array[i - 1][j]
                        if region_array[i - 1][j] != region_array[i][j - 1]:
                            temp = [region_array[i - 1][j], region_array[i][j - 1]]
                            temp = [min(temp), max(temp)]
                            flag = 0
                            for item in equivalent_elements:
                                if len(numpy.intersect1d(item, temp)):
                                    flag = 1
                                    equivalent_elements[equivalent_elements.index(item)] = list(numpy.union1d(item, temp))
                            if flag == 0:
                                equivalent_elements.append(temp)
                    else:
                        k += 1
                        region_array[i][j] = k

    for item in equivalent_elements:
        for item1 in equivalent_elements:
            if len(numpy.intersect1d(item, item1)):
                equivalent_elements[equivalent_elements.index(item)] = list(numpy.union1d(item, item1))
                equivalent_elements[equivalent_elements.index(item1)] = list(numpy.union1d(item, item1))

    for item in equivalent_elements:
        if equivalent_elements.count(item) > 1:
            for i in range(1, equivalent_elements.count(item)):
                equivalent_elements.remove(item)

    modified_image = numpy.zeros(numpy.shape(region_array))

    element = []
    [element.extend(item) for item in equivalent_elements]
    remaining = [i for i in range(1, k) if i not in element]
    length = len(equivalent_elements)
    for item in equivalent_elements:
        for i in item:
            modified_image[region_array == i] = equivalent_elements.index(item)+1
    for i in remaining:
        modified_image[region_array == i] = length+remaining.index(i)+1

    return modified_image


def vijay_hist_comp(input_image, num_bins=255):
    # ------------------------------------------------------------------------------------------
    # Histogram frequency finding
    maximum = numpy.max(input_image)
    minimum = numpy.min(input_image)
    block_width = (maximum - minimum) / num_bins
    bin_image_array = input_image - minimum
    freq_vec = numpy.zeros((num_bins,), dtype='int')
    for j in bin_image_array:
        for i in j:
            if round(i % block_width) == 0:
                if i / block_width != num_bins and i != 0:
                    freq_vec[(i / block_width).astype('int')] += 1
                if i / block_width == num_bins:
                    freq_vec[-1] += 1
                if i == 0:
                    freq_vec[0] += 1
            else:
                freq_vec[numpy.ceil(i / block_width).astype('int') - 1] += 1
    # ------------------------------------------------------------------------------------------
    # bin centers creating
    bins_vec = [(minimum + (block_width / 2)) + (i * block_width) for i in range(num_bins)]

    return [bins_vec, freq_vec]


def vijay_binary_conversion(input_image, threshold=None, check_threshold=0):
    # ------------------------------------------------------------------------------------------
    # Binary image creating from given threshold and image
    if check_threshold == 0:
        threshold = threshold_otsu(input_image)

    bin_image = numpy.zeros(numpy.shape(input_image))
    bin_image[input_image > threshold] = 255

    return bin_image

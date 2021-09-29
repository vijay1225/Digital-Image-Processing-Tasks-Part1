# Wrapper file to test all functions

from pathlib import Path

import numpy
import skimage.io
from matplotlib import pyplot

from AllFunctions import compute_hist, otsu_threshold, change_background
from AllFunctions import count_connected_components, binary_morphology, count_mser_components

def problem1():
    image_path = Path('Data/images/coins.png')
    num_bins = 10
    bins_vec, freq_vec, bins_vec_lib, freq_vec_lib = compute_hist(image_path, num_bins)
    bins_error = numpy.mean(numpy.square(bins_vec - bins_vec_lib))
    freq_error = numpy.mean(numpy.square(freq_vec - freq_vec_lib))
    print(f'Error in bins: {bins_error}')
    print(f'Error in freq: {freq_error}')
    return


def problem2():
    gray_image_path = Path('Data/images/coins.png')
    thr_w, thr_b, time_w, time_b, bin_image = otsu_threshold(gray_image_path)
    print(f'Using within class variance: Threshold: {thr_w}, Time: {time_w}')
    print(f'Using between class variance: Threshold: {thr_b}, Time: {time_b}')
    print()

    gray_image = skimage.io.imread(gray_image_path.as_posix())
    pyplot.subplot(121)
    pyplot.imshow(gray_image, cmap='gray')
    pyplot.subplot(122)
    pyplot.imshow(bin_image, cmap='gray')
    pyplot.show()
    return


def problem3():
    quote_image_path = Path('Data/images/quote.png')
    bg_image_path = Path('Data/images/background.png')
    modified_image = change_background(quote_image_path, bg_image_path)

    quote_image = skimage.io.imread(quote_image_path.as_posix())
    bg_image = skimage.io.imread(bg_image_path.as_posix())
    pyplot.subplot(131)
    pyplot.imshow(quote_image, cmap='gray')
    pyplot.subplot(132)
    pyplot.imshow(bg_image, cmap='gray')
    pyplot.subplot(133)
    pyplot.imshow(modified_image, cmap='gray')
    pyplot.show()
    return


def problem4():
    quote_image_path = Path('Data/images/quote.png')
    num_characters = count_connected_components(quote_image_path)
    print(f'Number of detected letters: {num_characters}')
    print()
    return


def problem5():
    noisy_image_path = Path('Data/images/noisy.png')
    clean_image = binary_morphology(noisy_image_path)

    noisy_image = skimage.io.imread(noisy_image_path.as_posix())
    pyplot.subplot(121)
    pyplot.imshow(noisy_image, cmap='gray')
    pyplot.subplot(122)
    pyplot.imshow(clean_image, cmap='gray')
    pyplot.show()
    return


def problem6():
    gray_image_path = Path('Data/images/mser.png')
    mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components = count_mser_components(gray_image_path)

    print(f'Number of characters detected using MSER: {num_mser_components}')
    print(f'Number of characters detected using Otsu: {num_otsu_components}')
    gray_image = skimage.io.imread(gray_image_path.as_posix())
    pyplot.subplot(131)
    pyplot.imshow(gray_image, cmap='gray')
    pyplot.title('Input Image')
    pyplot.subplot(132)
    pyplot.imshow(mser_binary_image, cmap='gray')
    pyplot.title('MSER Binary Image')
    pyplot.subplot(133)
    pyplot.imshow(otsu_binary_image, cmap='gray')
    pyplot.title('Otsu Binary Image')


def main():
    problem1()
    #problem2()
    # problem3()
    # problem4()
    # problem5()
    # problem6()
    return


if __name__ == '__main__':
    main()

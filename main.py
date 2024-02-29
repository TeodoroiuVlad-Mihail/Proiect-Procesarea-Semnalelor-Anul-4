from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.image as image
from skimage.metrics import structural_similarity as ssim


# ||x||2^2
def euclidian_norm_2(x):
    return np.sqrt(np.sum(x**2))


def rgb_to_gray(rgb):
    gray = np.zeros([rgb.shape[0], rgb.shape[1]])

    for i in range(gray.shape[0] - 1):
        for j in range(gray.shape[1] - 1):
            gray[i][j] = rgb[i][j][0] * 0.2989 + rgb[i][j][1] * 0.5870 + rgb[i][j][2] * 0.1140
    return gray


def gray_to_rgb(gray):
    rgb = np.zeros([gray.shape[0], gray.shape[1], 3])
    for i in range(rgb.shape[0] - 1):
        for j in range(rgb.shape[1] - 1):
            rgb[i][j][0] = gray[i][j]
            rgb[i][j][1] = gray[i][j]
            rgb[i][j][2] = gray[i][j]
    return rgb


def algorithm(h, y, sigma_e, denoising_operator, stopping_criterion):
    y_ = []
    x_ = []
    k = 0
    while k < stopping_criterion + 1:
        y_.append(y)
        x_.append(y)
        k = k + 1
    y_[0] = y
    k = 0
    delta = 2
    epsilon = 3 * np.e - 3
    delta_epsilon = 1 * np.e - 4
    tau = 3

    while k < stopping_criterion:
        k = k + 1
        if denoising_operator == "median_filer":
            x_[k] = median_filter(y_[k], sigma_e + delta)
        else:
            if denoising_operator == "gaussian_filter":
                x_[k] = gaussian_filter(y_[k], sigma_e + delta)
            else:
                x_[k] = median_filter(y_[k], sigma_e + delta)  # fallback
        # Compute y_[k] using (19)
        # g_ using (18)
        g_ = np.conjugate(fft(h)) / (abs(fft(h)) * abs(fft(h)) + epsilon * sigma_e * sigma_e)
        g_calc = ifft(g_ * (fft(y) - fft(h) * fft(x_[k])))
        y_[k] = g_calc + x_[k]

        # Compute nL and nR using (20)
        nL = 1 / (sigma_e * sigma_e) * euclidian_norm_2(y - ifft((fft(h) * fft(x_[k]))))
        nR = 1 / ((sigma_e + delta) * (sigma_e + delta)) * euclidian_norm_2(g_calc)

        if k > 1 and nL / nR < tau:
            epsilon = epsilon + delta_epsilon
            k = 0
    return x_[k]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    stopping_criterion = 30
    blurr_strength = 5
    denoising_operator = "median_filter"
    # denoising_operator = "gaussian_filter"

    # open results file
    results_file = open("results.txt", "w")
    average_ssim_file = open("average_ssim.txt", "w")

    for blurr_strength in [5, 10, 15, 25, 35, 50]:
        index = 0
        average_ssim = 0
        while index < 68:
            if index < 10:
                filename = "000" + str(index) + ".png"
            else:
                filename = "00" + str(index) + ".png"

            x = image.imread("CBSD68/original_png" + "/" + filename)
            x = rgb_to_gray(x)

            yi = image.imread("CBSD68/noisy" + str(blurr_strength) + "/" + filename)
            y = rgb_to_gray(yi)  # grayscale
            sizes = yi.shape
            h = np.eye(sizes[0], M=sizes[1])

            sigma_e = int(blurr_strength / 5)

            x_ = algorithm(h, y, sigma_e, denoising_operator, stopping_criterion)

            # Compare
            ssim_value = ssim(x, x_, data_range=x_.max() - x_.min())

            average_ssim += ssim_value

            # make it png format
            x_ = gray_to_rgb(x_)

            # write result to file
            plt.imsave("denoise results/noisy" + str(blurr_strength) + "/" + filename, x_)

            print("CBSD68/noisy" + str(blurr_strength) + "/" + filename + " processed! "
                  + "SSIM: " + '%.2f' % ssim_value)
            results_file.write("CBSD68/noisy" + str(blurr_strength) + "/" + filename + " processed! "
                               + "SSIM: " + '%.2f' % ssim_value + "\n")

            index += 1

        # write average SSIM
        average_ssim = average_ssim / 68
        average_ssim_file.write("noisy" + str(blurr_strength) + " Average SSIM: " + '%.2f' % average_ssim + "\n")

    # close file
    results_file.close()
    average_ssim_file.close()

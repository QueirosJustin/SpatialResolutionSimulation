#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import convolve2d

def apply_local_blur(image, sigma=1):
    """
    Apply Gaussian blur to the image locally.
    :param image: Input image.
    :param sigma: Standard deviation for Gaussian kernel.
    :return: Blurred image.
    """
    return scipy.ndimage.gaussian_filter(image, sigma=sigma)

def get_image_size_from_user():
    while True:
        try:
            size = int(input("Please enter the size of the image (between 3 and 1000): "))
            if 3 <= size <= 30:
                return size
            else:
                print("Size must be an integer between 3 and 100.")
        except ValueError:
            print("This is not a valid integer. Please try again.")

def system_model(x, noise_level, sigma=1):
    """
    Simulate the system's response with local blur and additive Gaussian noise.
    """
    blurred_x = apply_local_blur(x, sigma=sigma)
    y = blurred_x.flatten() + noise_level * np.random.randn(blurred_x.size)
    return y

def reconstruct_image(y, image_size):
    """
    A simple image reconstruction algorithm.
    """
    x_hat = y
    return x_hat.reshape(image_size, image_size)

def monte_carlo_local_impulse_response(noise_level, perturbation, num_samples, image_size, sigma=1):
    """
    Estimates the local impulse response using Monte Carlo simulation.
    """
    x = np.zeros((image_size, image_size))
    x[image_size // 2, image_size // 2] = 1  # Central pixel on

    measurements = [system_model(x, noise_level, sigma) for _ in range(num_samples)]
    reconstructions = np.array([reconstruct_image(y, image_size) for y in measurements])
    mean_reconstruction = np.mean(reconstructions, axis=0)

    x_perturbed = np.copy(x)
    x_perturbed[image_size // 2, image_size // 2] += perturbation
    perturbed_measurements = [system_model(x_perturbed, noise_level, sigma) for _ in range(num_samples)]
    perturbed_reconstructions = np.array([reconstruct_image(y, image_size) for y in perturbed_measurements])
    mean_perturbed_reconstruction = np.mean(perturbed_reconstructions, axis=0)

    local_impulse_response = (mean_perturbed_reconstruction - mean_reconstruction) / perturbation
    return local_impulse_response

def create_noisy_image(image, noise_level, sigma=1):
    """
    Creates a noise-altered image.
    """
    return system_model(image, noise_level, sigma).reshape(image.shape)

def gaussian_psf(sigma, size=5):
    """
    Generates a Gaussian Point Spread Function (PSF).
    :param sigma: Standard deviation of the Gaussian distribution.
    :param size: Size of the PSF. Should be odd.
    :return: Gaussian PSF.
    """
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = x[:, np.newaxis]
    x0, y0 = 0, 0  # Center of the PSF
    psf = np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))
    psf[psf < np.finfo(psf.dtype).eps * psf.max()] = 0
    sumh = psf.sum()
    if sumh != 0:
        psf /= sumh
    return psf


def richardson_lucy(image, psf, iterations=5):
    """
    Simple implementation of the Richardson-Lucy Deconvolution.
    :param image: The blurred and noisy image.
    :param psf: The Point Spread Function (PSF).
    :param iterations: Number of iterations.
    :return: Reconstructed image.
    """
    estimated = np.full(image.shape, 0.5)  # Initial estimate
    for _ in range(iterations):
        relative_blur = image / convolve2d(estimated, psf, 'same')
        estimated *= convolve2d(relative_blur, psf[::-1, ::-1], 'same')
    return estimated


def create_reconstructed_image(image, noise_level, num_samples, image_size, sigma=1):
    """
    Generates the average reconstructed image based on Monte Carlo simulations.
    """
    measurements = [system_model(image, noise_level, sigma) for _ in range(num_samples)]
    psf = gaussian_psf(sigma, size=5)  # Erzeugen Sie die PSF mit der gleichen Sigma wie beim Blur
    reconstructions = [richardson_lucy(y.reshape(image_size, image_size), psf, iterations=1200) for y in measurements]
    return np.mean(reconstructions, axis=0)


if __name__ == "__main__":
    image_size = get_image_size_from_user()
    noise_level = 0.0001
    perturbation = 0.01
    num_samples = 30
    sigma = 1

    original_image = np.zeros((image_size, image_size))
    #original_image = draw_smiley(image_size)
    original_image[image_size // 2, image_size // 2] = 1

    noisy_image = create_noisy_image(original_image, noise_level, sigma)
    reconstructed_image = create_reconstructed_image(original_image, noise_level, num_samples, image_size, sigma)
    lir = monte_carlo_local_impulse_response(noise_level, perturbation, num_samples, image_size, sigma)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(noisy_image, cmap='gray')
    ax[1].set_title('Low spatial resolution image')
    ax[1].axis('off')

    ax[2].imshow(reconstructed_image, cmap='gray')
    ax[2].set_title('Reconstructed image')
    ax[2].axis('off')

    ax[3].imshow(lir, cmap='hot', interpolation='nearest')
    ax[3].set_title('Estimated local impulse response')
    ax[3].axis('off')

    plt.show()
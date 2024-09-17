import numpy as np
import matplotlib.pyplot as plt

# Define the frequency parameters
F = 5  # Frequency of the cosine wave
Fs = 101  # Sampling frequency (number of samples, i.e., image width)

# Create the x-axis values
x = np.arange(Fs)

# Create the cosine wave
cos_wave = np.cos(2 * np.pi * F * x / Fs)

# Function to plot a wave and its Fourier transform
def plot_wave_and_fft(wave, title):
    plt.figure(figsize=(12, 6))

    # Plot the wave
    plt.subplot(1, 2, 1)
    plt.title(f'{title} Wave')
    plt.plot(wave)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Compute the Fourier transform of the wave
    fft_wave = np.fft.fftshift(np.fft.fft(wave))
    magnitude_spectrum = np.abs(fft_wave)

    # Plot the magnitude spectrum
    plt.subplot(1, 2, 2)
    plt.title('Fourier Transform Magnitude Spectrum')
    plt.plot(np.log(magnitude_spectrum + 1))  # Use log scale for better visualization
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

    plt.show()

# Plot the cosine wave and its Fourier transform
plot_wave_and_fft(cos_wave, 'Cosine')

# Create the image with the cosine wave pattern across rows
image_rows = np.tile(cos_wave, (Fs, 1))

# Create the image with the cosine wave pattern across columns
cos_wave_col = np.cos(2 * np.pi * F * x[:, np.newaxis] / Fs)
image_cols = np.tile(cos_wave_col, (1, Fs))

# Create the image with the cosine wave pattern diagonally
cos_wave_diag = np.cos(2 * np.pi * F * (x + x[:, np.newaxis]) / Fs)
image_diag = cos_wave_diag

# Function to plot image and its Fourier transform
def plot_image_and_fft(image, title):
    plt.figure(figsize=(12, 6))

    # Plot the image
    plt.subplot(1, 2, 1)
    plt.title(f'{title} Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Compute the Fourier transform of the image
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    magnitude_spectrum = np.abs(fft_image)

    # Plot the magnitude spectrum
    plt.subplot(1, 2, 2)
    plt.title('Fourier Transform Magnitude Spectrum')
    # Use log scale for better visualization
    plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
    plt.axis('off')

    plt.show()

# Plot the images and their Fourier transforms
plot_image_and_fft(image_rows, 'Cosine Wave Across Rows')
plot_image_and_fft(image_cols, 'Cosine Wave Across Columns')
plot_image_and_fft(image_diag, 'Cosine Wave Diagonally')
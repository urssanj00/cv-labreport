import numpy as np
import matplotlib.pyplot as plt

# Parameters of the sinusoid
frequency = 5  # Hz
amplitude = 1
duration = 1  # seconds

# Sampling rates
nyquist_rate = 2 * frequency
sampling_rates = [nyquist_rate / 1.5, nyquist_rate, nyquist_rate * 3]

# Time vector for continuous signal
t_continuous = np.linspace(0, duration, 500)
continuous_signal = amplitude * np.sin(2 * np.pi * frequency * t_continuous)

# Plotting
fig, axs = plt.subplots(len(sampling_rates), 2, figsize=(15, 5 * len(sampling_rates)))

for i, sampling_rate in enumerate(sampling_rates):
    # Time vector for sampled signal
    t_sampled = np.arange(0, duration, 1 / sampling_rate)
    sampled_signal = amplitude * np.sin(2 * np.pi * frequency * t_sampled)

    # Plot the time-domain signal
    axs[i, 0].plot(t_continuous, continuous_signal, label='Continuous Signal')
    axs[i, 0].stem(t_sampled, sampled_signal, linefmt='r-', markerfmt='ro', basefmt='r-',
                   label='Sampled Signal')
    axs[i, 0].set_xlabel('Time (s)')
    axs[i, 0].set_ylabel('Amplitude')
    axs[i, 0].set_title(f'Sampling at {sampling_rate} Hz '
                        f'({"Nyquist Rate" if sampling_rate == nyquist_rate else ("Below" if sampling_rate < nyquist_rate else "Above")})')
    axs[i, 0].legend()

    # Compute and plot the Fourier Transform
    frequencies = np.fft.fftfreq(len(sampled_signal), 1 / sampling_rate)
    fft_values = np.abs(np.fft.fft(sampled_signal))
    axs[i, 1].stem(frequencies, fft_values)
    axs[i, 1].set_xlabel('Frequency (Hz)')
    axs[i, 1].set_ylabel('Magnitude')
    axs[i, 1].set_title('Fourier Transform of Sampled Signal')
    axs[i, 1].set_xlim(-3 * frequency, 3 * frequency)  # Limit frequency range for clarity

plt.tight_layout()
plt.show()
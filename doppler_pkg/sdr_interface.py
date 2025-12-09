import numpy as np
import time
import datetime
from .constants import TX_FREQUENCY

try:
    from rtlsdr import RtlSdr
    HAS_RTLSDR = True
except ImportError:
    HAS_RTLSDR = False

class SDRInterface:
    """
    Interface for communicating with RTL-SDR dongle.
    """
    def __init__(self, sample_rate=2.048e6, center_freq=TX_FREQUENCY, gain='auto'):
        if not HAS_RTLSDR:
            raise ImportError("pyrtlsdr is not installed. Please install it to use live SDR mode.")
        
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq
        self.sdr.gain = gain
        self.sample_rate = sample_rate

    def close(self):
        self.sdr.close()

    def capture_samples(self, duration_sec):
        """
        Capture IQ samples for a specified duration.
        """
        num_samples = int(self.sdr.sample_rate * duration_sec)
        # Read samples (this is blocking)
        samples = self.sdr.read_samples(num_samples)
        return samples

    def measure_frequency(self, samples):
        """
        Estimate the carrier frequency from IQ samples using FFT.
        Returns (peak_freq, fft_magnitude_array)
        """
        # Use a window function to reduce spectral leakage
        window = np.hamming(len(samples))
        samples_windowed = samples * window
        
        # Compute FFT
        fft_result = np.fft.fft(samples_windowed)
        fft_freqs = np.fft.fftfreq(len(samples), d=1/self.sample_rate)
        
        # Find peak
        fft_result_shifted = np.fft.fftshift(fft_result)
        fft_freqs_shifted = np.fft.fftshift(fft_freqs)
        
        peak_idx = np.argmax(np.abs(fft_result_shifted))
        peak_freq_offset = fft_freqs_shifted[peak_idx]
        
        return self.sdr.center_freq + peak_freq_offset, np.abs(fft_result_shifted)

class MockSDR:
    """
    Simulates an RTL-SDR for testing without hardware.
    Generates a carrier with Doppler shift based on a fake pass.
    """
    def __init__(self, sample_rate=2.048e6, center_freq=TX_FREQUENCY, gain='auto'):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        print(f"[MockSDR] Initialized at {center_freq/1e6} MHz")
        self.start_time = time.time()

    def close(self):
        print("[MockSDR] Closed")

    def capture_samples(self, duration_sec):
        """
        Generate synthetic IQ samples.
        """
        num_samples = int(self.sample_rate * duration_sec)
        t = np.arange(num_samples) / self.sample_rate
        
        # Simulate a simple Doppler curve: f(t) = f0 + max_doppler * cos(...)
        # Just a linear drift for testing: starts at +3kHz, drifts to -3kHz over 10 mins
        elapsed = time.time() - self.start_time
        # 10 minute pass (600s)
        progress = elapsed / 600.0 
        doppler_shift = 3000 * (1 - 2 * progress) # +3k to -3k
        
        # Generate signal
        # exp(j * 2 * pi * f * t)
        signal = np.exp(1j * 2 * np.pi * doppler_shift * t)
        
        # Add noise
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.5
        
        return signal + noise

    def measure_frequency(self, samples):
        # Same logic as real SDR
        window = np.hamming(len(samples))
        samples_windowed = samples * window
        fft_result = np.fft.fft(samples_windowed)
        fft_freqs = np.fft.fftfreq(len(samples), d=1/self.sample_rate)
        
        fft_result_shifted = np.fft.fftshift(fft_result)
        fft_freqs_shifted = np.fft.fftshift(fft_freqs)
        
        peak_idx = np.argmax(np.abs(fft_result_shifted))
        peak_freq_offset = fft_freqs_shifted[peak_idx]
        
        return self.center_freq + peak_freq_offset, np.abs(fft_result_shifted)

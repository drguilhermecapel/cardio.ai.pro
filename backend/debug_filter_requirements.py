from scipy import signal as sig
import numpy as np

def debug_bandpass_filter_requirements():
    fs = 250
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = sig.butter(4, [low, high], btype='band')
    
    print(f'Bandpass Filter coefficients b length: {len(b)}')
    print(f'Bandpass Filter coefficients a length: {len(a)}')
    print(f'Expected padlen: {3 * max(len(a), len(b)) - 1}')
    print(f'Minimum signal length needed: {3 * (3 * max(len(a), len(b)) - 1)}')
    
    for length in [100, 500, 1000, 5000, 7500, 10000, 15000, 20000]:
        try:
            test_signal = np.random.randn(length)
            result = sig.filtfilt(b, a, test_signal)
            print(f'Signal length {length}: SUCCESS')
            if length >= 10000:  # Found a working length
                break
        except ValueError as e:
            print(f'Signal length {length}: FAILED - {e}')

if __name__ == "__main__":
    debug_bandpass_filter_requirements()

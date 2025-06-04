import numpy as np
from scipy import signal

print("Testing filter requirements for iirnotch...")

try:
    b, a = signal.iirnotch(50, Q=30, fs=250)
    test_signal = np.array([0.1] * 10)  # 10 samples
    result = signal.filtfilt(b, a, test_signal)
    print('10 samples: OK')
except ValueError as e:
    print(f'10 samples: {e}')

try:
    test_signal = np.array([0.1] * 20)  # 20 samples
    result = signal.filtfilt(b, a, test_signal)
    print('20 samples: OK')
except ValueError as e:
    print(f'20 samples: {e}')

try:
    test_signal = np.array([0.1] * 50)  # 50 samples
    result = signal.filtfilt(b, a, test_signal)
    print('50 samples: OK')
except ValueError as e:
    print(f'50 samples: {e}')

print(f"\nFilter coefficients length: b={len(b)}, a={len(a)}")
print(f"Filter order: {max(len(b), len(a)) - 1}")

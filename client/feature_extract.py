"""
Feature Extraction Module

This module converts I/Q samples from the RadioML dataset into
feature vectors for machine learning. Supports multiple feature
dimensionalities:

  - 8D  : Analog AMC features (instantaneous amplitude/frequency statistics)
  - 16D : Traditional features (I/Q stats + spectral + time-domain)
  - 24D : Extended features (16D + higher-order cumulants + envelope + phase)

The 24D mode adds discriminative features drawn from the AMC literature:
  Higher-Order Cumulants  (C20, C21, C40, C42)
  Signal envelope features (max/min ratio, crest factor)
  Phase features           (phase std, phase entropy)
"""

import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Instantaneous signal helpers
# ---------------------------------------------------------------------------

def compute_instantaneous_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous amplitude from complex signal.

    Args:
        signal: Complex-valued signal array

    Returns:
        Instantaneous amplitude (magnitude) of the signal
    """
    return np.abs(signal)


def compute_instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    Compute unwrapped instantaneous phase from complex signal.

    Args:
        signal: Complex-valued signal array

    Returns:
        Unwrapped instantaneous phase in radians
    """
    phase = np.angle(signal)
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase


def compute_instantaneous_frequency(signal: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Compute instantaneous frequency from complex signal.
    formula:freq = (1 / 2*pi) * d(phase)/dt = (1 / 2*pi) * d(phase)/dn * fs

    Args:
        signal: Complex-valued signal array
        fs: Sampling frequency (default: 128 for RML2016.10a dataset)

    Returns:
        Instantaneous frequency with same length as input (padded)
    """

    phase = compute_instantaneous_phase(signal)

    freq_diff = np.diff(phase) / (2 * np.pi) * fs


    instantaneous_freq = np.pad(freq_diff, (0, 1), mode='edge')

    return instantaneous_freq


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_statistical_features(data: np.ndarray, epsilon: float = 1e-9) -> Dict[str, float]:
    """
    Compute statistical features (mean, variance, skewness, kurtosis) from data.

    Uses epsilon-based stability for skewness and kurtosis calculations to handle
    edge cases with zero or near-zero standard deviation.

    Args:
        data: 1D array of values
        epsilon: Small value for numerical stability (default: 1e-9)

    Returns:
        Dictionary with keys: mean, variance, skewness, kurtosis (excess kurtosis)
    """
    mean = np.mean(data)
    variance = np.var(data)
    std = np.sqrt(variance)


    if std < epsilon:
        return {
            'mean': float(mean),
            'variance': float(variance),
            'skewness': 0.0,
            'kurtosis': 0.0
        }


    centered = data - mean


    skewness = np.mean(centered ** 3) / (std ** 3 + epsilon)


    raw_kurtosis = np.mean(centered ** 4) / (std ** 4 + epsilon)
    excess_kurtosis = raw_kurtosis - 3.0

    return {
        'mean': float(mean),
        'variance': float(variance),
        'skewness': float(skewness),
        'kurtosis': float(excess_kurtosis)
    }


# ---------------------------------------------------------------------------
# Higher-Order Cumulants (HOC)
# ---------------------------------------------------------------------------

def compute_higher_order_cumulants(signal: np.ndarray, epsilon: float = 1e-12) -> Dict[str, float]:
    """
    Compute higher-order cumulants that are widely used for automatic
    modulation classification (AMC).

    Definitions follow Swami & Sadler (2000) using centred moments:
        C20 = E[x^2]
        C21 = E[x * conj(x)]   (= signal power for zero-mean signals)
        C40 = cum(x,x,x,x)     = M40 - 3*M20^2
        C42 = cum(x,x,conj(x),conj(x)) = M42 - |M20|^2 - 2*M21^2

    where M_pq = E[x^p * conj(x)^q] computed on the centred signal.

    All returned values are real (magnitudes for complex cumulants).

    Args:
        signal: Complex-valued signal array (zero-mean preferred but
                centring is applied internally).
        epsilon: Small value to avoid division by zero.

    Returns:
        Dictionary with keys: C20, C21, C40, C42
    """
    x = signal - np.mean(signal)
    n = len(x)

    # Second-order moments
    M20 = np.mean(x ** 2)                 # E[x^2]
    M21 = np.mean(x * np.conj(x))         # E[|x|^2]  (real, non-negative)

    # Fourth-order moments
    M40 = np.mean(x ** 4)                 # E[x^4]
    M42 = np.mean((x ** 2) * (np.conj(x) ** 2))  # E[x^2 * conj(x)^2]

    # Cumulants
    C20 = M20
    C21 = M21
    C40 = M40 - 3.0 * M20 ** 2
    C42 = M42 - np.abs(M20) ** 2 - 2.0 * M21 ** 2

    return {
        'C20': float(np.abs(C20)),
        'C21': float(np.abs(C21)),
        'C40': float(np.abs(C40)),
        'C42': float(np.abs(C42)),
    }


# ---------------------------------------------------------------------------
# Signal envelope features
# ---------------------------------------------------------------------------

def compute_envelope_features(signal: np.ndarray, epsilon: float = 1e-12) -> Dict[str, float]:
    """
    Compute features derived from the signal envelope (instantaneous
    amplitude).

    Features:
        max_min_ratio : ratio of maximum to minimum envelope value.
                        When the minimum is near zero the ratio is clipped
                        to avoid Inf.
        crest_factor  : peak amplitude / RMS amplitude.  Measures how
                        "peaky" the signal is.

    Args:
        signal: Complex-valued signal array.
        epsilon: Small value for numerical stability.

    Returns:
        Dictionary with keys: max_min_ratio, crest_factor
    """
    envelope = np.abs(signal)

    env_max = np.max(envelope)
    env_min = np.min(envelope)

    # Max / min ratio (guard against zero denominator)
    if env_min < epsilon:
        max_min_ratio = env_max / epsilon
    else:
        max_min_ratio = env_max / env_min

    # Crest factor = peak / RMS
    rms = np.sqrt(np.mean(envelope ** 2) + epsilon)
    crest_factor = env_max / rms

    return {
        'max_min_ratio': float(np.nan_to_num(max_min_ratio, nan=0.0, posinf=1e6, neginf=0.0)),
        'crest_factor': float(np.nan_to_num(crest_factor, nan=0.0, posinf=1e6, neginf=0.0)),
    }


# ---------------------------------------------------------------------------
# Phase features
# ---------------------------------------------------------------------------

def compute_phase_features(signal: np.ndarray, n_bins: int = 64, epsilon: float = 1e-12) -> Dict[str, float]:
    """
    Compute features derived from the instantaneous phase.

    Features:
        phase_std     : standard deviation of the wrapped phase.  Captures
                        how spread out the constellation is in the angular
                        dimension.
        phase_entropy : Shannon entropy of the phase histogram.  Uniform
                        phase (e.g. noise) yields high entropy; clustered
                        phase (e.g. BPSK) yields low entropy.

    Non-zero samples only are used so that silence / zero-padding does not
    bias the phase distribution toward zero.

    Args:
        signal: Complex-valued signal array.
        n_bins: Number of histogram bins for entropy calculation.
        epsilon: Small value for numerical stability.

    Returns:
        Dictionary with keys: phase_std, phase_entropy
    """
    # Use only non-trivial samples to avoid phase noise at zero amplitude
    mask = np.abs(signal) > epsilon
    if np.sum(mask) < 2:
        return {'phase_std': 0.0, 'phase_entropy': 0.0}

    phase = np.angle(signal[mask])  # wrapped [-pi, pi]

    phase_std = float(np.std(phase))

    # Shannon entropy of phase histogram
    counts, _ = np.histogram(phase, bins=n_bins, range=(-np.pi, np.pi))
    probs = counts / (np.sum(counts) + epsilon)
    # Avoid log(0)
    probs = probs[probs > 0]
    phase_entropy = -float(np.sum(probs * np.log2(probs + epsilon)))

    return {
        'phase_std': float(np.nan_to_num(phase_std, nan=0.0)),
        'phase_entropy': float(np.nan_to_num(phase_entropy, nan=0.0, posinf=0.0, neginf=0.0)),
    }


# ---------------------------------------------------------------------------
# 8D analog features (unchanged from original)
# ---------------------------------------------------------------------------

def extract_analog_features(signal: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Extract 8-dimensional feature vector for analog modulation classification.

    This function implements the feature extraction approach from the notebook
    for AMC using instantaneous amplitude and frequency statistics.

    Args:
        signal: Complex-valued signal array (I + jQ)
        fs: Sampling frequency (default: 128 for RML2016.10a dataset)

    Returns:
        8-dimensional feature vector as numpy array with the following features:
        [amp_mean, amp_variance, amp_skewness, amp_kurtosis,
         freq_mean, freq_variance, freq_skewness, freq_kurtosis]
    """

    amplitude = compute_instantaneous_amplitude(signal)

    frequency = compute_instantaneous_frequency(signal, fs=fs)


    amp_features = compute_statistical_features(amplitude)


    freq_features = compute_statistical_features(frequency)


    feature_vector = np.array([
        amp_features['mean'],
        amp_features['variance'],
        amp_features['skewness'],
        amp_features['kurtosis'],
        freq_features['mean'],
        freq_features['variance'],
        freq_features['skewness'],
        freq_features['kurtosis']
    ], dtype=np.float32)


    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):

        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector


# ---------------------------------------------------------------------------
# 16D traditional features (unchanged from original)
# ---------------------------------------------------------------------------

def _extract_16d_features(iq_sample: np.ndarray) -> np.ndarray:
    """
    Internal helper: extract the original 16-dimensional feature vector.

    Features (in order):
        [ 0] I mean
        [ 1] I std
        [ 2] I variance
        [ 3] I skewness
        [ 4] I kurtosis
        [ 5] Q mean
        [ 6] Q std
        [ 7] Q variance
        [ 8] Q skewness
        [ 9] Q kurtosis
        [10] FFT peak magnitude
        [11] FFT peak frequency
        [12] Spectral centroid
        [13] Spectral bandwidth
        [14] Zero-crossing rate
        [15] Energy

    Args:
        iq_sample: I/Q sample with shape (2, 128)

    Returns:
        16-element float32 numpy array
    """
    i_channel = iq_sample[0, :]
    q_channel = iq_sample[1, :]

    features = []

    # --- I-channel statistics (5) ---
    features.append(np.mean(i_channel))
    features.append(np.std(i_channel))
    features.append(np.var(i_channel))
    i_skew = stats.skew(i_channel)
    features.append(0.0 if np.isnan(i_skew) else i_skew)
    i_kurt = stats.kurtosis(i_channel)
    features.append(0.0 if np.isnan(i_kurt) else i_kurt)

    # --- Q-channel statistics (5) ---
    features.append(np.mean(q_channel))
    features.append(np.std(q_channel))
    features.append(np.var(q_channel))
    q_skew = stats.skew(q_channel)
    features.append(0.0 if np.isnan(q_skew) else q_skew)
    q_kurt = stats.kurtosis(q_channel)
    features.append(0.0 if np.isnan(q_kurt) else q_kurt)

    # --- Frequency domain features (4) ---
    complex_signal = i_channel + 1j * q_channel

    fft_result = fft(complex_signal)
    fft_magnitude = np.abs(fft_result)
    fft_freqs = np.fft.fftfreq(len(complex_signal))

    peak_magnitude = np.max(fft_magnitude)
    features.append(peak_magnitude)

    peak_idx = np.argmax(fft_magnitude)
    peak_frequency = np.abs(fft_freqs[peak_idx])
    features.append(peak_frequency)

    half = len(fft_freqs) // 2
    spectral_centroid = (
        np.sum(fft_freqs[:half] * fft_magnitude[:half])
        / (np.sum(fft_magnitude[:half]) + 1e-10)
    )
    features.append(spectral_centroid)

    spectral_bandwidth = np.sqrt(
        np.sum(((fft_freqs[:half] - spectral_centroid) ** 2) * fft_magnitude[:half])
        / (np.sum(fft_magnitude[:half]) + 1e-10)
    )
    features.append(spectral_bandwidth)

    # --- Time domain features (2) ---
    i_zero_crossings = np.sum(np.diff(np.sign(i_channel)) != 0) / len(i_channel)
    q_zero_crossings = np.sum(np.diff(np.sign(q_channel)) != 0) / len(q_channel)
    zero_crossing_rate = (i_zero_crossings + q_zero_crossings) / 2
    features.append(zero_crossing_rate)

    energy = np.sum(np.abs(complex_signal) ** 2) / len(complex_signal)
    features.append(energy)

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# 24D extended features
# ---------------------------------------------------------------------------

def _extract_extended_features(signal: np.ndarray) -> np.ndarray:
    """
    Internal helper: extract the 8 *additional* features that turn
    a 16D vector into a 24D vector.

    Additional features (indices 16-23 when concatenated):
        [16] C20  — 2nd-order cumulant (magnitude)
        [17] C21  — 2nd-order cumulant (conjugate, = power)
        [18] C40  — 4th-order cumulant
        [19] C42  — 4th-order cumulant (mixed conjugate)
        [20] Envelope max/min ratio
        [21] Crest factor
        [22] Phase standard deviation
        [23] Phase entropy

    Args:
        signal: Complex-valued signal array of length 128.

    Returns:
        8-element float32 numpy array
    """
    hoc = compute_higher_order_cumulants(signal)
    env = compute_envelope_features(signal)
    pha = compute_phase_features(signal)

    extra = np.array([
        hoc['C20'],
        hoc['C21'],
        hoc['C40'],
        hoc['C42'],
        env['max_min_ratio'],
        env['crest_factor'],
        pha['phase_std'],
        pha['phase_entropy'],
    ], dtype=np.float32)

    # Sanitise
    extra = np.nan_to_num(extra, nan=0.0, posinf=1e6, neginf=0.0)
    return extra


# ---------------------------------------------------------------------------
# Public API — backward-compatible entry point
# ---------------------------------------------------------------------------

def extract_features_from_iq(
    iq_sample: np.ndarray,
    use_analog_features: bool = False,
) -> np.ndarray:
    """
    Extract feature vector from I/Q samples.

    Supports two modes:
    1. Traditional 16-dimensional features (default)
    2. Analog AMC 8-dimensional features (when use_analog_features=True)

    Traditional features include:
    - Statistical features (10): mean, std, variance, skewness, kurtosis for I and Q
    - Frequency domain features (4): FFT peak magnitude, FFT peak frequency, spectral centroid, spectral bandwidth
    - Time domain features (2): zero-crossing rate, energy

    Analog AMC features include:
    - Amplitude statistics (4): mean, variance, skewness, kurtosis
    - Frequency statistics (4): mean, variance, skewness, kurtosis

    Args:
        iq_sample: I/Q sample with shape (2, 128) where first row is I, second is Q
        use_analog_features: If True, extract 8D analog features; if False, extract 16D traditional features

    Returns:
        Feature vector of shape (8,) or (16,) depending on mode
    """

    if iq_sample.shape != (2, 128):
        raise ValueError(f"Expected shape (2, 128), got {iq_sample.shape}")

    signal = iq_sample[0, :] + 1j * iq_sample[1, :]

    if use_analog_features:
        return extract_analog_features(signal, fs=128)

    return _extract_16d_features(iq_sample)


# ---------------------------------------------------------------------------
# Public API — new 24D entry point
# ---------------------------------------------------------------------------

def extract_features_from_iq_extended(iq_sample: np.ndarray) -> np.ndarray:
    """
    Extract 24-dimensional feature vector from I/Q samples.

    This is the *full* feature set, combining the original 16D
    traditional features with 8 additional discriminative features
    used in the AMC literature.

    Feature layout (24D):
        [ 0- 4] I-channel statistics   (mean, std, var, skew, kurt)
        [ 5- 9] Q-channel statistics   (mean, std, var, skew, kurt)
        [10-13] Spectral features       (FFT peak mag, FFT peak freq,
                                         spectral centroid, spectral BW)
        [14-15] Time-domain features    (zero-crossing rate, energy)
        ------- new features below -------
        [16]    C20  — 2nd-order cumulant |E[x^2]|
        [17]    C21  — 2nd-order cumulant E[|x|^2] (signal power)
        [18]    C40  — 4th-order cumulant
        [19]    C42  — 4th-order cumulant (mixed conjugate)
        [20]    Envelope max/min ratio
        [21]    Crest factor (peak / RMS)
        [22]    Phase standard deviation
        [23]    Phase entropy (Shannon, histogram-based)

    Args:
        iq_sample: I/Q sample with shape (2, 128)

    Returns:
        24-element float32 numpy array
    """
    if iq_sample.shape != (2, 128):
        raise ValueError(f"Expected shape (2, 128), got {iq_sample.shape}")

    # 16D base features
    base = _extract_16d_features(iq_sample)

    # Complex signal for extended features
    signal = iq_sample[0, :] + 1j * iq_sample[1, :]
    extra = _extract_extended_features(signal)

    combined = np.concatenate([base, extra])
    combined = np.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=0.0)
    return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API — mode-based convenience function
# ---------------------------------------------------------------------------

def extract_features(iq_sample: np.ndarray, mode: str = "16d") -> np.ndarray:
    """
    Unified feature extraction with selectable dimensionality.

    Args:
        iq_sample: I/Q sample with shape (2, 128).
        mode: One of
            ``"8d"``  — 8-dimensional analog AMC features
            ``"16d"`` — 16-dimensional traditional features (default)
            ``"24d"`` — 24-dimensional extended features

    Returns:
        Feature vector whose length matches the requested mode.

    Raises:
        ValueError: If *mode* is not one of the supported strings or
                    *iq_sample* has an unexpected shape.
    """
    mode = mode.strip().lower()
    if mode == "8d":
        return extract_features_from_iq(iq_sample, use_analog_features=True)
    elif mode == "16d":
        return extract_features_from_iq(iq_sample, use_analog_features=False)
    elif mode == "24d":
        return extract_features_from_iq_extended(iq_sample)
    else:
        raise ValueError(
            f"Unknown feature mode '{mode}'. Choose from '8d', '16d', '24d'."
        )


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------

def process_dataset(
    samples: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
    use_analog_features: bool = False,
    mode: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for an entire dataset of I/Q samples.

    The *mode* parameter takes precedence when supplied.  When it is
    ``None`` (default) the legacy *use_analog_features* flag is honoured
    for backward compatibility.

    Args:
        samples: Array of I/Q samples with shape (n_samples, 2, 128)
        labels: Array of labels with shape (n_samples,)
        verbose: Whether to print progress information
        use_analog_features: Legacy flag — if True, extract 8D analog
            features; if False, extract 16D traditional features.
            Ignored when *mode* is explicitly provided.
        mode: ``"8d"``, ``"16d"`` or ``"24d"``.  When provided this
            overrides *use_analog_features*.

    Returns:
        Tuple of (features, labels) where:
            - features: numpy array of shape (n_samples, D)
            - labels: numpy array of shape (n_samples,) (unchanged)
    """
    # Resolve effective mode
    if mode is not None:
        effective_mode = mode.strip().lower()
    else:
        effective_mode = "8d" if use_analog_features else "16d"

    dim_map = {"8d": 8, "16d": 16, "24d": 24}
    if effective_mode not in dim_map:
        raise ValueError(
            f"Unknown feature mode '{effective_mode}'. "
            f"Choose from {list(dim_map.keys())}."
        )
    feature_dim = dim_map[effective_mode]

    n_samples = samples.shape[0]
    features_list = []

    if verbose:
        print(
            f"Extracting {effective_mode} ({feature_dim}D) features "
            f"from {n_samples} samples..."
        )

    for i in range(n_samples):
        try:
            feat = extract_features(samples[i], mode=effective_mode)
            features_list.append(feat)
        except Exception as e:
            if verbose:
                print(
                    f"Warning: Failed to extract features for sample {i}: "
                    f"{str(e)}"
                )
            features_list.append(np.zeros(feature_dim, dtype=np.float32))

        if verbose and (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{n_samples} samples...")

    features_array = np.array(features_list, dtype=np.float32)

    if verbose:
        print(f"Feature extraction complete. Shape: {features_array.shape}")

    return features_array, labels


def normalize_features(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.

    Works with any feature dimensionality (8D, 16D, 24D, etc.).

    Args:
        features: Feature array of shape (n_samples, D)

    Returns:
        Tuple of (normalized_features, mean, std) where:
            - normalized_features: Normalized feature array
            - mean: Mean values for each feature dimension
            - std: Standard deviation for each feature dimension
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)


    std = np.where(std == 0, 1, std)

    normalized_features = (features - mean) / std

    return normalized_features, mean, std

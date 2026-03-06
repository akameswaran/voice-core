"""Computed acoustic diagnostics derived from analysis output.

Second-order computations: they take analysis results (not raw audio)
and compute diagnostic measures. All functions are pure — no file I/O,
no coaching text, no UI concerns.

Callable from research, evaluator, and live tiers.
"""


def compute_coupling_index(delta_f_a: float, delta_f_b: float) -> float:
    """ΔF difference between two vowels.

    Args:
        delta_f_a: ΔF of reference vowel (typically /ah/).
        delta_f_b: ΔF of test vowel (typically /ih/).

    Returns:
        Positive = test vowel suppresses ΔF (coupling present).
        Zero = no coupling. Negative = unusual.
    """
    return round(delta_f_a - delta_f_b, 1)


# Tilt-ΔF regression coefficients (placeholder — update with more data)
_FULLNESS_SLOPE = -0.015    # dB/oct per Hz of ΔF
_FULLNESS_INTERCEPT = 11.0  # dB/oct at ΔF=0


def compute_fullness_residual(
    delta_f_hz: float,
    spectral_tilt: float,
    slope: float = _FULLNESS_SLOPE,
    intercept: float = _FULLNESS_INTERCEPT,
) -> float | None:
    """Tilt-ΔF regression residual as exploratory fullness proxy.

    Positive = heavier than expected for this size.
    Negative = thinner than expected.
    Near-zero = matched (high fullness).

    Returns None if inputs are missing/zero.
    """
    if delta_f_hz <= 0 or spectral_tilt == 0:
        return None
    expected = slope * delta_f_hz + intercept
    return round(spectral_tilt - expected, 3)


def detect_opc_f1_zscore(f1_hz: float, norm_mean: float, norm_std: float) -> float:
    """F1 z-score against population norm for a given vowel.

    Positive = F1 higher than norm (potential OPC).
    """
    return (f1_hz - norm_mean) / norm_std if norm_std > 0 else 0.0


# Amplitude-based OPC thresholds (calibrated from n=1 speaker, 3 evals)
_AMP_OPC_F4_F1_THRESHOLD = -15.0   # dB
_AMP_OPC_F1_PROM_THRESHOLD = 25.0  # dB
_AMP_OPC_DF_FLOOR = 1200.0         # Hz


def detect_opc_amplitude(
    a4_a1_db: float,
    f1_prominence_db: float,
    delta_f_hz: float,
    f4_f1_threshold: float = _AMP_OPC_F4_F1_THRESHOLD,
    prom_threshold: float = _AMP_OPC_F1_PROM_THRESHOLD,
    df_floor: float = _AMP_OPC_DF_FLOOR,
) -> bool:
    """Detect constriction from amplitude ratios.

    Returns True if the vowel shows constriction signature:
    high ΔF (looks good) but F4/F1 collapsed and F1 prominence extreme.
    """
    return (
        delta_f_hz > df_floor
        and a4_a1_db < f4_f1_threshold
        and f1_prominence_db > prom_threshold
    )

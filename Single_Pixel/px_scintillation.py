"""
Script: px_scintillation.py
Author: Alexander Weiser
License: TBD
Date: November 2026
AI use: Syntax, layout of the script, structure of the functions (for efficiency and
reusability), file writing and saving, phrasing of the docstrings and spelling were 
supported by Anthropic's Claude.

This script models the scintillation seen by one SNSPD pixel and weights the click
probability of the pixel with it (SQ1). For each case, the normalised irradiance
u = I/<I> (mean 1) follows a model distribution: log-normal or gamma-gamma with parameters
from the Rytov variance, or negative exponential (speckle). For the log-normal and speckle
cases, a time series u(t) is generated from low-pass filtered Gaussian noise (optionally
multiplied with a pointing-jitter fade) and checked against the model. The static click
curve of the pixel, interpolated from the static sweep (placeholder until measured), and
an ideal curve with the fitted system detection efficiency are averaged over p(u). This
gives the mean click probability, the pixel efficiency and its ratio to the static case.
The time series are exported as attenuation sequences for the VOA (dynamic check,
px_dynamic.py).
The approach is inspired by R. Saathof's lecture on optical propagation (AE4900
Spacecraft Communications, TU Delft, 2025).

Usage (from the MTP-code folder, with the virtual environment active) on macOS:
    source MTPvenv/bin/activate
    python Single_Pixel/px_scintillation.py

Inputs:  settings in `config` below; static sweep (not implemented yet: placeholder)
Outputs: in a new subfolder of config["output_dir"] per run,
         Run__fc_<f_c>Hz__pointing_<on/off>__seed_<seed>__<date>_<time>:
         results.json (computed values and `config`), the figures pdf_cases, series_cases
         and click_curves, and sequence_<case>.csv per time series (t_s, u, delta_A_dB)

References
Al-Habash, M. A., Andrews, L. C., & Phillips, R. L. (2001). Mathematical model for the
    irradiance probability density function of a laser beam propagating through
    turbulent media. Optical Engineering, 40(8), 1554-1562. https://doi.org/10.1117/1.1386641
Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through random media
    (2nd ed.). SPIE Press. https://doi.org/10.1117/3.626196
Chahine, Y. K., Katz, E. J., Vyhnalek, B. E., & Tedder, S. A. (2023). Statistical
    analysis of fading power vectors for real-time atmospheric channel emulation.
    Proceedings of SPIE, 12413, 124131L. https://doi.org/10.1117/12.2648447
Diament, P., & Teich, M. C. (1970). Photodetection of low-level radiation through the
    turbulent atmosphere. Journal of the Optical Society of America, 60(11), 1489-1494.
Downey, J. N., Chahine, Y. K., Tedder, S. A., Katz, E. J., Marsden, M. A., &
    Clapham, J. S. (2024). Performance of a real-time photon counting optical receiver
    in the presence of emulated channel fading. Proceedings of SPIE, 12877, 128770Z.
    https://doi.org/10.1117/12.3002472
Giggenbach, D., Shrestha, A., Moll, F., Fuchs, C., & Saucke, K. (2019). Reference power
    vectors for the optical LEO downlink channel. IEEE International Conference on Space
    Optical Systems and Applications (ICSOS). https://doi.org/10.1109/ICSOS45490.2019.8978994
Goodman, J. W. (2020). Speckle phenomena in optics: Theory and applications (2nd ed.).
    SPIE Press. https://doi.org/10.1117/3.2548484
Ivanov, H., Leitgeb, E., Pezzei, P., & Freiberger, G. (2019). Experimental
    characterization of SNSPD receiver technology for deep space FSO under laboratory
    testbed conditions. Optik, 195, 163101. https://doi.org/10.1016/j.ijleo.2019.163101
Shrestha, A., Giggenbach, D., Mustafa, A., Pacheco-Labrador, J., Ramirez, J., & Rein, F.
    (2016). Fading testbed for free-space optical communications. Proceedings of SPIE,
    9991, 999105. https://doi.org/10.1117/12.2239111
Vasylyev, D., Vogel, W., & Semenov, A. A. (2018). Theory of atmospheric quantum channels
    based on the law of total probability. Physical Review A, 97, 063852.
    https://doi.org/10.1103/PhysRevA.97.063852
Wollman, E. E., Allmaras, J. P., Beyer, A. D., Korzh, B., Runyan, M. C., Narváez, L.,
    Farr, W. H., Marsili, F., Briggs, R. M., Miles, G. J., & Shaw, M. D. (2024).
    SNSPD-based detector system for NASA's Deep Space Optical Communications project.
    Optics Express, 32(27), 48185-48198. https://doi.org/10.1364/OE.541425
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, signal, special, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Plotting"))
import figstyle as fs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/alexanderweiser/Documents/TUDelft/MTP/MTP-Workplace/MTP-code/Data"    # data folder

config = {
    # cases (default values)
    "cases": {
        "none":     {"label": "no turbulence", "distribution": "lognormal",   "sigma_R2": 0.0},
        "Dr0_2":    {"label": "D/r0 = 2",      "distribution": "lognormal",   "sigma_R2": 0.023},
        "Dr0_8":    {"label": "D/r0 = 8",      "distribution": "lognormal",   "sigma_R2": 0.23},
        "moderate": {"label": "moderate",      "distribution": "gammagamma",  "sigma_R2": 2.0},
        "strong":   {"label": "strong",        "distribution": "gammagamma",  "sigma_R2": 20.0},
        "speckle":  {"label": "speckle",       "distribution": "exponential", "sigma_R2": None},
    },                              # possible values: name: {"label": str, "distribution": "lognormal", "gammagamma" or "exponential", "sigma_R2": float >= 0 (None for exponential: no free parameter)}

    # time series of the scintillation
    "f_c_Hz": 20.0,                 # cut-off frequency of the PSD # possible values: float > 0
    "psd_slope": 4,                 # slope a of the PSD # possible values: even integer >= 2 (Butterworth order a/2)
    "sample_rate_Hz": 1000.0,       # samples per second # possible values: float, at least about 20 x f_c
    "duration_s": 100.0,            # length of each time series # possible values: float, much longer than the correlation time
    "seed": 1,                      # seed of the random number generator # possible values: integer

    # pointing jitter
    "pointing": False,              # multiply every time series with a pointing fade # possible values: True, False
    "pointing_beta": 3.0,           # beta of the pointing fade, beta = theta^2 / (4 sigma^2), theta = 1/e^2 beam radius (angle), sigma = rms jitter per axis (angle) # possible values: float > 0
    "pointing_f_c_Hz": 10.0,        # cut-off frequency of each jitter axis # possible values: float > 0
    "pointing_psd_slope": 2,        # slope of each jitter axis # possible values: even integer >= 2

    # detector (placeholder until the static sweep is measured)
    "eta_sde": 0.8,                 # system detection efficiency: probability that one incident photon gives a click # possible values: float, 0 to 1
    "dc_per_pulse": 1e-4,           # mean number of dark counts in the time window of one pulse (DCR x window) # possible values: float >= 0

    # checks
    "fade_thresholds_dB": [3, 6, 10],   # fade depths for the fade statistics # possible values: list of floats > 0

    # output
    "output_dir": os.path.join(DATA_DIR, "OUTPUT_px_scintillation"),   # each run saves in a subfolder Run__fc_<f_c>Hz__pointing_<on/off>__seed_<seed>__<date>_<time> # possible values: any folder path (created if missing)
    "save_figures": True,           # possible values: True, False
    "show_figures": False,          # possible values: True, False
    "figure_format": "png",         # possible values: "png", "pdf", "svg"
    "save_sequences": True,         # time series as CSV for the VOA # possible values: True, False
}

# ---------------------------------------------------------------------------
# Save figures and files
# ---------------------------------------------------------------------------

def finish_figure(fig, config, name):
    """Save and/or show a finished figure, following the switches in config.

    Parameters
    ----------
    fig : matplotlib Figure, the finished figure
    config : dict, settings (uses "save_figures", "show_figures", "run_output_dir",
             "figure_format")
    name : str, file name without extension

    Returns
    -------
    None
    """
    if config["save_figures"]:
        os.makedirs(config["run_output_dir"], exist_ok=True)       # create folder if needed
        path = os.path.join(config["run_output_dir"], f"{name}.{config['figure_format']}")
        fig.savefig(path)                                           # dpi and bbox from thesis.mplstyle
        print(f"  Figure saved:   {os.path.basename(path)}")

    if config["show_figures"]:
        plt.show()
    else:
        plt.close(fig)

def save_results(results, config):
    """Save the computed values and the settings as results.json in config["run_output_dir"].

    Parameters
    ----------
    results : dict, {case name: dict of computed values}, filled in __main__
    config : dict, settings (uses "run_output_dir"); also stored in the file

    Returns
    -------
    None
    """
    output = {"created": datetime.now().isoformat(),
              "config": config,
              "cases": results}

    os.makedirs(config["run_output_dir"], exist_ok=True)
    path = os.path.join(config["run_output_dir"], "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=4, default=lambda value: value.tolist())   # numpy -> list / number
    print(f"  Results saved:  {os.path.basename(path)}")

# ---------------------------------------------------------------------------
# Scintillation parameters
# ---------------------------------------------------------------------------

def log_irradiance_variances(sigma_R2):
    """Large- and small-scale log-irradiance variances of a plane wave, from the Rytov variance.

    Andrews & Phillips (2005, Eq. 9.41, p.335 and Eq. 9.46, p.336), zero inner scale and
    infinite outer scale; X and Y are the factors of u = X Y (Eq. 9.7, p.329).

    Parameters
    ----------
    sigma_R2 : float, Rytov variance

    Returns
    -------
    sigma_lnX2 : float, log-variance of the large-scale fluctuations
    sigma_lnY2 : float, log-variance of the small-scale fluctuations
    """
    sigma_lnX2 = 0.49 * sigma_R2 / (1 + 1.11 * sigma_R2 ** (6 / 5)) ** (7 / 6)
    sigma_lnY2 = 0.51 * sigma_R2 / (1 + 0.69 * sigma_R2 ** (6 / 5)) ** (5 / 6)

    return sigma_lnX2, sigma_lnY2

def scintillation_index_strong(sigma_R2):
    """Scintillation index for weak to strong irradiance fluctuations, from the Rytov variance.

    sigma_I^2 = exp(sigma_lnX^2 + sigma_lnY^2) - 1 for a downlink with a point receiver
    (Andrews & Phillips, 2005, Eq. 12.40, p.497), with the log-variances from
    log_irradiance_variances.

    Parameters
    ----------
    sigma_R2 : float, Rytov variance

    Returns
    -------
    sigma_I2 : float, scintillation index
    """
    sigma_lnX2, sigma_lnY2 = log_irradiance_variances(sigma_R2)
    sigma_I2 = np.exp(sigma_lnX2 + sigma_lnY2) - 1

    return sigma_I2

def lognormal_parameters(sigma_I2):
    """Parameters of a log-normal distribution of u = I/<I> with mean 1.

    ln u is Gaussian with mean mu and standard deviation sigma (Andrews & Phillips, 2005,
    Eq. 5.93, p.156), with sigma^2 = ln(1 + sigma_I^2) (Eq. 5.95, p.157); <u> = 1 gives
    mu = -sigma^2 / 2.

    Parameters
    ----------
    sigma_I2 : float, scintillation index

    Returns
    -------
    parameters : dict with
        "distribution" : str, "lognormal" (selects the formula in irradiance_pdf and
                         scintillation_series)
        "sigma_I2"     : float, scintillation index (input)
        "mu"           : float, mean of ln u
        "sigma"        : float, standard deviation of ln u
    """
    sigma = np.sqrt(np.log(1 + sigma_I2))
    mu = -sigma ** 2 / 2

    parameters = {"distribution": "lognormal",
                  "sigma_I2": sigma_I2,
                  "mu": mu,
                  "sigma": sigma}
    return parameters

def gamma_gamma_parameters(sigma_R2):
    """Parameters of a gamma-gamma distribution of u = I/<I> with mean 1.

    u = X Y with independent, gamma-distributed X and Y of mean 1 (Al-Habash et al., 2001;
    Andrews & Phillips, 2005, Eq. 9.134-9.135, p.370); alpha and beta from the
    log-variances (Eq. 9.138, p.370), sigma_I^2 from alpha and beta (Eq. 9.139, p.371).

    Parameters
    ----------
    sigma_R2 : float, Rytov variance (> 0)

    Returns
    -------
    parameters : dict with
        "distribution" : str, "gammagamma" (selects the formula in irradiance_pdf)
        "sigma_I2"     : float, scintillation index from alpha and beta
        "alpha"        : float, shape parameter of the large-scale fluctuations X
        "beta"         : float, shape parameter of the small-scale fluctuations Y
    """
    sigma_lnX2, sigma_lnY2 = log_irradiance_variances(sigma_R2)

    alpha = 1 / (np.exp(sigma_lnX2) - 1)
    beta = 1 / (np.exp(sigma_lnY2) - 1)

    sigma_I2 = 1 / alpha + 1 / beta + 1 / (alpha * beta)

    parameters = {"distribution": "gammagamma",
                  "sigma_I2": sigma_I2,
                  "alpha": alpha,
                  "beta": beta}
    return parameters

def exponential_parameters():
    """Parameters of the negative-exponential distribution of u (fully developed speckle).

    p(u) = exp(-u) and sigma_I^2 = 1 (Goodman, 2020, Eq. 3.15-3.16, p.28): no free parameter.

    Returns
    -------
    parameters : dict with
        "distribution" : str, "exponential" (selects the formula in irradiance_pdf and
                         scintillation_series)
        "sigma_I2"     : float, scintillation index, always 1
    """
    parameters = {"distribution": "exponential",
                  "sigma_I2": 1.0}
    return parameters

def case_parameters(case):
    """Distribution parameters of one case of config["cases"].

    Parameters
    ----------
    case : dict, one entry of config["cases"] (uses "distribution", "sigma_R2")

    Returns
    -------
    parameters : dict from the parameter function of the distribution

    Raises
    ------
    ValueError : for an unknown distribution
    """
    distribution = case["distribution"]

    if distribution == "lognormal":
        parameters = lognormal_parameters(scintillation_index_strong(case["sigma_R2"]))
    elif distribution == "gammagamma":
        parameters = gamma_gamma_parameters(case["sigma_R2"])
    elif distribution == "exponential":
        parameters = exponential_parameters()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return parameters

# ---------------------------------------------------------------------------
# Scintillation PDFs
# ---------------------------------------------------------------------------

def irradiance_pdf(u, parameters):
    """Probability density p(u) of the normalised irradiance u = I/<I> for one case.

    lognormal:   Andrews & Phillips (2005, Eq. 5.93, p.156), scipy.stats.lognorm with
                 s = sigma and scale = exp(mu)
    gammagamma:  Andrews & Phillips (2005, Eq. 9.137, p.370)
    exponential: p(u) = exp(-u) (Goodman, 2020, Eq. 3.15, p.28), scipy.stats.expon

    Parameters
    ----------
    u : float or array, normalised irradiance (> 0)
    parameters : dict from lognormal_parameters, gamma_gamma_parameters or
                 exponential_parameters

    Returns
    -------
    p : float or array (same shape as u), probability density at u

    Raises
    ------
    ValueError : if sigma_I^2 = 0 (no turbulence: p(u) is a delta at u = 1) or the
                 distribution is unknown
    """
    u = np.asarray(u, dtype=float)
    distribution = parameters["distribution"]

    # case no turbulence
    if parameters["sigma_I2"] == 0:
        raise ValueError("No turbulence (sigma_I^2 = 0): p(u) is a delta function at u = 1.")

    if distribution == "lognormal":
        p = stats.lognorm.pdf(u, s=parameters["sigma"], scale=np.exp(parameters["mu"]))
    elif distribution == "gammagamma":
        alpha, beta = parameters["alpha"], parameters["beta"]
        p = (2 * (alpha * beta) ** ((alpha + beta) / 2) / (special.gamma(alpha) * special.gamma(beta))
             * u ** ((alpha + beta) / 2 - 1) * special.kv(alpha - beta, 2 * np.sqrt(alpha * beta * u)))  # special.kv(order, x) = modified Bessel function of the second kind K_order(x)
    elif distribution == "exponential":
        p = stats.expon.pdf(u)  # scipy returns 0 for u < 0 (intensities cannot be negative)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return p

# ---------------------------------------------------------------------------
# Scintillation time series
# ---------------------------------------------------------------------------

def correlation_time(f_c_Hz, psd_slope):
    """Correlation time of a process with PSD Phi(f) = 1 / (1 + (f/f_c)^a).

    omega_c tau_c = K(a), K(a) = 2.32^(1 - 3.86 a^-2) (Chahine et al., 2023, Eq. 7, p.4);
    tau_c is the lag at which the autocovariance has dropped to 0.315 of its maximum.

    Parameters
    ----------
    f_c_Hz : float, cut-off (-3 dB) frequency of the PSD [Hz]
    psd_slope : float, slope a of the PSD

    Returns
    -------
    tau_c_s : float, correlation time [s]
    """
    K = 2.32 ** (1 - 3.86 * psd_slope ** -2)
    tau_c_s = K / (2 * np.pi * f_c_Hz)

    return tau_c_s

def gaussian_process(n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng, n_warmup_tau=10):
    """Standard-normal time series Z(t) with PSD Phi(f) = 1 / (1 + (f/f_c)^a).

    Gaussian white noise through a Butterworth low-pass filter of order a/2, which has
    |H(f)|^2 = Phi(f) (Chahine et al., 2023, p.5; Downey et al., 2024, p.3). The first
    n_warmup_tau correlation times are dropped, then Z is normalised with its sample mean
    and standard deviation.

    Parameters
    ----------
    n_samples : int, length of the returned series
    f_c_Hz : float, cut-off (-3 dB) frequency of the PSD [Hz]
    psd_slope : int, slope a of the PSD, even (Butterworth order a/2)
    sample_rate_Hz : float, samples per second [Hz], much larger than 2 f_c
    rng : numpy.random.Generator, random number generator (from np.random.default_rng)
    n_warmup_tau : float, warm-up length in correlation times, dropped (default 10)

    Returns
    -------
    Z : array (n_samples,), Gaussian time series with mean 0 and variance 1
    """
    if psd_slope % 2 != 0:
        raise ValueError(f"psd_slope must be even (Butterworth order a/2), psd_slope = {psd_slope}.")

    # design of the filter
    order = psd_slope // 2
    sos = signal.butter(order, f_c_Hz, btype="lowpass", fs=sample_rate_Hz, output="sos")  # "sos" means "second-order sections": a chain of small 2nd-order filters

    # white noise and filtering
    n_warmup = int(np.ceil(n_warmup_tau * correlation_time(f_c_Hz, psd_slope) * sample_rate_Hz))
    white_noise = rng.standard_normal(n_samples + n_warmup)
    z = signal.sosfilt(sos, white_noise)[n_warmup:]  # runs the noise through the filter

    # normalisation
    Z = (z - np.mean(z)) / np.std(z)  # mean = 0, variance = 1

    return Z

def scintillation_series(parameters, n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng):
    """Time series u(t) of the normalised irradiance u = I/<I> for one case.

    lognormal:   u = exp(mu + sigma Z), ln u as a Gaussian process (Chahine et al., 2023, p.5)
    exponential: u = (Z_1^2 + Z_2^2) / 2 (Chahine et al., 2023, Eq. 15, p.10-11, k = 1)
    Z, Z_1 and Z_2 are independent series from gaussian_process; no series for gamma-gamma.

    Parameters
    ----------
    parameters : dict from lognormal_parameters or exponential_parameters
    n_samples : int, length of the series
    f_c_Hz : float, cut-off (-3 dB) frequency of the PSD of Z [Hz]
    psd_slope : int, slope a of the PSD of Z, even
    sample_rate_Hz : float, samples per second [Hz]
    rng : numpy.random.Generator, random number generator

    Returns
    -------
    u : array (n_samples,), normalised irradiance over time (mean about 1)

    Raises
    ------
    ValueError : for gamma-gamma or an unknown distribution
    """
    distribution = parameters["distribution"]

    if distribution == "lognormal":
        Z = gaussian_process(n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng)
        u = np.exp(parameters["mu"] + parameters["sigma"] * Z)
    elif distribution == "exponential":
        Z_1 = gaussian_process(n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng)
        Z_2 = gaussian_process(n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng)  # rng continues its sequence: new numbers, independent of Z_1
        u = (Z_1 ** 2 + Z_2 ** 2) / 2
    else:
        raise ValueError(f"No time series for distribution '{distribution}'")

    return u

def pointing_series(beta_pointing, n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng):
    """Time series u_p(t) of the pointing-jitter fade, normalised to mean 1.

    Gaussian beam and Gaussian jitter in two axes give a beta-distributed fade with
    b = beta_pointing (Giggenbach et al., 2019, p.2; Downey et al., 2024, p.4). With the
    jitter axes Z_x, Z_y from gaussian_process this is I_p = exp(-(Z_x^2 + Z_y^2) / (2 b)),
    normalised with <I_p> = b / (b + 1) (Downey et al., 2024, p.4): u_p = I_p (b + 1) / b.

    Parameters
    ----------
    beta_pointing : float, beta of the pointing fade, ratio of the squared beam radius to
                    four times the squared rms jitter (not the beta of the gamma-gamma)
    n_samples : int, length of the series
    f_c_Hz : float, cut-off (-3 dB) frequency of the PSD of each jitter axis [Hz]
    psd_slope : int, slope a of the PSD of each jitter axis, even
    sample_rate_Hz : float, samples per second [Hz]
    rng : numpy.random.Generator, random number generator

    Returns
    -------
    u_p : array (n_samples,), pointing fade normalised to mean 1 (multiply with u(t))
    """
    # jitter angles of the two axes
    Z_x = gaussian_process(n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng)
    Z_y = gaussian_process(n_samples, f_c_Hz, psd_slope, sample_rate_Hz, rng)  # rng continues: independent of Z_x

    # Gaussian beam profile turned into irradiance relative to on-axis
    I_p = np.exp(-(Z_x ** 2 + Z_y ** 2) / (2 * beta_pointing))

    u_p = I_p * (beta_pointing + 1) / beta_pointing  # mean = 1

    return u_p

# ---------------------------------------------------------------------------
# Checks of the time series
# ---------------------------------------------------------------------------

def measured_correlation_time(u, sample_rate_Hz):
    """Correlation time of a time series, measured from its autocovariance.

    First lag at which the autocovariance (Andrews & Phillips, 2005, Eq. 2.27, p.42),
    normalised by its value at lag 0, drops to 0.315 (definition of tau_c: Chahine et
    al., 2023, p.4).

    Parameters
    ----------
    u : array, time series (not constant)
    sample_rate_Hz : float, samples per second [Hz]

    Returns
    -------
    tau_c_s : float, correlation time [s] (nan if the autocovariance never drops to 0.315)
    """
    x = u - np.mean(u)                                   # fluctuation around the mean
    variance = np.mean(x ** 2)                           # autocovariance at lag 0

    for lag in range(1, x.size):
        autocovariance = np.mean(x[:-lag] * x[lag:]) / variance
        if autocovariance <= 0.315:
            return lag / sample_rate_Hz                  # first lag below 0.315, in seconds

    return np.nan

def series_statistics(u, parameters, sample_rate_Hz):
    """Statistics of one time series u(t), compared with its model.

    Parameters
    ----------
    u : array, time series of one case
    parameters : dict from case_parameters (uses "sigma_I2")
    sample_rate_Hz : float, samples per second [Hz]

    Returns
    -------
    statistics : dict with
        "mean"           : float, sample mean (model: 1)
        "sigma_I2_series": float, sample scintillation index var(u) / mean(u)^2
        "sigma_I2_model" : float, scintillation index of the model (without pointing)
        "tau_c_series_s" : float, measured correlation time of u [s]
        "u_min", "u_max" : float, smallest and largest value of u
    """
    statistics = {"mean": np.mean(u),
                  "sigma_I2_series": np.var(u) / np.mean(u) ** 2,
                  "sigma_I2_model": parameters["sigma_I2"],
                  "tau_c_series_s": measured_correlation_time(u, sample_rate_Hz),
                  "u_min": np.min(u),
                  "u_max": np.max(u)}
    return statistics

def fade_statistics(u, thresholds_dB, sample_rate_Hz):
    """Fade statistics of one time series, as in Giggenbach et al. (2019, Table III, p.3).

    A fade of depth D dB is a period with u < 10^(-D/10), i.e. more than D dB below the
    mean (u = 1).

    Parameters
    ----------
    u : array, time series of one case
    thresholds_dB : list of float, fade depths [dB]
    sample_rate_Hz : float, samples per second [Hz]

    Returns
    -------
    fades : dict {depth in dB: dict} with
        "fraction"      : float, fraction of the time in a fade
        "number"        : int, number of fades
        "mean_length_s" : float, mean duration of a fade [s] (nan if there is none)
    """
    fades = {}
    for depth_dB in thresholds_dB:
        in_fade = u < 10 ** (-depth_dB / 10)                 # True where u is below the threshold

        # finds where fades start ("+1") and end ("-1")
        edges = np.diff(np.concatenate(([0], in_fade.astype(int), [0])))
        starts = np.flatnonzero(edges == 1)  # indices where a fade starts
        ends = np.flatnonzero(edges == -1)    # indices where a fade has ended

        lengths_s = (ends - starts) / sample_rate_Hz

        fades[depth_dB] = {"fraction": np.mean(in_fade),
                           "number": starts.size,  # number of fades
                           "mean_length_s": np.mean(lengths_s) if starts.size > 0 else np.nan}
    return fades

def plot_pdfs(series_by_case, parameters_by_case, config):
    """p(u) of every case: model (line) and histogram of the time series (dots).

    Parameters
    ----------
    series_by_case : dict {case name: u(t)}, cases with a time series
    parameters_by_case : dict {case name: parameter dict}
    config : dict, settings (uses "cases"; passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure ("pdf_cases")
    """
    bin_edges = np.logspace(-3, 1, 61)                        # u from 1e-3 to 10, 15 bins per decade
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])     # geometric centres (log axis)
    u_grid = np.logspace(-3, 1, 400)                          # for the model lines

    names = list(parameters_by_case)
    n_rows = int(np.ceil(len(names) / 3))                    # three panels per row
    fig, axes = plt.subplots(n_rows, 3, figsize=fs.figsize(1.0, aspect=0.3 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        parameters = parameters_by_case[name]

        if parameters["sigma_I2"] == 0:
            ax.axvline(1.0, color=fs.COLORS[i], label="model (delta)")
        else:
            ax.plot(u_grid, irradiance_pdf(u_grid, parameters), color=fs.COLORS[i], label="model")

        if name in series_by_case:
            u = series_by_case[name]
            counts, _ = np.histogram(u, bins=bin_edges)  # count how many values fall into each bin
            density = counts / (u.size * np.diff(bin_edges))  # counts -> probability density (definition of p(u))
            ax.plot(bin_centres, density, linestyle="none", marker=fs.MARKERS[i],
                    markersize=3, color=fs.COLORS[i], label="series")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e2)
        ax.set_title(config["cases"][name]["label"])
        ax.legend()
    for ax in axes.flat[len(names):]:                         # empty panels if the last row is not full
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("normalised irradiance $u$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$p(u)$")
    finish_figure(fig, config, "pdf_cases")

def plot_series(series_by_case, config, duration_shown_s=2.0):
    """First seconds of every time series u(t), one panel per case.

    Parameters
    ----------
    series_by_case : dict {case name: u(t)}
    config : dict, settings (uses "sample_rate_Hz", "cases"; passed on to finish_figure)
    duration_shown_s : float, length of the shown part [s] (default 2)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure ("series_cases")
    """
    n_shown = int(round(duration_shown_s * config["sample_rate_Hz"]))
    t_s = np.arange(n_shown) / config["sample_rate_Hz"]  # time axis [s]

    fig, axes = plt.subplots(len(series_by_case), 1, figsize=fs.figsize(1.0, aspect=0.8),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]                                         # one column: 1D array of panels
    case_names = list(config["cases"])                        # fixed colour per case, as in the other figures
    for ax, (name, u) in zip(axes, series_by_case.items()):
        i = case_names.index(name)
        ax.plot(t_s, u[:n_shown], color=fs.COLORS[i])
        ax.axhline(1.0, color=fs.GREY, linestyle="--")       # mean
        ax.set_ylabel("$u$")
        ax.set_title(config["cases"][name]["label"])
    axes[-1].set_xlabel("time [s]")
    finish_figure(fig, config, "series_cases")

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def ideal_click_probability(n_mean, eta_sde, dc_per_pulse):
    """Click probability of the pixel for a pulse with a mean of n_mean photons.

    Counts in the time window of one pulse are Poisson distributed with mean eta_SDE n + d
    (Ivanov et al., 2019, Eq. 4, p.3), which gives P_click = 1 - exp(-(eta_SDE n + d)).

    Parameters
    ----------
    n_mean : float or array, mean number of photons per pulse at the system input
    eta_sde : float, system detection efficiency (0 to 1)
    dc_per_pulse : float, mean number of dark counts in the time window of one pulse

    Returns
    -------
    p_click : float or array (same shape as n_mean), probability of at least one click
    """
    n_mean = np.asarray(n_mean, dtype=float)

    p_click = -np.expm1(-(eta_sde * n_mean + dc_per_pulse))  # 1 - exp(-x), accurate also for very small x

    return p_click

def load_static_sweep(path):
    """Measured static sweep in the format of placeholder_sweep (not implemented yet)."""
    pass

def placeholder_sweep(eta_sde, dc_per_pulse, n_points=61):
    """Placeholder for load_static_sweep: laser-off point and points of the ideal click curve.

    Points at n = 0 (laser off) and from 1e-3 to 1e3 photons per pulse.

    Parameters
    ----------
    eta_sde : float, system detection efficiency (0 to 1)
    dc_per_pulse : float, mean number of dark counts in the time window of one pulse
    n_points : int, number of points (default 61, i.e. 10 per decade)

    Returns
    -------
    sweep : dict with
        "n_mean"       : array, mean number of photons per pulse of the points; first point 0
                         (laser off)
        "p_click"      : array, click probability of the points
        "dc_per_pulse" : float, mean number of dark counts in the time window of one pulse
        "interpolator" : PCHIP interpolator of p_click over n_mean, used by
                         measured_click_probability
    """
    n_mean = np.concatenate(([0.0], np.logspace(-3, 3, n_points)))   # laser-off point (n = 0) + sweep points
    p_click = ideal_click_probability(n_mean, eta_sde, dc_per_pulse)

    sweep = {"n_mean": n_mean,
             "p_click": p_click,
             "dc_per_pulse": dc_per_pulse,
             "interpolator": interpolate.PchipInterpolator(n_mean, p_click)}   # over n_mean: includes n = 0
    return sweep

def fit_eta_sde(n_mean, p_click, dc_per_pulse, p_max=0.1):
    """System detection efficiency from the linear part of a static sweep.

    SDE = photo-count rate / incident photon rate (Wollman et al., 2024, p.48190). The
    inverted ideal click curve, y = -ln(1 - P_click) - d = eta_SDE n, is fitted by least
    squares through the origin.

    Parameters
    ----------
    n_mean : array, mean number of photons per pulse of the sweep points
    p_click : array, measured click probability of the sweep points
    dc_per_pulse : float, mean number of dark counts in the time window of one pulse
    p_max : float, only points with p_click < p_max are used (default 0.1)

    Returns
    -------
    eta_sde : float, system detection efficiency
    """
    n_mean = np.asarray(n_mean, dtype=float)
    p_click = np.asarray(p_click, dtype=float)

    linear = p_click < p_max                                    # boolean mask: True for the points of the linear part
    y = -np.log1p(-p_click[linear]) - dc_per_pulse              # -ln(1 - P) - d = eta_SDE n
    eta_sde = np.sum(n_mean[linear] * y) / np.sum(n_mean[linear] ** 2)  # points at larger n get more weight

    return eta_sde

def measured_click_probability(n_mean, sweep):
    """Click probability at any n_mean, interpolated from a static sweep.

    Monotone cubic interpolation (PCHIP) over n, from the laser-off point (n = 0) to the
    highest sweep point; above it, the last value.

    Parameters
    ----------
    n_mean : float or array, mean number of photons per pulse
    sweep : dict from placeholder_sweep or load_static_sweep (uses "n_mean", "interpolator")

    Returns
    -------
    p_click : array (same shape as n_mean), click probability
    """
    n_mean = np.asarray(n_mean, dtype=float)
    p_click = sweep["interpolator"](np.clip(n_mean, 0, sweep["n_mean"][-1]))   # above the highest point: last value

    return p_click

# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

def mean_click_probability(n_mean, parameters, click_probability):
    """Mean click probability of the pixel under scintillation.

    P_mean(n) = integral_0^inf P_click(n u) p(u) du (Diament & Teich, 1970, Eq. 6, p.1491;
    Vasylyev et al., 2018, Eq. 45 and 47, p.10), with P_click from click_probability.
    No turbulence (sigma_I^2 = 0): P_mean = P_click(n).

    Parameters
    ----------
    n_mean : float or array, mean number of photons per pulse at the system input
    parameters : dict from case_parameters
    click_probability : function, P_click(n) of the pixel for an array of n (ideal or
                        measured click curve)

    Returns
    -------
    p_mean : array (same length as n_mean), mean click probability
    """
    n_mean = np.atleast_1d(np.asarray(n_mean, dtype=float))  # at least 1D, so that the loop also works for one number

    if parameters["sigma_I2"] == 0:                       # no turbulence: p(u) is a delta at u = 1
        return click_probability(n_mean)

    p_mean = np.empty_like(n_mean)                        # one result per n_mean, filled in the loop
    for i, n in enumerate(n_mean):
        # click probability at irradiance u times how often that irradiance occurs
        def integrand(u):                                 # function of u, called by integrate.quad
            return click_probability(n * u) * irradiance_pdf(u, parameters)
        p_mean[i], _ = integrate.quad(integrand, 0, np.inf, epsabs=1e-7, limit=100)   # tolerance above the interpolation error

    return p_mean

def pixel_efficiency(n_mean, p_click, p_dark):
    """Pixel efficiency: clicks caused by light per incident photon.

    epsilon = (P - P_dark) / n, with the dark clicks subtracted as for the system
    detection efficiency (Wollman et al., 2024, p.48190).

    Parameters
    ----------
    n_mean : array, mean number of photons per pulse
    p_click : array (same shape), click probability at n_mean (static or mean under
              scintillation)
    p_dark : float, click probability with the laser off (first point of the sweep)

    Returns
    -------
    efficiency : array (same shape as n_mean), pixel efficiency
    """
    efficiency = (np.asarray(p_click) - p_dark) / np.asarray(n_mean)

    return efficiency

def plot_click_curves(results, config):
    """Mean click probability, pixel efficiency and efficiency ratio over n, per case.

    Lines: ideal click curve; markers: measured sweep.

    Parameters
    ----------
    results : dict {case name: dict}, uses "n_mean", "p_mean_*", "efficiency_*", "ratio_*"
              with * = "ideal" and "measured"
    config : dict, settings (uses "cases"; passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure ("click_curves")
    """
    panels = [("p_mean", r"mean click probability $\bar P$"),
              ("efficiency", r"pixel efficiency $\varepsilon$"),
              ("ratio", r"efficiency ratio $r$")]

    fig, axes = plt.subplots(len(panels), 1, figsize=fs.figsize(1.0, aspect=1.1), sharex=True)
    for i, (name, case) in enumerate(config["cases"].items()):  # i: fixed colour of the case
        n_mean = results[name]["n_mean"]
        for ax, (key, ylabel) in zip(axes, panels):
            ax.plot(n_mean, results[name][key + "_ideal"], color=fs.COLORS[i], label=case["label"])
            ax.plot(n_mean, results[name][key + "_measured"], linestyle="none", marker=fs.MARKERS[i],
                    markersize=3, markevery=5, color=fs.COLORS[i])          # every 5th point, so the lines stay visible
            ax.set_ylabel(ylabel)

    axes[0].set_yscale("log")
    axes[0].legend(title="lines: ideal, markers: sweep")
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel(r"mean number of photons per pulse $\bar n$")
    finish_figure(fig, config, "click_curves")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def relative_attenuation_dB(u):
    """Attenuation relative to the mean that the VOA adds to play u(t).

    delta_A = -10 log10(u) [dB]; A(t) = A_mean + delta_A then plays the normalised power
    vector times the mean power (Shrestha et al., 2016, p.5).

    Parameters
    ----------
    u : array, normalised irradiance (mean 1)

    Returns
    -------
    delta_A_dB : array (same shape as u), attenuation relative to the mean [dB]
    """
    delta_A_dB = -10 * np.log10(u)

    return delta_A_dB

def export_sequence(u, delta_A_dB, name, config):
    """Save one time series as CSV for the VOA: time, u and relative attenuation.

    Parameters
    ----------
    u : array, normalised irradiance over time
    delta_A_dB : array (same shape), attenuation relative to the mean [dB]
    name : str, case name (file name sequence_<name>.csv)
    config : dict, settings (uses "sample_rate_Hz", "run_output_dir")

    Returns
    -------
    None
    """
    t_s = np.arange(u.size) / config["sample_rate_Hz"]                    # time axis [s]
    table = np.column_stack((t_s, u, delta_A_dB))                         # one row per sample

    os.makedirs(config["run_output_dir"], exist_ok=True)
    path = os.path.join(config["run_output_dir"], f"sequence_{name}.csv")
    np.savetxt(path, table, delimiter=",", header="t_s,u,delta_A_dB", comments="", fmt="%.6g")

    print(f"  Sequence saved: {os.path.basename(path)}")

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    fs.use()                                           # style of all figures according to Plotting/

    # run folder, random numbers, length of the series
    pointing_text = "on" if config["pointing"] else "off"
    run_name = (f"Run__fc_{config['f_c_Hz']:g}Hz__pointing_{pointing_text}__seed_{config['seed']}"
                f"__{datetime.now():%Y%m%d_%H%M}")
    config["run_output_dir"] = os.path.join(config["output_dir"], run_name)   # new folder per run
    rng = np.random.default_rng(config["seed"])        # one generator for the whole run; makes it reproducible
    n_samples = int(round(config["duration_s"] * config["sample_rate_Hz"]))
    tau_c_s = correlation_time(config["f_c_Hz"], config["psd_slope"])

    print("\n=== Run ===")
    print(f"  output folder: {config['run_output_dir']}")
    print(f"  time series:   {n_samples} samples, tau_c = {tau_c_s * 1e3:.1f} ms "
          f"({config['duration_s'] / tau_c_s:.0f} correlation times), pointing {pointing_text}")
    print(f"  detector:      placeholder sweep (eta_SDE = {config['eta_sde']}, d = {config['dc_per_pulse']:g})")

    # parameters and time series of every case
    print("\n=== Cases ===")
    print(f"  {'case':<15} {'distribution':<12} {'sigma_I2':>8}  time series")
    results = {}                                       # {case name: computed values}, saved to results.json
    parameters_by_case = {}                            # {case name: parameter dict}
    series_by_case = {}                                # {case name: u(t)}, only cases with a time series
    for name, case in config["cases"].items():
        parameters = case_parameters(case)
        parameters_by_case[name] = parameters
        results[name] = dict(parameters)               # copy, so that later additions do not change parameters

        if parameters["distribution"] != "gammagamma" and parameters["sigma_I2"] > 0:   # gamma-gamma: only p(u); no turbulence: u = 1, no series
            u = scintillation_series(parameters, n_samples, config["f_c_Hz"], config["psd_slope"],
                                     config["sample_rate_Hz"], rng)
            if config["pointing"]:
                u = u * pointing_series(config["pointing_beta"], n_samples, config["pointing_f_c_Hz"],
                                        config["pointing_psd_slope"], config["sample_rate_Hz"], rng)
            series_by_case[name] = u

        has_series = "yes" if name in series_by_case else "no"
        print(f"  {case['label']:<15} {parameters['distribution']:<12} {parameters['sigma_I2']:8.3f}  {has_series}")

    # checks of the time series against the model
    print("\n=== Checks of the time series ===")
    print(f"  {'case':<15} {'mean':>6} {'sigma_I2':>9} {'(model)':>8} {'tau_c [ms]':>11}")
    for name, u in series_by_case.items():
        statistics = series_statistics(u, parameters_by_case[name], config["sample_rate_Hz"])
        results[name].update(statistics)
        results[name]["fades"] = fade_statistics(u, config["fade_thresholds_dB"], config["sample_rate_Hz"])
        print(f"  {config['cases'][name]['label']:<15} {statistics['mean']:6.3f} "
              f"{statistics['sigma_I2_series']:9.3f} {statistics['sigma_I2_model']:8.3f} "
              f"{statistics['tau_c_series_s'] * 1e3:11.1f}")

    print(f"\n  {'fades':<15} {'depth':>6} {'time [%]':>9} {'number':>7} {'mean length [ms]':>17}")
    for name in series_by_case:
        label = config["cases"][name]["label"]
        for depth_dB, fade in results[name]["fades"].items():
            length_text = f"{fade['mean_length_s'] * 1e3:.2f}" if fade["number"] > 0 else "-"
            print(f"  {label:<15} {depth_dB:>3} dB {fade['fraction'] * 100:9.2f} {fade['number']:7d} {length_text:>17}")
            label = ""                                 # case name only in the first row

    # click curves: sweep, fitted eta_SDE, ideal and measured curve
    print("\n=== Click curves ===")
    sweep = placeholder_sweep(config["eta_sde"], config["dc_per_pulse"])      # placeholder until the static sweep is measured
    n_mean = sweep["n_mean"][1:]                                              # sweep points without the laser-off point
    p_dark = sweep["p_click"][0]                                              # laser-off point
    eta_sde_fit = fit_eta_sde(sweep["n_mean"], sweep["p_click"], sweep["dc_per_pulse"])

    def ideal(n):                                      # ideal curve with the fitted eta_SDE and the measured dark counts
        return ideal_click_probability(n, eta_sde_fit, sweep["dc_per_pulse"])

    def measured(n):                                   # interpolated sweep
        return measured_click_probability(n, sweep)

    p_exponential = mean_click_probability(n_mean, exponential_parameters(), ideal)
    p_formula = 1 - np.exp(-sweep["dc_per_pulse"]) / (1 + eta_sde_fit * n_mean)
    print(f"  eta_SDE fitted to the sweep:  {eta_sde_fit:.4f}")
    print(f"  self-test, exponential p(u):  max deviation {np.max(np.abs(p_exponential - p_formula)):.1e}")

    # mean click probability, pixel efficiency and efficiency ratio of every case
    efficiency_static_ideal = pixel_efficiency(n_mean, ideal(n_mean), p_dark)
    efficiency_static_measured = pixel_efficiency(n_mean, sweep["p_click"][1:], p_dark)

    print("\n=== Mean click probability ===")
    print(f"  {'case':<15} {'lowest r':>8} {'at n':>6} {'|measured - ideal|':>19} {'|series - ideal|':>17}")
    for name, parameters in parameters_by_case.items():
        p_ideal = mean_click_probability(n_mean, parameters, ideal)
        p_measured = mean_click_probability(n_mean, parameters, measured)
        efficiency_ideal = pixel_efficiency(n_mean, p_ideal, p_dark)
        efficiency_measured = pixel_efficiency(n_mean, p_measured, p_dark)
        ratio_ideal = efficiency_ideal / efficiency_static_ideal                    # fraction of the light-induced clicks that survives
        ratio_measured = efficiency_measured / efficiency_static_measured
        results[name].update({"n_mean": n_mean,
                              "p_mean_ideal": p_ideal, "p_mean_measured": p_measured,
                              "efficiency_ideal": efficiency_ideal, "efficiency_measured": efficiency_measured,
                              "ratio_ideal": ratio_ideal, "ratio_measured": ratio_measured})

        series_text = "-"
        if name in series_by_case:                     # check: time average over the series instead of p(u)
            p_series = np.array([np.mean(ideal(n * series_by_case[name])) for n in n_mean])
            results[name]["p_mean_series"] = p_series
            series_text = f"{np.max(np.abs(p_series - p_ideal)):.1e}"

        print(f"  {config['cases'][name]['label']:<15} {np.min(ratio_ideal):8.3f} {n_mean[np.argmin(ratio_ideal)]:6.2g} "
              f"{np.max(np.abs(p_measured - p_ideal)):19.1e} {series_text:>17}")
    print("  (r = efficiency ratio of the ideal curve; differences of P_mean: maximum over n)")

    # attenuation sequences for the VOA
    print("\n=== Sequences for the VOA ===")
    print(f"  {'case':<15} {'min [dB]':>8} {'max [dB]':>8} {'range [dB]':>10}")
    delta_A_by_case = {}                               # {case name: delta_A(t)}
    for name, u in series_by_case.items():
        delta_A_dB = relative_attenuation_dB(u)
        delta_A_by_case[name] = delta_A_dB
        results[name].update({"delta_A_min_dB": np.min(delta_A_dB), "delta_A_max_dB": np.max(delta_A_dB)})
        print(f"  {config['cases'][name]['label']:<15} {np.min(delta_A_dB):8.1f} {np.max(delta_A_dB):8.1f} "
              f"{np.max(delta_A_dB) - np.min(delta_A_dB):10.1f}")

    # output files
    print("\n=== Output ===")
    if config["save_sequences"]:
        for name, u in series_by_case.items():
            export_sequence(u, delta_A_by_case[name], name, config)
    save_results(results, config)
    plot_pdfs(series_by_case, parameters_by_case, config)
    plot_series(series_by_case, config)
    plot_click_curves(results, config)
    print()

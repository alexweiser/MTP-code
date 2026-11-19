"""
Script: TPSim_Analysis.py
Author: Alexander Weiser
License: TBD
Date: November 2026
AI use: Syntax, layout of the script, structure of the functions (for efficiency and
reusability), file writing and saving, phrasing of the docstrings and spelling were 
supported by Anthropic's Claude.

This script analyses the output of the turbulence propagation simulation in
TPSimulation.py, one .h5 file per turbulence strength (D/r0). The statistics are
computed over the realisations, per simulation pixel and without a detector model.
To validate the simulation, the statistics of the simulated pupil and focal-plane
fields are compared with analytic results and with the simulations of Saathof and
Zadeh (2025). The simulated atmosphere itself (Cn2 profile, layers, r0 per
layer, as stored by TPSimulation.py) is only described, as the input of the
simulation. In addition, single realisations are plotted to give an intuitive
understanding of the results.

Function names: simulation quantities in own words (e.g. scintillation_index),
theory functions with the term of the source (e.g. rytov_variance, airy_disk).

Usage (from the MTP-code folder, with the virtual environment active) on MacOS:
    source MTPvenv/bin/activate
    python Turbulence_Propagation_Simulation/TPSim_Analysis.py

Inputs:  .h5 files of TPSimulation.py in config["input_dir"], one per D/r0 in
         config["D_over_r0_cases"] (newest file per case); analysis settings in
         `config` below.
Outputs: printed per run: atmosphere table; pupil and power scintillation index and
         Strehl ratio, each with its theory value; focal / pupil power.
         Figures in a new subfolder of config["output_dir"] per run,
         Analysis_Run__model_<model>__Dr0_<D/r0 list>__M_<largest M_used>__tiptilt_<on/off>__<date>_<time>:
         atmosphere_cn2, pupil_scintillation_convergence_Dr0_<D/r0>, focal_scintillation_maps,
         focal_irradiance_cdfs, example_realisations_k_<k>, focal_airy_check.
         results.json in the same folder: all printed values per run plus `config`.

References
Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through random media
    (2nd ed.). SPIE Press. https://doi.org/10.1117/3.626196
Goodman, J. W. (1996). Introduction to Fourier optics (2nd ed.). McGraw-Hill.
Kaushal, H., & Kaddoum, G. (2017). Optical communication in space: Challenges and
    mitigation techniques. IEEE Communications Surveys & Tutorials, 19(1), 57-96.
    https://doi.org/10.1109/COMST.2016.2603518
Saathof, R., & Zadeh, I. E. (2025). Focal plane irradiance statistics for laser
    downlink beams. Proceedings of SPIE, 13669, 136690M. https://doi.org/10.1117/12.3069432
Schmidt, J. D. (2010). Numerical simulation of optical wave propagation with examples in
    MATLAB. SPIE Press. https://doi.org/10.1117/3.866274
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j1
from scipy.integrate import quad
from aotools import centre_of_gravity
import hcipy as hci
import os
import glob
import json
from datetime import datetime
from TPSimulation import get_cn2_function, slab_edges, make_optics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/alexanderweiser/Documents/TUDelft/MTP/MTP-Workplace/MTP-code/Data"    # data folder

config = {
    # input
    "cn2_model": "HV",                                              # Cn2 model of the runs to analyse # possible values: a model name used in TPSimulation.py (config["cn2_model"] there)
    "D_over_r0_cases": [0, 2, 8],                                   # state which runs to analyse # possible values: list of D/r0 with a .h5 file in input_dir (matched with 2 decimals, e.g. 2 -> "Dr0_2.00"); 0 needed as reference for Strehl ratio, example plot and Airy check
    "input_dir": os.path.join(DATA_DIR, "OUTPUT_TPSimulation"),     # folder with the .h5 files of TPSimulation.py # possible values: any folder path

    # output
    "output_dir": os.path.join(DATA_DIR, "OUTPUT_TPSim_Analysis"),  # each run saves its figures in a subfolder Analysis_Run__model_<model>__Dr0_<D/r0 list>__M_<largest M_used>__tiptilt_<on/off>__<date>_<time> # possible values: any folder path (created if missing)
    "save_figures": True,                                           # possible values: True, False
    "show_figures": False,                                          # possible values: True, False
    "figure_format": "png",                                         # possible values: "png", "pdf", "svg"

    # analysis settings
    "n_realisations_max": None,                                     # set max number of realisations for analysis # possible values: None (use all realisations), integer >= 1 (larger than M: all are used)
    "cdf_x_m": [0.0, 2.3e-6, 4.7e-6, 7.0e-6, 9.3e-6, 11.6e-6],      # focal positions along x (y = 0) for the irradiance CDFs, nearest pixel is used # possible values: list of positions in m, None (no CDFs)
    "tip_tilt_correction": False,                                    # possible values: True, False # True: every focal image is tip-tilt corrected
    "example_realisation": 0,                                       # realisation choice for example plot # possible values: integer from 0 to M-1 (larger values: last realisation of each run)
}

# ---------------------------------------------------------------------------
# Reading and loading the .h5 files
# ---------------------------------------------------------------------------

def find_run_file(config, D_over_r0):
    """This function finds the path of the newest .h5 file of TPSimulation.py for one
    Cn2 model and one D/r0 case.

    Searches config["input_dir"] for files named
    TPSimulation__model_<cn2_model>__Dr0_<D/r0 with 2 decimals>__*.h5. If there are
    several, the newest one is taken, by the date and time at the end of the file
    name (M and seed are not compared).
    Limitation: the file name holds the D/r0 that was reached. Runs made with
    target_D_over_r0 = None (unscaled profile) have an arbitrary D/r0 in the name
    and are only found if that exact value is asked for in config.

    Parameters
    ----------
    config : dict, analysis settings (uses "input_dir", "cn2_model")
    D_over_r0 : float, turbulence strength of the wanted run (e.g. 0, 2 or 8)

    Returns
    -------
    path : str, full path of the newest matching .h5 file

    Raises
    ------
    FileNotFoundError : if no file matches
    """

    pattern = os.path.join(config["input_dir"],
                           f"TPSimulation__model_{config['cn2_model']}__Dr0_{D_over_r0:.2f}__*.h5")   # same format as the file name in TPSimulation.py
    matching_paths = glob.glob(pattern)              # list of all matching files (unsorted)

    if len(matching_paths) == 0:                     # nothing found: stop with a clear message
        raise FileNotFoundError(f"No file for model {config['cn2_model']} and D/r0 = {D_over_r0} in {config['input_dir']}")

    path = sorted(matching_paths, key=lambda p: p.split("__")[-1])[-1]   # newest file = latest date_time at the end of the name
    print(f"  Data file: {os.path.basename(path)}")                   # basename: file name without the folder
    return path

def load_run(path):
    """Read the settings and all small datasets of one .h5 file of TPSimulation.py.

    The large datasets pupil_field and focal_field (M x N x N complex values, about
    2 GB each for M = 4000) are deliberately not loaded here; they are read one
    realisation at a time by the analysis functions.
    The nested dict cn2_params was stored flattened as attributes cn2_params_<name>
    (see run_simulation in TPSimulation.py); it is rebuilt here, so that
    settings["cn2_params"] can be passed directly to get_cn2_function.

    Parameters
    ----------
    path : str, path of the .h5 file (e.g. from find_run_file)

    Returns
    -------
    run : dict with
        "path"        : str, path of the file
        "settings"    : dict, all attributes (config of TPSimulation.py, created,
                        cn2_scale, D_over_r0) plus the rebuilt "cn2_params"
        "layer_heights_sorted_m"   : array, phase-screen heights, low -> high [m]
        "r0_layers_m" : array, r0 of each screen, same order [m]
        "r0_total_m"  : float, total r0 of all screens [m] (inf without turbulence)
        "pupil_x_m"   : array, pupil-grid coordinates [m] (y identical, square grid)
        "focal_x_m"   : array, focal-grid coordinates [m] (y identical, square grid)
        "pupil_power" : array (M,), total power in the aperture per realisation
        "focal_power" : array (M,), total power in the focal image per realisation
        "M"           : int, number of realisations
    """
    with h5py.File(path, "r") as f:                  # open read-only; closed again at the end of the block
        settings = dict(f.attrs)                     # all attributes: the config of TPSimulation.py

        run = {
            "path": path,
            "settings": settings,

            # layers
            "layer_heights_sorted_m": f["layer_heights_sorted_m"][:],          # [:] reads the whole dataset into a numpy array
            "r0_layers_m": f["r0_layers_m"][:],
            "r0_total_m": float(f["r0_total_m"][()]),   # scalar dataset (shape ()): [()] instead of [:]

            # grid coordinates (square grid, x = y)
            "pupil_x_m": f["pupil_x_m"][:],
            "focal_x_m": f["focal_x_m"][:],

            # total power, one value per realisation
            "pupil_power": f["pupil_power"][:],
            "focal_power": f["focal_power"][:],

            # number of realisations, from the shape only (no field data is read)
            "M": f["focal_field"].shape[0],
        }

    # rebuild cn2_params from dict
    cn2_params = {}
    for key, value in settings.items():
        if key.startswith("cn2_params_"):
            short_key = key.removeprefix("cn2_params_")
            cn2_params[short_key] = value
    settings["cn2_params"] = cn2_params

    return run

def number_of_realisations_used(run, config):
    """Number of realisations used in the analysis of one run.

    Parameters
    ----------
    run : dict from load_run (uses "M")
    config : dict, analysis settings (uses "n_realisations_max")

    Returns
    -------
    M_used : int, number of realisations used
    """
    if config["n_realisations_max"] is None:                 # None: use all realisations (from config)
        M_used = run["M"]
    else:
        M_used = min(config["n_realisations_max"], run["M"])
    return M_used

def load_realisation(run, k, optics=None):
    """Read the pupil and focal field of one realisation of one run.

    Only realisation k is read from the file, for the example figures.

    Parameters
    ----------
    run : dict from load_run (uses "path"; with optics also "focal_x_m" and "settings")
    k : int, index of the realisation, 0 ... M - 1
    optics : dict from make_optics, optional; if given, the focal field is
             tip-tilt corrected (focal_centroid, remove_tip_tilt); None (default): as stored

    Returns
    -------
    fields : dict with
        "pupil_field_k" : complex array (n_pupil x n_pupil), field in the aperture
        "focal_field_k" : complex array (n_focal x n_focal), field in the focal plane
                          (tip-tilt corrected if optics is given)
        "k"             : int, index of realisation used
    """
    with h5py.File(run["path"], "r") as f:
        pupil_field_k = f["pupil_field"][k].astype(np.complex128)
        focal_field_k = f["focal_field"][k].astype(np.complex128)

    # apply centeroid correction for 'Example realisation' figures
    if optics is not None:                                      # only if config["tip_tilt_correction"] is True
        centroid_m = focal_centroid(np.abs(focal_field_k) ** 2, run)
        focal_field_k = remove_tip_tilt(pupil_field_k, centroid_m, optics, run)

    return {"pupil_field_k": pupil_field_k,
            "focal_field_k": focal_field_k,
            "k": k}

# ---------------------------------------------------------------------------
# Save Figures and Files
# ---------------------------------------------------------------------------

def finish_figure(fig, config, name):
    """Save and/or show a finished figure, following the switches in config.

    Called at the end of every plot function. Figures that are not shown are closed.

    Parameters
    ----------
    fig : matplotlib Figure, the finished figure
    config : dict, analysis settings (uses "save_figures", "show_figures",
             "run_output_dir" (set in __main__), "figure_format")
    name : str, file name without extension, e.g. "atmosphere_cn2"; for figures of
           one run include D/r0 and M, so that files of different runs do not
           overwrite each other

    Returns
    -------
    None
    """
    if config["save_figures"]:
        os.makedirs(config["run_output_dir"], exist_ok=True)       # create folder if needed
        path = os.path.join(config["run_output_dir"], f"{name}.{config['figure_format']}")
        fig.savefig(path, dpi=300, bbox_inches="tight")         # dpi: print resolution; tight: no empty borders
        short_path = os.path.join(*path.split(os.sep)[-3:])
        print(f"\nFigure saved to {short_path}")

    if config["show_figures"]:
        plt.show()
    else:
        plt.close(fig)

def save_results(results, config):
    """Save the computed values of all runs and the analysis settings as results.json.

    Written to folder config["run_output_dir"].

    Parameters
    ----------
    results : dict, {D/r0: dict of computed values}, filled in __main__
    config : dict, analysis settings (uses "run_output_dir"); also stored in the file

    Returns
    -------
    None
    """
    output = {"created": datetime.now().isoformat(),
              "config": config,
              "runs": results}

    os.makedirs(config["run_output_dir"], exist_ok=True)
    path = os.path.join(config["run_output_dir"], "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=4)
    short_path = os.path.join(*path.split(os.sep)[-3:])
    print(f"\nResults saved to {short_path}")

# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------

def atmosphere_summary(run):
    """Describe the simulated atmosphere of one run: print a table and return the
    quantities for the atmosphere figure.

    This is a description, not an independent validation: r0 per layer, total r0
    and D/r0 are the values stored by TPSimulation.py. Only quantities that are not
    stored are computed here, with the same functions as in the simulation:
    the scaled Cn2 profile and the slab boundaries (slab_edges).

    Parameters
    ----------
    run : dict from load_run

    Returns
    -------
    atmosphere : dict with
        "cn2_function" : function h_m -> Cn2 [m^-2/3], profile as used in the run (scaled)
        "layer_heights_sorted_m"    : array, phase-screen heights, low -> high [m]
        "edges_m"      : array, slab boundaries, low -> high [m] (one more than layer_heights_sorted_m)
        "r0_layers_m"  : array, r0 of each screen [m] (inf without turbulence)
        "D_over_r0"    : float, D/r0 reached in the run
    """
    settings = run["settings"]

    cn2_function = get_cn2_function(settings, scale=settings["cn2_scale"])
    edges_m = slab_edges(run["layer_heights_sorted_m"], settings["h_ground_m"], settings["h_top_m"])
    D_over_r0 = settings["D_rx_m"] / run["r0_total_m"]

    # printed summary
    print(f"  D/r0 = {D_over_r0:.2f}, r0 = {run['r0_total_m'] * 100:.1f} cm, "
          f"cn2_scale = {settings['cn2_scale']:.3f}")
    print(f"  {'screen [m]':>10}  {'slab from - to [m]':>20}  {'r0 [cm]':>8}")
    for i, h in enumerate(run["layer_heights_sorted_m"]):
        print(f"  {h:10.0f}  {edges_m[i]:9.0f} - {edges_m[i + 1]:8.0f}  "
              f"{run['r0_layers_m'][i] * 100:8.1f}")

    return {"cn2_function": cn2_function,
            "cn2_model": settings["cn2_model"],
            "cn2_params": settings["cn2_params"],
            "layer_heights_sorted_m": run["layer_heights_sorted_m"],
            "edges_m": edges_m,
            "r0_layers_m": run["r0_layers_m"],
            "D_over_r0": D_over_r0}

def plot_atmosphere(atmospheres, config):
    """Cn2 profile against height for all turbulent cases in one plot, so that the
    cases can be compared, with the phase screens and the slab boundaries.

    The profile is the one used in each run: the Cn2 model and parameters stored in
    its settings (e.g. Hufnagel-Valley with HV5/7, Andrews & Phillips, 2005, Eq. 12.1,
    p.481), multiplied by its cn2_scale. Both axes are logarithmic; D/r0 = 0 is
    skipped (Cn2 = 0 cannot be shown on a log axis). Screens and slabs are the same
    in all runs; they are taken from the first case.

    Parameters
    ----------
    atmospheres : dict {D/r0: atmosphere dict from atmosphere_summary}
    config : dict, analysis settings (passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure ("atmosphere_cn2")
    """
    fig, ax = plt.subplots(figsize=(5, 6))

    # Cn2 profile of each turbulent case
    for D_over_r0, atmosphere in atmospheres.items():
        if D_over_r0 == 0:                                       # no turbulence: Cn2 = 0, not visible on a log axis
            continue
        h_top_m = atmosphere["edges_m"][-1]                      # last slab boundary = top of the atmosphere
        height_grid_m = np.logspace(0, np.log10(h_top_m), 1000)
        # build label for Cn2 model with parameters
        params_text = ", ".join(f"{key} = {value:.3g}"
                                for key, value in atmosphere["cn2_params"].items())
        label = f"{atmosphere['cn2_model']} ({params_text}), D/r0 = {D_over_r0:.0f}"  # to not print parameters, exclude "({params_text})"
        ax.plot(atmosphere["cn2_function"](height_grid_m), height_grid_m, label=label)

    # phase screens (solid) and slab boundaries (dashed), identical for all cases
    first = list(atmospheres.values())[0]                        # choose any since it's the same for all
    for i, h in enumerate(first["layer_heights_sorted_m"]):
        ax.axhline(h, color="grey", linewidth=0.8,
                   label="phase screen" if i == 0 else None)     # label only once, for the legend
    for i, edge in enumerate(first["edges_m"][1:]):              # skip the ground (0 m): not on a log axis
        ax.axhline(edge, color="grey", linewidth=0.8, linestyle="--",
                   label="slab boundary" if i == 0 else None)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$C_n^2$ [m$^{-2/3}$]")
    ax.set_ylabel("height above ground [m]")
    ax.set_xlim(1e-20, None)                                     # cut the negligible Cn2 near the top
    ax.set_title("Simulated Cn2 profiles")
    ax.legend()
    fig.tight_layout()
    finish_figure(fig, config, "atmosphere_cn2")

# ---------------------------------------------------------------------------
# Tip-tilt correction
# ---------------------------------------------------------------------------

def focal_centroid(I_focal, run):
    """Centre of gravity of one focal image.

    Tip-tilt is measured by "determining the center of gravity" of the focal
    image (Saathof & Zadeh, 2025, p.2). Computed with the AOtools function
    centre_of_gravity: x_c = sum(I x) / sum(I), y_c likewise, in pixel indices
    (origin at pixel 0, from the source code aotools/image_processing/centroiders.py);
    converted to metres with the focal grid of the run.

    Parameters
    ----------
    I_focal : array (n_focal x n_focal), focal intensity of one realisation, index [y, x]
    run : dict from load_run (uses "focal_x_m")

    Returns
    -------
    centroid_m : array (2,), (x_c, y_c) [m]
    """
    x_index, y_index = centre_of_gravity(I_focal)               # from AOtools

    x_m = run["focal_x_m"]
    pixel_m = x_m[1] - x_m[0]                                   # pixel size
    x_c = x_m[0] + x_index * pixel_m                            # convert pixel index into position
    y_c = x_m[0] + y_index * pixel_m

    return np.array([x_c, y_c])

def remove_tip_tilt(E_pupil, centroid_m, optics, run):
    """Focal field of one realisation with the tip-tilt removed.

    The pupil field is multiplied by the tilt phase
    phi(x, y) = -2 pi (x_c x + y_c y) / (lambda f), which shifts the focal field
    by -(x_c, y_c), i.e. moves the centre of gravity back to the axis: the lens
    is a Fourier transform (Goodman, 1996, Eq. 5.15, p.103) and a linear phase
    shifts the transform (shift theorem, Goodman, 1996, Eq. 2.13, p.8). It is then
    propagated again with the lens of the simulation (optics from make_optics of
    TPSimulation.py), so the result lies on the same focal grid.

    Parameters
    ----------
    E_pupil : array (N_px x N_px), complex pupil field of one realisation
              (after the aperture), index [y, x]
    centroid_m : array (2,), (x_c, y_c) [m] from focal_centroid
    optics : dict from make_optics (uses "pupil_grid", "prop_focal")
    run : dict from load_run (uses "settings": "wavelength_m", "focal_length_m")

    Returns
    -------
    E_focal : array (n_focal x n_focal), complex focal field after the correction, index [y, x]
    """
    settings = run["settings"]
    wavelength_m = settings["wavelength_m"]
    focal_length_m = settings["focal_length_m"]

    x_c, y_c = centroid_m
    pupil_field = hci.Field(E_pupil.ravel(), optics["pupil_grid"])      # 2D array to 1D array for HCIPy
    x = optics["pupil_grid"].x
    y = optics["pupil_grid"].y

    # tilted plane across the pupil, one value per pixel
    tilt_phase = -2 * np.pi * (x_c * x + y_c * y) / (wavelength_m * focal_length_m)
    # exp(i phi) changes only the phase, not the amplitude; hci.Wavefront is needed for optics["prop_focal"]
    wf_pupil = hci.Wavefront(pupil_field * np.exp(1j * tilt_phase), wavelength_m)

    wf_focal = optics["prop_focal"](wf_pupil)                   # same propagation as in TPSimulation.py

    return wf_focal.electric_field.shaped                       # back to a 2D array

def tip_tilt_texts(config):
    """Texts for the folder name and the figure titles, following
    config["tip_tilt_correction"].

    Parameters
    ----------
    config : dict, analysis settings (uses "tip_tilt_correction")

    Returns
    -------
    texts : dict with
        "short" : str, "on" or "off" (output folder name)
        "title" : str, "with tip-tilt correction" or "without tip-tilt correction"
    """
    if config["tip_tilt_correction"]:
        return {"short": "on", "title": "with tip-tilt correction"}
    return {"short": "off", "title": "without tip-tilt correction"}

# ---------------------------------------------------------------------------
# Statistics over the realisations
# ---------------------------------------------------------------------------

def accumulate_intensity_sums(run, M_used, mask, cdf_pixels=None, optics=None):
    """Sum the intensity I and I^2 per pixel over the realisations of one run, for
    the pupil and the focal plane.

    The fields are read one realisation at a time, so the memory use does not
    depend on M. Mean and scintillation index follow from these sums
    (mean_intensity, scintillation_index). After each realisation, the pupil
    scintillation index averaged over the aperture mask is also stored, to show
    its convergence with the number of realisations. If cdf_pixels is given, the
    focal intensity at these pixels is also kept for every realisation, for the
    irradiance CDFs.

    Parameters
    ----------
    run : dict from load_run
    M_used : int, number of realisations used (from number_of_realisations_used)
    mask : boolean array (N_px x N_px) from aperture_mask
    cdf_pixels : tuple (i_0, j) from cdf_pixel_indices, optional;
                 None (default): no CDF samples
    optics : dict from make_optics, optional; if given, every focal image is
             tip-tilt corrected (focal_centroid, remove_tip_tilt) before it is used;
             None (default): no correction

    Returns
    -------
    sums : dict with
        "sum_I_pupil"      : array (N_px x N_px), sum of I over the realisations, per pixel
        "sum_I2_pupil"     : array (N_px x N_px), sum of I^2, per pixel
        "sum_I_focal"      : array (n_focal x n_focal), sum of I, per pixel
        "sum_I2_focal"     : array (n_focal x n_focal), sum of I^2, per pixel
        "sigma_I2_pupil_m" : array (M_used,), pupil sigma_I^2 over the mask after
                             1, 2, ..., M_used realisations
        "I_focal_cdf"      : array (M_used, n_cdf), focal intensity at the CDF pixels
                             per realisation; None if cdf_pixels is None
        "centroids_m"      : array (M_used, 2), centre of gravity (x_c, y_c) [m] of
                             each focal image before the correction; None without correction
    """
    sigma_I2_pupil_m = np.zeros(M_used)                         # one value per number of realisations

    if cdf_pixels is not None:                                  # CDF samples only if pixels are given
        i_0, j = cdf_pixels
        I_focal_cdf = np.zeros((M_used, len(j)))                # one row per realisation, one column per pixel
    else:
        I_focal_cdf = None

    centroids_m = np.zeros((M_used, 2)) if optics is not None else None   # centre of gravity per realisation, only with correction

    with h5py.File(run["path"], "r") as f:
        pupil_field = f["pupil_field"]                          # link into the file, no data read yet
        focal_field = f["focal_field"]

        # one running sum per pixel (shape of one image), starting at 0
        sum_I_pupil = np.zeros(pupil_field.shape[1:])           # pupil_field.shape = (M, N, N)
        sum_I2_pupil = np.zeros(pupil_field.shape[1:])
        sum_I_focal = np.zeros(focal_field.shape[1:])
        sum_I2_focal = np.zeros(focal_field.shape[1:])

        for k in range(M_used):
            E_pupil = pupil_field[k].astype(np.complex128)      # read realisation k only; complex128 against rounding
            I_pupil = np.abs(E_pupil) ** 2

            sum_I_pupil += I_pupil
            sum_I2_pupil += I_pupil ** 2

            # pupil sigma_I^2 over the mask after k + 1 realisations
            sigma_I2_pupil_m[k] = np.mean(scintillation_index(
                sum_I_pupil[mask], sum_I2_pupil[mask], k + 1))

            E_focal = focal_field[k].astype(np.complex128)      # stored focal field
            if optics is not None:                              # only if config["tip_tilt_correction"] is True
                centroids_m[k] = focal_centroid(np.abs(E_focal) ** 2, run)          # centre of gravity is measured
                E_focal = remove_tip_tilt(E_pupil, centroids_m[k], optics, run)     # tilt removed, propagated again to the focal plane
            I_focal = np.abs(E_focal) ** 2

            sum_I_focal += I_focal
            sum_I2_focal += I_focal ** 2

            if cdf_pixels is not None:
                I_focal_cdf[k] = I_focal[i_0, j]                # row i_0 (y = 0), columns j (the x positions)

            if (k + 1) % 500 == 0 or k + 1 == M_used:           # progress every 500 realisations and at the end
                print(f"\r  realisation {k + 1}/{M_used} summed", end="", flush=True)
        print()

    return {"sum_I_pupil": sum_I_pupil, "sum_I2_pupil": sum_I2_pupil,
            "sum_I_focal": sum_I_focal, "sum_I2_focal": sum_I2_focal,
            "sigma_I2_pupil_m": sigma_I2_pupil_m,
            "I_focal_cdf": I_focal_cdf,
            "centroids_m": centroids_m}

def mean_intensity(sum_I, M):
    """Mean intensity per pixel, <I> = sum_I / M.

    The ensemble average <.> is estimated by the mean over M independent, identically
    distributed realisations (Schmidt, 2010, p.179), summed in a loop as in
    Schmidt (2010, Listing 9.8, p.181).

    Parameters
    ----------
    sum_I : array, sum of I over the realisations (from accumulate_intensity_sums)
    M : int, number of realisations in the sum

    Returns
    -------
    array, mean intensity per pixel
    """
    return sum_I / M

def scintillation_index(sum_I, sum_I2, M):
    """Scintillation index per pixel, sigma_I^2 = <I^2> / <I>^2 - 1
    (Kaushal & Kaddoum, 2017, Eq. 10, p.66). The ensemble averages <.> are estimated
    by the means over the realisations, as in mean_intensity (Schmidt, 2010, p.179).

    Pixels without light (<I> = 0) give nan instead of a division-by-zero warning.

    Parameters
    ----------
    sum_I, sum_I2 : arrays, sums of I and I^2 over the realisations
    M : int, number of realisations in the sums

    Returns
    -------
    array, scintillation index per pixel (nan where <I> = 0)
    """
    with np.errstate(divide="ignore", invalid="ignore"):        # 0/0 outside the light -> nan, no warning
        return (sum_I2 / M) / (sum_I / M) ** 2 - 1

# ---------------------------------------------------------------------------
# Pupil statistics
# ---------------------------------------------------------------------------

def aperture_mask(run):
    """Pixels fully inside the receiver aperture.

    A pixel counts as inside if its farthest corner is inside the circle:
    r_centre + dx/sqrt(2) <= D/2. Assumes a square grid (y identical to x).

    Parameters
    ----------
    run : dict from load_run

    Returns
    -------
    mask : boolean array (N x N), True for pixels fully inside the aperture
    """
    x_m = run["pupil_x_m"]                       # pixel-centre coordinates
    dx_m = x_m[1] - x_m[0]                       # pupil pixel size
    D_rx_m = run["settings"]["D_rx_m"]           # receiver aperture diameter

    X_m, Y_m = np.meshgrid(x_m, x_m)             # 2D coordinates
    r_centre_m = np.hypot(X_m, Y_m)              # distance of every pixel centre from the axis
    # comparison on the whole array -> boolean mask
    return r_centre_m + dx_m / np.sqrt(2) <= D_rx_m / 2

def rytov_variance(run, atmosphere):
    """Rytov variance sigma_R^2 of a plane-wave downlink (weak-fluctuation theory).

    sigma_R^2 = 2.25 k^(7/6) sec^(11/6)(zenith) * integral from h0 to H of
                Cn2(h) (h - h0)^(5/6) dh
    (Andrews & Phillips, 2005, Eq. 12.38, p.495), valid for a downlink at H >> 20 km.
    In weak turbulence this equals the scintillation index sigma_I^2 at a point in
    the receiver plane. Here h0 = h_ground_m, and the integral stops at
    h_top_m instead of the satellite altitude (Cn2 is negligible above).

    Parameters
    ----------
    run : dict from load_run
    atmosphere : dict from atmosphere_summary (uses "cn2_function", scaled profile)

    Returns
    -------
    sigma_R2 : float, Rytov variance
    """
    settings = run["settings"]
    k_per_m = 2 * np.pi / settings["wavelength_m"]          # wavenumber
    zenith_rad = np.radians(settings["zenith_deg"])
    h_ground_m = settings["h_ground_m"]
    h_top_m = settings["h_top_m"]

    h_m = np.linspace(h_ground_m, h_top_m, 100_000)          # height grid
    integral = np.trapezoid(atmosphere["cn2_function"](h_m) * (h_m - h_ground_m) ** (5 / 6), h_m)
    sec = 1 / np.cos(zenith_rad)                             # secant of the zenith angle
    sigma_R2 = 2.25 * k_per_m ** (7 / 6) * sec ** (11 / 6) * integral
    return sigma_R2

def scintillation_index_strong(sigma_R2):
    """Scintillation index for moderate-to-strong irradiance fluctuations.

    sigma_I^2 = exp[0.49 sigma_R^2 / (1 + 1.11 sigma_R^(12/5))^(7/6)
                    + 0.51 sigma_R^2 / (1 + 0.69 sigma_R^(12/5))^(5/6)] - 1
    (Andrews & Phillips, 2005, Eq. 12.40, p.497).
    Scintillation index in the centre of the beam on a downlink channel with a
    point receiver, computed from the Rytov variance.

    Parameters
    ----------
    sigma_R2 : float, Rytov variance from rytov_variance

    Returns
    -------
    sigma_I2 : float, scintillation index
    """
    term_1 = 0.49 * sigma_R2 / (1 + 1.11 * sigma_R2 ** (6 / 5)) ** (7 / 6)
    term_2 = 0.51 * sigma_R2 / (1 + 0.69 * sigma_R2 ** (6 / 5)) ** (5 / 6)
    sigma_I2 = np.exp(term_1 + term_2) - 1
    return sigma_I2

def plot_scintillation_convergence(sigma_I2_m, sigma_R2, D_over_r0, config):
    """Pupil scintillation index against the number of realisations.

    Shows how the pupil scintillation index over the aperture mask changes as
    realisations are added, compared with the weak-fluctuation value (Andrews &
    Phillips, 2005, Eq. 12.38, p.495) and the moderate-to-strong value (Eq. 12.40,
    p.497).

    Parameters
    ----------
    sigma_I2_m : array (M_used,), from accumulate_intensity_sums ("sigma_I2_pupil_m")
    sigma_R2 : float, Rytov variance from rytov_variance
    D_over_r0 : float, turbulence strength of the run (for title and file name)
    config : dict, analysis settings (passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure
    """
    m = np.arange(1, len(sigma_I2_m) + 1)                     # number of realisations (until M_used)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(m, sigma_I2_m, marker=".", label="simulation")    # simulated trace
    ax.axhline(sigma_R2, color="grey", linestyle="--",
               label="theory, weak turbulence")               # horizontal line at sigma_R^2 (weak)
    ax.axhline(scintillation_index_strong(sigma_R2), color="grey", linestyle=":",
               label="theory, moderate-to-strong turbulence") # horizontal line at sigma_I^2 (moderate - strong)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(1, len(sigma_I2_m))
    ax.set_xlabel("number of realisations M")
    ax.set_ylabel(r"pupil $\sigma_I^2$ (mean over aperture)")
    ax.set_title(f"Convergence of the pupil scintillation index, D/r0 = {D_over_r0}")
    ax.legend()
    fig.tight_layout()
    finish_figure(fig, config, f"pupil_scintillation_convergence_Dr0_{D_over_r0}")

def power_scintillation_index(run, M_used):
    """Scintillation index of the total received power from the simulation.

    sigma_P^2 = (<P^2> - <P>^2) / <P>^2, with P the total power in the aperture of
    each realisation (pupil_power). Andrews and Phillips call this the irradiance
    flux variance (Andrews & Phillips, 2005, Eq. 10.54, p.410).

    Parameters
    ----------
    run : dict from load_run (uses "pupil_power")
    M_used : int, number of realisations used

    Returns
    -------
    sigma_P2 : float, scintillation index of the total received power
    """
    P = run["pupil_power"][:M_used]                  # total pupil power of the realisations used
    sigma_P2 = scintillation_index(np.sum(P), np.sum(P ** 2), M_used)
    return sigma_P2

def irradiance_flux_variance(run, atmosphere):
    """Irradiance flux variance of a plane-wave downlink (theory).

    Irradiance flux variance:
    sigma_I^2(D_G) = 8.70 k^(7/6) (H - h0)^(5/6) sec^(11/6)(zenith) * Re integral from h0 to H of
                     Cn2(h) [(k D_G^2 / (16 L) + i (h - h0) / (H - h0))^(5/6) - (k D_G^2 / (16 L))^(5/6)] dh
    (Andrews & Phillips, 2005, Eq. 12.39, p.496), with L = (H - h0) sec(zenith)
    (Andrews & Phillips, 2005, p.490). Here D_G = D_rx_m, h0 = h_ground_m and
    H = h_top_m. Compared with power_scintillation_index.

    Parameters
    ----------
    run : dict from load_run
    atmosphere : dict from atmosphere_summary (uses "cn2_function", scaled profile)

    Returns
    -------
    sigma_P2 : float, irradiance flux variance (scintillation index of the total received power)
    """
    settings = run["settings"]
    k_per_m = 2 * np.pi / settings["wavelength_m"]          # wavenumber
    zenith_rad = np.radians(settings["zenith_deg"])
    h_ground_m = settings["h_ground_m"]
    H_m = settings["h_top_m"]
    D_rx_m = settings["D_rx_m"]

    h_m = np.linspace(h_ground_m, H_m, 100_000)              # height grid
    sec = 1 / np.cos(zenith_rad)
    L_m = (H_m - h_ground_m) * sec                           # path length

    a = k_per_m * D_rx_m ** 2 / (16 * L_m)
    xi = (h_m - h_ground_m) / (H_m - h_ground_m)

    integrand = atmosphere["cn2_function"](h_m) * np.real((a + 1j * xi) ** (5 / 6) - a ** (5 / 6))
    sigma_P2 = 8.70 * k_per_m ** (7 / 6) * (H_m - h_ground_m) ** (5 / 6) * sec ** (11 / 6) * np.trapezoid(integrand, h_m)

    return sigma_P2

# ---------------------------------------------------------------------------
# Focal-plane statistics
# ---------------------------------------------------------------------------

def strehl_ratio_long_exposure(D_over_r0):
    """Long-exposure Strehl ratio from theory.

    SR = 1 / [1 + (D/r0)^(5/3)]^(6/5)
    (Andrews & Phillips, 2005, Eq. 10.50, p.408; Schmidt, 2010, Eq. 9.57, p.162),
    an approximation of Fried's integral for all D/r0. Long exposure: the mean
    image over many realisations, without tip-tilt correction (Schmidt, 2010, p.161).

    Parameters
    ----------
    D_over_r0 : float, aperture diameter over the plane-wave Fried parameter

    Returns
    -------
    SR : float, long-exposure Strehl ratio
    """
    SR = 1 / (1 + D_over_r0 ** (5 / 3)) ** (6 / 5)
    return SR

def strehl_ratio_fried(D_over_r0, alpha):
    """Strehl ratio from theory, with or without tip-tilt correction.

    SR = 16/pi * integral from 0 to 1 of f' [arccos(f') - f' sqrt(1 - f'^2)]
         * exp{-3.44 (f' D/r0)^(5/3) [1 - alpha f'^(1/3)]} df'
    (Schmidt, 2010, Eq. 9.56, p.162), with Fried's average MTF of the atmosphere
    (Eq. 9.54, p.161) and f' the spatial frequency normalised to the cutoff.
    alpha (Schmidt, 2010, Eq. 9.55, p.161): 0 long exposure, 1 short exposure
    without scintillation, 1/2 short exposure with scintillation. Evaluated
    numerically with scipy.integrate.quad.

    Parameters
    ----------
    D_over_r0 : float, aperture diameter over the plane-wave Fried parameter
    alpha : float, 0, 1/2 or 1

    Returns
    -------
    SR : float, Strehl ratio
    """
    def integrand(f):
        aperture_term = f * (np.arccos(f) - f * np.sqrt(1 - f ** 2))
        atmosphere_term = np.exp(-3.44 * (f * D_over_r0) ** (5 / 3) * (1 - alpha * f ** (1 / 3)))
        return aperture_term * atmosphere_term

    integral, _ = quad(integrand, 0, 1)
    SR = 16 / np.pi * integral
    return SR

def strehl_ratio(mean_focal_images, D_over_r0, run):
    """Strehl ratio of the simulated mean focal image.

    SR = <I(0)> / I_0(0): on-axis mean intensity with turbulence over the on-axis
    (mean) intensity without turbulence (Andrews & Phillips, 2005, p.407). The run
    with D/r0 = 0 is the reference; without it, nan is returned. The on-axis
    pixel is the one at focal_x_m = 0. Assumes a square focal grid (y identical to x).

    Parameters
    ----------
    mean_focal_images : dict {D/r0: array (n_focal x n_focal)}, mean focal intensity
                        of each run, same grid; must contain D/r0 = 0 as reference
    D_over_r0 : float, turbulence strength of the run (key in mean_focal_images)
    run : dict from load_run (uses "focal_x_m")

    Returns
    -------
    SR : float, Strehl ratio of the simulation (nan if there is no D/r0 = 0 run)
    """
    if 0 not in mean_focal_images:                            # no reference without turbulence
        print("  No D/r0 = 0 run: simulated Strehl ratio cannot be computed.")
        return np.nan

    i_0 = np.argmin(np.abs(run["focal_x_m"]))                 # index of the pixel closest to x = 0 (on axis)
    SR = mean_focal_images[D_over_r0][i_0, i_0] / mean_focal_images[0][i_0, i_0]
    return SR

def focal_power_fraction(run, M_used):
    """Fraction of the pupil power that falls on the focal grid.

    Ratio focal_power / pupil_power per realisation. Light outside the focal grid
    is lost, so the ratio checks whether the grid is large enough. Besides the mean,
    the smallest ratio is returned: a single strongly spread realisation is hidden
    in the mean.

    Parameters
    ----------
    run : dict from load_run (uses "focal_power", "pupil_power")
    M_used : int, number of realisations used

    Returns
    -------
    fraction : dict with
        "mean"    : float, mean of the ratios over the realisations
        "minimum" : float, smallest ratio (realisation with the largest loss)
    """
    focal_power = run["focal_power"][:M_used]
    pupil_power = run["pupil_power"][:M_used]

    ratio = focal_power / pupil_power

    return {"mean": np.mean(ratio),
            "minimum": np.min(ratio)}

def plot_focal_scintillation_maps(scint_focal_maps, runs, config):
    """Maps of the focal-plane scintillation index, one panel per run.

    sigma_I^2(x, y) = <I^2> / <I>^2 - 1 per focal pixel (Saathof & Zadeh, 2025, Eq. 1,
    p.6; Kaushal & Kaddoum, 2017, Eq. 10, p.66), to compare with Saathof and Zadeh
    (2025, Fig. 3, p.5). With or without tip-tilt correction, following config["tip_tilt_correction"].
    Axes in units of lambda f / D. D/r0 = 0 is kept as a check (sigma_I^2 = 0).

    Parameters
    ----------
    scint_focal_maps : dict {D/r0: array (n_focal x n_focal)}, sigma_I^2 per focal pixel
    runs : dict {D/r0: run dict from load_run} (uses "settings", "focal_x_m", "M_used")
    config : dict, analysis settings (passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure
    """
    n_cases = len(scint_focal_maps)
    fig, axes = plt.subplots(1, n_cases, figsize=(4.5 * n_cases, 4))
    axes = np.atleast_1d(axes)                                   # also a list for a single case

    for i, (ax, (D_over_r0, scint_focal_map)) in enumerate(zip(axes, scint_focal_maps.items())):
        run = runs[D_over_r0]
        settings = run["settings"]
        lambda_f_over_D_m = settings["wavelength_m"] * settings["focal_length_m"] / settings["D_rx_m"]
        x_lD = run["focal_x_m"] / lambda_f_over_D_m              # focal coordinates in units of lambda f / D

        im = ax.imshow(scint_focal_map, origin="lower", vmin=0, vmax=2,
                       extent=[x_lD[0], x_lD[-1], x_lD[0], x_lD[-1]])
        fig.colorbar(im, ax=ax, label=r"$\sigma_I^2$")

        ax.set_xlim(-10, 10)                                     # zoom
        ax.set_ylim(-10, 10)
        ax.set_xlabel(r"$x$ [$\lambda f / D$]")
        ax.set_ylabel(r"$y$ [$\lambda f / D$]")
        ax.set_title(f"({'abcdefgh'[i]}) D/r0 = {D_over_r0}, M = {run['M_used']}")   # panel letter

    fig.suptitle(f"Focal-plane scintillation index, {tip_tilt_texts(config)['title']}")
    fig.tight_layout()
    finish_figure(fig, config, "focal_scintillation_maps")

def cdf_pixel_indices(run, config):
    """Focal pixels for the irradiance CDFs: along x, on the row y = 0.

    For each position in config["cdf_x_m"] the pixel with the closest centre is
    taken, as in Saathof and Zadeh (2025, Fig. 4, p.5). Assumes a square
    focal grid (y identical to x).

    Parameters
    ----------
    run : dict from load_run (uses "focal_x_m")
    config : dict, analysis settings (uses "cdf_x_m")

    Returns
    -------
    (i_0, j) : tuple, or None if config["cdf_x_m"] is None
        i_0 : int, row index of y = 0 (on axis)
        j : array of int, column index of each position in config["cdf_x_m"]
    """
    if config["cdf_x_m"] is None:
        return None

    x_m = run["focal_x_m"]
    i_0 = np.argmin(np.abs(x_m))                                   # row of y = 0 (on axis)
    j = [np.argmin(np.abs(x_m - x)) for x in config["cdf_x_m"]]    # column closest to each wanted x
    return i_0, np.array(j)

def irradiance_cdf(I_samples):
    """Empirical irradiance CDF at the CDF pixels.

    The samples of each pixel are normalised by the mean of that pixel and sorted;
    the k-th smallest value gets the probability k / M. Normalisation as in Saathof
    and Zadeh (2025, Fig. 4, p.5), assumed to be by the pixel's own mean
    although not checked.

    Parameters
    ----------
    I_samples : array (M, n_cdf), focal intensity at the CDF pixels per realisation
                ("I_focal_cdf" from accumulate_intensity_sums)

    Returns
    -------
    I_sorted : array (M, n_cdf), normalised intensity, sorted per pixel (column)
    probability : array (M,), cumulative probability 1/M ... 1, the same for all pixels
    """
    I_normalised = I_samples / np.mean(I_samples, axis=0)
    I_sorted = np.sort(I_normalised, axis=0)

    # probabilities
    M = I_samples.shape[0]
    probability = np.arange(1, M + 1) / M

    return I_sorted, probability

def plot_irradiance_cdfs(cdfs, runs, config):
    """Irradiance CDFs at the focal pixels, one panel per turbulent run.

    One curve per position in config["cdf_x_m"], to compare with Saathof and
    Zadeh (2025, Fig. 4, p.5): log-log axes with their ranges, normalised
    irradiance 1e-3 to 10 and probability 1e-3 to 1. The title states whether
    tip-tilt correction is applied (their figure has it on). D/r0 = 0 is skipped
    (one realisation, no distribution).

    Parameters
    ----------
    cdfs : dict {D/r0: (I_sorted, probability)} from irradiance_cdf
    runs : dict {D/r0: run dict from load_run} (uses "M_used")
    config : dict, analysis settings (uses "cdf_x_m", "tip_tilt_correction";
             passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure
    """
    turbulent_cdfs = {D_over_r0: cdf for D_over_r0, cdf in cdfs.items() if D_over_r0 > 0}               # dict comprehension

    fig, axes = plt.subplots(1, len(turbulent_cdfs), figsize=(5 * len(turbulent_cdfs), 4))
    axes = np.atleast_1d(axes)

    for i, (ax, (D_over_r0, (I_sorted, probability))) in enumerate(zip(axes, turbulent_cdfs.items())):  # loop: over cases
        for n, x_m in enumerate(config["cdf_x_m"]):                                                     # loop: one curve per pixel position
            ax.loglog(I_sorted[:, n], probability, label=f"x = {x_m * 1e6:.1f} µm")

        ax.set_xlim(1e-3, 1e1)
        ax.set_ylim(1e-3, 1)
        ax.set_xlabel("normalised irradiance")
        ax.set_ylabel("probability")
        ax.grid(True, which="both")
        ax.legend()
        ax.set_title(f"({'abcdefgh'[i]}) D/r0 = {D_over_r0}, M = {runs[D_over_r0]['M_used']}")

    fig.suptitle(f"Irradiance CDFs at the focal pixels, {tip_tilt_texts(config)['title']}")
    fig.tight_layout()
    finish_figure(fig, config, "focal_irradiance_cdfs")

# ---------------------------------------------------------------------------
# Example realisations
# ---------------------------------------------------------------------------

def plot_example_realisations(examples, runs, config):
    """Pupil and focal plane of one realisation per run, one row per run.

    Columns: |E|^2 and np.angle(E) in the pupil (nan outside aperture_mask), focal
    intensity linear and logarithmic. The focal intensity is divided by the on-axis
    intensity of the D/r0 = 0 image (as for the Strehl ratio, Andrews & Phillips, 2005,
    p.407), so the run with D/r0 = 0 is needed. The pupil intensity has one colour
    scale for all runs. With tip-tilt correction, the focal panels are corrected and
    the pupil panels show the field before the tip-tilt mirror.

    Parameters
    ----------
    examples : dict {D/r0: fields dict from load_realisation}
    runs : dict {D/r0: run dict from load_run} (uses "settings", "pupil_x_m", "focal_x_m")
    config : dict, analysis settings (uses "example_realisation", "tip_tilt_correction"; passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure
    """
    if 0 not in examples:
        print("  No D/r0 = 0 run: example realisations are not plotted.")
        return

    # reference is centre pixel of case D_over_r0 = 0
    i_0 = np.argmin(np.abs(runs[0]["focal_x_m"]))  # index closest to x = 0
    I_focal_ref = np.abs(examples[0]["focal_field_k"][i_0, i_0]) ** 2

    # maximum pupil intensity of all cases inside aperture
    I_pupil_max = max(np.max(np.abs(fields["pupil_field_k"][aperture_mask(runs[D_over_r0])]) ** 2)
                      for D_over_r0, fields in examples.items())

    letters = "abcdefghijklmnopqrstuvwxyz"  # panel letters
    fig, axes = plt.subplots(len(examples), 4, figsize=(18, 4 * len(examples)), squeeze=False)

    for row, (D_over_r0, fields) in enumerate(examples.items()):  # loops over runs
        run = runs[D_over_r0]
        settings = run["settings"]
        mask = aperture_mask(run)

        x_m = run["pupil_x_m"]
        pupil_extent = [x_m[0], x_m[-1], x_m[0], x_m[-1]]  # image edges for imshow
        lambda_f_over_D_m = settings["wavelength_m"] * settings["focal_length_m"] / settings["D_rx_m"]
        x_lD = run["focal_x_m"] / lambda_f_over_D_m  # convert focal coordinates to units of lambda f / D
        focal_extent = [x_lD[0], x_lD[-1], x_lD[0], x_lD[-1]]

        I_pupil = np.where(mask, np.abs(fields["pupil_field_k"]) ** 2, np.nan)
        phase_rad = np.where(mask, np.angle(fields["pupil_field_k"]), np.nan)  # np.angle gives phase of each complex value in rad
        I_focal_norm = np.abs(fields["focal_field_k"]) ** 2 / I_focal_ref

        ax = axes[row]
        # pupil panel
        im = ax[0].imshow(I_pupil, origin="lower", extent=pupil_extent, vmin=0, vmax=I_pupil_max)
        fig.colorbar(im, ax=ax[0], label="intensity (plane wave = 1)")
        # phase panel
        im = ax[1].imshow(phase_rad, origin="lower", extent=pupil_extent, cmap="twilight",
                          vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=ax[1], label="phase [rad]")
        # linear focal panel
        im = ax[2].imshow(I_focal_norm, origin="lower", extent=focal_extent, cmap="hot", vmin=0)
        fig.colorbar(im, ax=ax[2], label=r"$I / I_0(0)$")
        # log focal panel
        im = ax[3].imshow(I_focal_norm, origin="lower", extent=focal_extent, cmap="hot",
                          norm=LogNorm(vmin=1e-6, vmax=1))
        fig.colorbar(im, ax=ax[3], label=r"$I / I_0(0)$")

        phase_title = "pupil phase (before tip-tilt mirror)" if config["tip_tilt_correction"] else "pupil phase"
        titles = ["pupil intensity", phase_title, "focal intensity", "focal intensity (log)"]
        for col in range(4):
            ax[col].set_title(f"({letters[4 * row + col]}) {titles[col]}")
        ax[0].text(-0.35, 0.5, f"D/r0 = {D_over_r0}\nk = {fields['k']}", transform=ax[0].transAxes,
                   rotation=90, ha="center", va="center", fontsize=16, fontweight="bold")   # row heading
        for col in [0, 1]:
            ax[col].set_xlim(-settings["D_rx_m"] / 2, settings["D_rx_m"] / 2)  # zoom pupil panel to +- D / 2
            ax[col].set_ylim(-settings["D_rx_m"] / 2, settings["D_rx_m"] / 2)
            ax[col].set_xlabel(r"$x$ [m]")
            ax[col].set_ylabel(r"$y$ [m]")
        ax[2].set_xlim(-10, 10)  # zoom linear focal panel to +- 10 lambda f / D
        ax[2].set_ylim(-10, 10)
        for col in [2, 3]:
            ax[col].set_xlabel(r"$x$ [$\lambda f / D$]")
            ax[col].set_ylabel(r"$y$ [$\lambda f / D$]")

    fig.suptitle(f"Single realisations, {tip_tilt_texts(config)['title']}")
    fig.tight_layout()
    finish_figure(fig, config, f"example_realisations_k_{config['example_realisation']}")

def airy_disk(x_lD):
    """Normalised Airy pattern of a circular aperture (theory).

    I(x) / I(0) = [2 J1(pi x) / (pi x)]^2 with x = D r / (lambda f), the focal radius in
    units of lambda f / D (Goodman, 1996, Eq. 4.31, p.77, normalised as in Table 4.1,
    p.78; focal plane of a lens with z = f, p.103). Set to 1 at x = 0.

    Parameters
    ----------
    x_lD : array, focal radius in units of lambda f / D

    Returns
    -------
    array, I / I(0)
    """
    x_lD = np.asarray(x_lD, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = 2 * j1(np.pi * x_lD) / (np.pi * x_lD)
    amplitude = np.where(x_lD == 0, 1.0, amplitude)  # case where x = 0

    return amplitude ** 2

def plot_airy_check(examples, runs, config):
    """Cut through the D/r0 = 0 focal image at y = 0, compared with airy_disk.

    The cut is divided by its on-axis value and shown for x >= 0 on a log axis, as
    the comparison of simulation and analytic result in Schmidt (2010, Fig. 4.1, p.58).

    Parameters
    ----------
    examples : dict {D/r0: fields dict from load_realisation}; must contain D/r0 = 0
    runs : dict {D/r0: run dict from load_run} (uses "settings", "focal_x_m")
    config : dict, analysis settings (passed on to finish_figure)

    Returns
    -------
    None; the figure is saved and/or shown by finish_figure
    """
    if 0 not in examples:  # case for no D_over_r0 = 0
        print("  No D/r0 = 0 run: Airy check is not plotted.")
        return

    run = runs[0]  # run with D/r0 = 0
    settings = run["settings"]
    lambda_f_over_D_m = settings["wavelength_m"] * settings["focal_length_m"] / settings["D_rx_m"]
    x_lD = run["focal_x_m"] / lambda_f_over_D_m

    i_0 = np.argmin(np.abs(run["focal_x_m"]))  # index of x = 0 (on axis)

    # cut through the image
    I_focal = np.abs(examples[0]["focal_field_k"]) ** 2
    I_cut = I_focal[i_0, :] / I_focal[i_0, i_0]  # cut along x-axis, y = 0

    x_theory_lD = np.linspace(0, x_lD[-1], 5000)  # grid for theory curve

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(x_lD[i_0:], I_cut[i_0:], "o", markersize=3, label="simulation, D/r0 = 0")
    ax.semilogy(x_theory_lD, airy_disk(x_theory_lD), label="Airy disk (theory)")

    ax.set_xlim(0, 10)  # zoom to 0-10 lambda f / D
    ax.set_ylim(1e-6, 1.5)

    ax.set_xlabel(r"$x$ [$\lambda f / D$], $y = 0$")
    ax.set_ylabel(r"$I / I(0)$")
    ax.set_title("Focal plane without turbulence: simulation and Airy disk")
    ax.legend()
    fig.tight_layout()
    finish_figure(fig, config, "focal_airy_check")

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n=== Atmosphere ===")
    atmospheres = {}                                   # {D/r0: atmosphere dict}
    runs = {}                                          # {D/r0: run}, read once, used in all parts
    results = {}                                       # {D/r0: computed values}, saved to results.json
    for D_over_r0 in config["D_over_r0_cases"]:
        print(f"\n--- D/r0 = {D_over_r0} ---")
        run = load_run(find_run_file(config, D_over_r0))
        runs[D_over_r0] = run
        run["M_used"] = number_of_realisations_used(run, config)   # stored in the run, used in the folder name and all parts
        run["optics"] = make_optics(run["settings"]) if config["tip_tilt_correction"] else None   # grids and lens of the run's simulation; None without correction
        atmospheres[D_over_r0] = atmosphere_summary(run)
        results[D_over_r0] = {"input_file": os.path.basename(run["path"]),
                              "cn2_model": str(run["settings"]["cn2_model"]),
                              "M": int(run["M"]),
                              "D_over_r0": float(atmospheres[D_over_r0]["D_over_r0"]),
                              "r0_total_m": float(run["r0_total_m"])}

    # output folder of this analysis run, named after the runs actually read
    models = sorted({str(run["settings"]["cn2_model"]) for run in runs.values()})
    model_text = "_".join(models)
    D_over_r0_text = "_".join(f"{D_over_r0:g}" for D_over_r0 in config["D_over_r0_cases"])
    M_max = max(run["M_used"] for run in runs.values())
    tip_tilt_text = tip_tilt_texts(config)["short"]
    config["run_output_dir"] = os.path.join(config["output_dir"], f"Analysis_Run__model_{model_text}__Dr0_{D_over_r0_text}__M_{M_max}__tiptilt_{tip_tilt_text}__{datetime.now():%Y%m%d_%H%M}")
    print(f"\nOutput folder: {config['run_output_dir']}")

    plot_atmosphere(atmospheres, config)

    print("\n=== Pupil statistics ===")
    all_sums = {}                                      # {D/r0: sums}, kept for the focal plane
    for D_over_r0 in config["D_over_r0_cases"]:
        print(f"\n--- D/r0 = {D_over_r0} ---")
        run = runs[D_over_r0]
        M_used = run["M_used"]
        mask = aperture_mask(run)
        cdf_pixels = cdf_pixel_indices(run, config)
        sums = accumulate_intensity_sums(run, M_used, mask, cdf_pixels, run["optics"])
        scint_pupil = scintillation_index(sums["sum_I_pupil"], sums["sum_I2_pupil"], M_used)
        pupil_sigma_I2 = np.mean(scint_pupil[mask])
        sigma_R2 = rytov_variance(run, atmospheres[D_over_r0])
        sigma_I2_strong = scintillation_index_strong(sigma_R2)

        print(f"  M_used = {M_used}, aperture pixels = {mask.sum()}")
        print(f"  pupil sigma_I2 (simulation)         = {pupil_sigma_I2:.3f}")
        print(f"  sigma_R2 (Eq. 12.38, weak)          = {sigma_R2:.3f}")
        print(f"  sigma_I2 (Eq. 12.40, mod.-strong)   = {sigma_I2_strong:.3f}")

        sigma_P2_sim = power_scintillation_index(run, M_used)
        sigma_P2_theory = irradiance_flux_variance(run, atmospheres[D_over_r0])
        print(f"  power sigma_P2 (simulation)         = {sigma_P2_sim:.2e}")
        print(f"  sigma_P2 (Eq. 12.39, D = {run['settings']['D_rx_m']} m)     = {sigma_P2_theory:.2e}")

        results[D_over_r0].update({"M_used": int(M_used),
                                   "scintillation_index": float(pupil_sigma_I2),
                                   "rytov_variance": float(sigma_R2),
                                   "scintillation_index_strong": float(sigma_I2_strong),
                                   "power_scintillation_index": float(sigma_P2_sim),
                                   "irradiance_flux_variance": float(sigma_P2_theory)})

        if D_over_r0 > 0:                              # no turbulence: trace is 0, nothing to plot
            plot_scintillation_convergence(sums["sigma_I2_pupil_m"], sigma_R2, D_over_r0, config)

        all_sums[D_over_r0] = sums

    print("\n=== Focal-plane statistics ===")
    mean_focal_images = {}                             # {D/r0: mean focal image}
    scint_focal_maps = {}                              # {D/r0: focal sigma_I^2 map}
    cdfs = {}
    for D_over_r0, sums in all_sums.items():
        mean_focal_images[D_over_r0] = mean_intensity(sums["sum_I_focal"], runs[D_over_r0]["M_used"])
        scint_focal_maps[D_over_r0] = scintillation_index(sums["sum_I_focal"], sums["sum_I2_focal"],
                                                          runs[D_over_r0]["M_used"])
        if sums["I_focal_cdf"] is not None:
            cdfs[D_over_r0] = irradiance_cdf(sums["I_focal_cdf"])

    for D_over_r0 in all_sums:
        print(f"\n--- D/r0 = {D_over_r0} ---")
        strehl_sim = strehl_ratio(mean_focal_images, D_over_r0, runs[D_over_r0])
        strehl_theory = strehl_ratio_long_exposure(atmospheres[D_over_r0]["D_over_r0"])
        alphas = (0.5, 1) if config["tip_tilt_correction"] else (0,)
        strehl_fried = {alpha: strehl_ratio_fried(atmospheres[D_over_r0]["D_over_r0"], alpha)
                        for alpha in alphas}
        print(f"  Strehl ratio (simulation)            = {strehl_sim:.3f}")
        print(f"  Strehl ratio (theory, approx for long exposure) = {strehl_theory:.3f}")
        for alpha, value in strehl_fried.items():
            print(f"  Strehl ratio (theory, alpha = {alpha:<3g}) = {value:.3f}")

        fraction = focal_power_fraction(runs[D_over_r0], runs[D_over_r0]["M_used"])
        print(f"  focal / pupil power, mean            = {fraction['mean']:.4f}")
        print(f"  focal / pupil power, min             = {fraction['minimum']:.4f}")

        results[D_over_r0].update({"strehl_ratio": float(strehl_sim),
                                   "strehl_ratio_long_exposure": float(strehl_theory),
                                   "focal_power_fraction_mean": float(fraction["mean"]),
                                   "focal_power_fraction_minimum": float(fraction["minimum"])})
        results[D_over_r0].update({f"strehl_ratio_fried_alpha_{alpha:g}": float(value)
                                   for alpha, value in strehl_fried.items()})

    save_results(results, config)

    plot_focal_scintillation_maps(scint_focal_maps, runs, config)

    if cdfs:
        plot_irradiance_cdfs(cdfs, runs, config)

    print("\n=== Example realisations ===")
    examples = {}
    for D_over_r0, run in runs.items():
        k = min(config["example_realisation"], run["M_used"] - 1)
        examples[D_over_r0] = load_realisation(run, k, run["optics"])
        print(f"  D/r0 = {D_over_r0}: realisation k = {k}")
    plot_example_realisations(examples, runs, config)
    plot_airy_check(examples, runs, config)
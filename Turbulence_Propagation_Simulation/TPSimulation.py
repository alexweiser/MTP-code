'''
Script: TPSimulation.py
Author: Alexander Weiser
License: TBD
Date: November 2026
AI use: Syntax, layout and structure of the functions (for performance and reusability),
phrasing of the docstrings, debugging and the check function were supported by
Anthropic's Claude.

This script simulates the propagation of a laser beam (plane wave) from a satellite
through the turbulent atmosphere down to a ground telescope, and images it onto the
focal plane. It uses the split-step method: the atmosphere is reduced to a few thin
phase screens (layers); the field picks up the phase of each screen and is then
propagated to the next screen (angular spectrum method). The turbulence strength per
layer follows from a Cn2 profile, here the Hufnagel-Valley model with the HV5/7
parameters (Andrews & Phillips, 2005, Eq. 12.1). The general approach follows
Saathof and Zadeh (2025).

Packages: HCIPy (Por et al., 2018) for grids, propagation and the focal plane;
AOtools (Townson et al., 2019) for the phase screens.

Usage (from the upper folder, with the virtual environment active) on MacOS:
    source MTPvenv/bin/activate
    python Turbulence_Propagation_Simulation/TPSimulation.py
Set the parameters in `config` before running. For a quick check with one realisation
and plots, remove the "#" before run_checks(config) at the end of the file.

Inputs:  all settings in the dictionary `config` below (units in the key names).
Outputs: per realisation the complex pupil and focal-plane field, the coordinate
         grids and the settings, saved as an HDF5 file (.h5).

References
Andrews, L. C., & Phillips, R. L. (2005). Laser beam propagation through random media
    (2nd ed.). SPIE Press. https://doi.org/10.1117/3.626196
Kaushal, H., & Kaddoum, G. (2017). Optical communication in space: Challenges and
    mitigation techniques. IEEE Communications Surveys & Tutorials, 19(1), 57-96.
    https://doi.org/10.1109/COMST.2016.2603518
Por, E. H., Haffert, S. Y., Radhakrishnan, V. M., Doelman, D. S., van Kooten, M., &
    Bos, S. P. (2018). High Contrast Imaging for Python (HCIPy): An open-source adaptive
    optics and coronagraph simulator. Proceedings of SPIE, 10703, 1070342.
    https://doi.org/10.1117/12.2314407
Saathof, R., & Zadeh, I. E. (2025). Focal plane irradiance statistics for laser
    downlink beams. Proceedings of SPIE, 13669, 136690M. https://doi.org/10.1117/12.3069432
Schmidt, J. D. (2010). Numerical simulation of optical wave propagation with examples in
    MATLAB. SPIE Press. https://doi.org/10.1117/3.866274
Townson, M. J., Farley, O. J. D., Orban de Xivry, G., Osborn, J., & Reeves, A. P. (2019).
    AOtools: A Python package for adaptive optics modelling and analysis. Optics Express,
    27(22), 31316. https://doi.org/10.1364/OE.27.031316
'''

import numpy as np
import hcipy as hci
import matplotlib.pyplot as plt
from aotools.turbulence import ft_sh_phase_screen
from aotools.opticalpropagation import angularSpectrum
import h5py
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# Cn2 model parameters
# ---------------------------------------------------------------------------
# HV5/7 parameters (Andrews & Phillips, 2005, p.481).
HV57_PARAMETERS = {
    "A": 1.7e-14,   # ground-level Cn2 [m^-2/3]; keys named like the arguments of the Cn2 model function
    "v": 21.0,      # rms high-altitude wind speed [m/s]
}

# ---------------------------------------------------------------------------
# Configuration: every number of the simulation lives here
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/alexanderweiser/Documents/TUDelft/MTP/MTP-Workplace/MTP-code/Data"    # data folder

config = {
    # source
    "wavelength_m": 1550e-9,
    "source_type": "plane_wave",

    # atmosphere
    "cn2_model": "HV",                  # name of the Cn2 profile model
    "cn2_params": HV57_PARAMETERS,      # parameters of that model
    "target_D_over_r0": 2,              # sets the turbulence strength directly; None = profile as it is, 0 = no turbulence (other values: e.g. 2, 8)
    "layer_heights_m": [40000, 20000, 10000, 5800, 2500, 1250, 600, 100],  # phase-screen heights
    "h_ground_m": 0.0,                  # ground level (for integration)
    "h_top_m": 40000.0,                 # top of the atmosphere (for integration)
    "zenith_deg": 60.0,                 # zenith angle of the link
    "L0_m": 1000.0,                     # outer scale
    "l0_m": 5e-3,                       # inner scale

    # numerical grid (phase screens and pupil plane)
    "N_px": 256,                        # pupil grid: N_px x N_px samples
    "screen_size_m": 2.0,               # side length of the grid

    # receiver optics (telescope)
    "D_rx_m": 1.0,                      # receiver aperture diameter
    "focal_length_m": 6.0,              # effective focal length
    # Focal-plane grid ("virtual camera"), both in units of lambda/D (= lambda*f/D = 9.3 um here):
    #   focal_q:        pixel size = lambda*f/(D*q). q >= 2 (Nyquist for intensity), 4 = fine.
    #                   Larger q -> smaller pixels.
    #   focal_num_airy: half field of view. Weak turbulence: ~8 is enough; strong (D/r0 = 8):
    #                   >= 16 (spot ~ 8 lambda/D wide + wander). Increase if light reaches the edge.
    #   Grid size: (2 * focal_q * focal_num_airy)^2 pixels.
    "focal_q": 4,                       # number of focal-plane samples per lambda/D
    "focal_num_airy": 32,               # half width of the focal field of view, in lambda/D (other values: 16, 24)

    # run settings
    "n_realisations": 4000,                # testing: 20, production runs: 4000, no turbulence: 1
    "master_seed": 1,                   # seeds the random generator
    "output_dir": os.path.join(DATA_DIR, "OUTPUT_TPSimulation"),
}

# ---------------------------------------------------------------------------
# Cn2 profile models
# ---------------------------------------------------------------------------
def hufnagel_valley(h_m, A, v):
    """Hufnagel-Valley Cn2 profile (Andrews & Phillips, 2005, Eq. 12.1, p.481).

    Parameters
    ----------
    h_m : float or numpy array
        Height above the ground [m].
    A : float
        Ground-level Cn2 [m^-2/3] (HV5/7: 1.7e-14).
    v : float
        rms high-altitude wind speed [m/s] (HV5/7: 21).

    Returns
    -------
    Cn2 at each height [m^-2/3], same shape as h_m.
    """
    h_m = np.asarray(h_m, dtype=float)    # turn input into a numpy array (if it isn't already)
    tropopause = 0.00594 * (v / 27.0)**2 * (1e-5 * h_m)**10 * np.exp(-h_m / 1000.0)
    background = 2.7e-16 * np.exp(-h_m / 1500.0)
    ground = A * np.exp(-h_m / 100.0)
    return tropopause + background + ground


def get_cn2_function(config, scale=1.0):
    """Build the Cn2 profile chosen in the config, as a function of height only.
    The rest of the script only calls cn2(h_m). To use another profile (e.g. real
    data), add a new branch for its config["cn2_model"] name.

    Parameters
    ----------
    config : dict, simulation settings (uses "cn2_model" and "cn2_params")
    scale : float, factor that multiplies the whole profile (1 = profile as it is)

    Returns
    -------
    cn2 : function h_m -> Cn2 [m^-2/3], including the factor scale.
    """
    model = config["cn2_model"]
    params = config["cn2_params"]

    if model == "HV":
        return lambda h_m: scale * hufnagel_valley(h_m, **params)
    else:
        raise ValueError(f"Unknown Cn2 model: {model}")


# ---------------------------------------------------------------------------
# Layers: Fried parameter per phase screen
# ---------------------------------------------------------------------------

def slab_edges(layer_heights_m, h_ground_m, h_top_m):
    """Boundaries of the atmospheric slabs that the phase screens represent.

    Each screen stands for one slab of the atmosphere. The boundaries lie halfway
    between neighbouring screens; the lowest slab starts at h_ground_m, the highest
    ends at h_top_m. Slab i goes from edges[i] to edges[i+1] and belongs to the
    i-th screen counted from the bottom.

    Parameters
    ----------
    layer_heights_m : list or array, heights of the phase screens [m] (any order)
    h_ground_m, h_top_m : float, lower and upper end of the atmosphere [m]

    Returns
    -------
    edges : array, len(layer_heights_m) + 1 boundaries, sorted low -> high [m]
    """
    layer_heights_sorted_m = np.sort(np.asarray(layer_heights_m, dtype=float))   # low -> high
    midpoints = (layer_heights_sorted_m[:-1] + layer_heights_sorted_m[1:]) / 2                # halfway between neighbours
    edges = np.concatenate([[h_ground_m], midpoints, [h_top_m]])    # slab i goes from edges[i] to edges[i+1]
    return edges

def layer_r0(cn2_function, layer_heights_m, wavelength_m, zenith_deg, h_ground_m, h_top_m):
    """Fried parameter r0 of each phase screen (layer), along the slanted path.

    Each screen represents a slab of the atmosphere, with the boundaries from
    slab_edges (halfway between neighbouring screens, from h_ground_m to h_top_m).
    Per slab, Cn2 is integrated over the height and converted to r0
    with r0 = (0.423 k^2 sec(zeta) * integral Cn2 dh)^(-3/5)
    (Kaushal & Kaddoum, 2017, Eq. 14; plane wave / downlink).

    Parameters
    ----------
    cn2_function : function h_m -> Cn2 [m^-2/3] (see get_cn2_function)
    layer_heights_m : list or array, heights of the phase screens [m] (any order)
    wavelength_m : float, wavelength [m]
    zenith_deg : float, zenith angle of the link [deg]
    h_ground_m, h_top_m : float, lower and upper end of the atmosphere [m]

    Returns
    -------
    layer_heights_sorted_m : array, screen heights sorted low -> high [m]
    r0_m : array, r0 of each screen [m], same order as layer_heights_sorted_m
    """
    layer_heights_sorted_m = np.sort(np.asarray(layer_heights_m, dtype=float))   # low -> high, same order as the slabs
    edges = slab_edges(layer_heights_sorted_m, h_ground_m, h_top_m)              # slab i goes from edges[i] to edges[i+1]

    k = 2 * np.pi / wavelength_m                # wavenumber [m^-1]
    sec = 1 / np.cos(np.radians(zenith_deg))    # secant of the zenith angle (slanted path)

    r0_m = np.zeros(len(layer_heights_sorted_m))

    for i in range(len(layer_heights_sorted_m)):
        h = np.linspace(edges[i], edges[i + 1], 1000)
        cn2_integral = np.trapezoid(cn2_function(h), h)       # [m^(1/3)]
        if cn2_integral == 0:                                 # no turbulence in this slab
            r0_m[i] = np.inf                                  # infinitely large r0
        else:
            r0_m[i] = (0.423 * k**2 * sec * cn2_integral) ** (-3 / 5)

    return layer_heights_sorted_m, r0_m


def total_r0(r0_layers_m):
    """Total Fried parameter of all layers together, for a plane wave (downlink):
    r0 = (sum_i r0_i^(-5/3))^(-3/5)    (Schmidt, 2010, p.165, Eq. 9.71).

    Valid for thin layers, each with its own plane-wave r0_i (Schmidt, 2010, p.165, Eq. 9.70).
    """
    total = np.sum(np.asarray(r0_layers_m) ** (-5 / 3))      # sum over the layers
    if total == 0:                                            # no turbulence in any layer
        return np.inf                                         # infinitely large r0
    return total ** (-3 / 5)


def cn2_scale_for_r0(config, r0_target_m):
    """Scale factor s for the Cn2 profile that gives the total Fried parameter r0_target_m.

    r0 is proportional to (integral of Cn2)^(-3/5) (Schmidt, 2010, p.165, Eq. 9.70), so
    scaling the profile by s changes r0 by s^(-3/5). Solved for s:
    s = (r0_unscaled / r0_target)^(5/3).

    Parameters
    ----------
    config : dict, simulation settings
    r0_target_m : float, wanted total Fried parameter [m]

    Returns
    -------
    s : float, scale factor for the Cn2 profile
    """
    # r0 per layer for the unscaled reference profile (get_cn2_function uses scale=1.0 by default)
    _, r0_layers_m = layer_r0(get_cn2_function(config), config["layer_heights_m"],
                              config["wavelength_m"], config["zenith_deg"],
                              config["h_ground_m"], config["h_top_m"])
    s = (total_r0(r0_layers_m) / r0_target_m) ** (5 / 3)    # total unscaled r0 / target r0, to the power 5/3
    return s


def atmosphere_layers(config):
    """Phase-screen heights and r0 per screen, with the Cn2 profile scaled to the
    target D/r0 if config["target_D_over_r0"] is set.

    config["target_D_over_r0"]:
        None : profile as it is (cn2_scale = 1)
        0    : no turbulence (reference case). The profile is scaled by 0, so r0 is
               infinite in every layer (set to np.inf in layer_r0 and total_r0, which
               avoids a division by zero). run_simulation then makes no phase screens.
        > 0  : profile scaled so that D/r0 equals the target

    Returns
    -------
    layer_heights_sorted_m, r0_layers_m : arrays, sorted low -> high
    cn2_scale : factor applied to the Cn2 profile (1 = profile as it is, 0 = no turbulence)
    """

    if config["target_D_over_r0"] is None:
        cn2_scale = 1.0
    elif config["target_D_over_r0"] == 0:        # no turbulence
        cn2_scale = 0.0
    else:
        r0_target_m = config["D_rx_m"] / config["target_D_over_r0"]
        cn2_scale = cn2_scale_for_r0(config, r0_target_m)
    cn2 = get_cn2_function(config, scale=cn2_scale)
    layer_heights_sorted_m, r0_layers_m = layer_r0(cn2, config["layer_heights_m"], config["wavelength_m"],
                                      config["zenith_deg"], config["h_ground_m"], config["h_top_m"]) # scaled r0 per layer
    return layer_heights_sorted_m, r0_layers_m, cn2_scale


# ---------------------------------------------------------------------------
# Receiver Optics: Pupil Grid, Aperture, Focal Grid, Propagator
# ---------------------------------------------------------------------------
def make_optics(config):
    """Receiver optics (telescope) on HCIPy grids.

    Parameters
    ----------
    config : dict, simulation settings (pupil grid, aperture, focal grid, focal length)

    Returns
    -------
    dict with
        "pupil_grid" : hci grid of the pupil plane, N_px x N_px samples over screen_size_m
        "aperture"   : hci.Field, circular aperture of diameter D_rx_m (1 inside, 0 outside,
                       edge pixels averaged over 4 x 4 sub-samples)
        "focal_grid" : hci grid of the focal plane [m], focal_q samples per lambda*f/D,
                       half width focal_num_airy * lambda*f/D
        "prop_focal" : hci.FraunhoferPropagator, lens from the pupil to the focal plane
    """
    # pupil grid
    N_px = config["N_px"]
    screen_size_m = config["screen_size_m"]
    pupil_grid = hci.make_pupil_grid(N_px, screen_size_m)

    # aperture
    D_rx_m = config["D_rx_m"]
    aperture = hci.evaluate_supersampled(hci.make_circular_aperture(D_rx_m), pupil_grid, 4)

    # focal grid
    focal_q = config["focal_q"]
    focal_num_airy = config["focal_num_airy"]
    spatial_res = config["wavelength_m"] * config["focal_length_m"] / D_rx_m

    # propagator
    focal_grid = hci.make_focal_grid(focal_q, focal_num_airy, spatial_resolution=spatial_res)
    focal_length_m = config["focal_length_m"]
    prop_focal = hci.FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length_m)

    return {"pupil_grid": pupil_grid,
            "aperture": aperture,
            "focal_grid": focal_grid,
            "prop_focal": prop_focal}

# ---------------------------------------------------------------------------
# Source field
# ---------------------------------------------------------------------------
def make_source_field(config, pupil_grid):
    """Complex electric field at the top of the atmosphere, on the whole pupil grid.

    config["source_type"]:
        "plane_wave": amplitude 1 and flat phase everywhere (Saathof & Zadeh,
                      2025, p.4: the beam arrives as a plane wave on top of the atmosphere)

    Returns
    -------
    hci.Field, complex, on pupil_grid
    """
    if config["source_type"] == "plane_wave":
        return hci.Field(np.ones(pupil_grid.size, dtype=complex), pupil_grid)
    else:
        raise ValueError(f"Unknown source type: {config['source_type']}")


# ---------------------------------------------------------------------------
# Phase screen
# ---------------------------------------------------------------------------
def make_phase_screen(r0_m, config, seed, pupil_grid):
    """Random phase screen [rad] with Fried parameter r0_m, on the pupil grid.

    FFT method with subharmonics (Schmidt, 2010, Listing 9.3, p.170), using the AOtools
    function ft_sh_phase_screen; von Karman spectrum with outer scale L0_m and inner
    scale l0_m from the config. AOtools returns a 2D numpy array; .ravel() flattens it
    into the 1D order of an HCIPy Field.

    Parameters
    ----------
    r0_m : float, Fried parameter of this screen at the simulation wavelength [m]
    config : dict, simulation settings (N_px, screen_size_m, L0_m, l0_m)
    seed : int or numpy random Generator; pass the run's generator so that every
           screen is different but the run is reproducible
    pupil_grid : hci grid on which the screen is returned

    Returns
    -------
    hci.Field, phase in radians
    """
    pixel_size_m = config["screen_size_m"] / config["N_px"]

    phase = ft_sh_phase_screen(r0_m, N=config["N_px"], delta=pixel_size_m,
                               L0=config["L0_m"], l0=config["l0_m"], seed=seed)
    phase_field = hci.Field(phase.ravel(), pupil_grid)
    return phase_field


# ---------------------------------------------------------------------------
# Split-step method
# ---------------------------------------------------------------------------
def layer_distances(config, layer_heights_sorted_m):
    """Propagation distances between the phase screens, from top to bottom [m].

    Distance i goes from screen i (counted from the top) down to the next screen; the
    last one goes from the lowest screen down to the ground. Measured along the slanted
    path: delta_h / cos(zenith).

    Parameters
    ----------
    config : dict, simulation settings (zenith_deg, h_ground_m)
    layer_heights_sorted_m : array, screen heights sorted low -> high [m] (output of layer_r0)

    Returns
    -------
    array of distances [m], ordered from top to bottom
    """
    cos_zenith = np.cos(np.radians(config["zenith_deg"]))
    heights_top_down = layer_heights_sorted_m[::-1]
    heights_below = np.append(heights_top_down[1:], config["h_ground_m"])
    return (heights_top_down - heights_below) / cos_zenith


def propagate_once(config, optics, layer_heights_sorted_m, r0_layers_m, rng, turbulence=True):
    """One realisation of the downlink: source -> all layers -> aperture -> focal plane.

    Split-step method (Schmidt, 2010, Sect. 9.1, p.150, Eqs. 9.1-9.3): at each layer
    (from top to bottom) the field is multiplied by exp(-i*phi) of the phase screen
    (Eq. 9.2), then it is propagated to the next layer.

    Propagation between the layers: angular-spectrum method, using the function
    angularSpectrum of AOtools, a Python version of Schmidt's Listing 6.5 (Schmidt,
    2010, p.102); AOtools states that its propagation methods are "extracted from the
    book by Schmidt, 2010" (https://aotools.readthedocs.io/en/latest/opticalpropagation.html).
    This propagation is periodic on the grid, so a plane wave stays exactly uniform.

    Parameters
    ----------
    config : dict, simulation settings
    optics : dict from make_optics (pupil grid, aperture, focal propagator)
    layer_heights_sorted_m, r0_layers_m : arrays from layer_r0, sorted low -> high
    rng : numpy random Generator, created once per run from config["master_seed"]
    turbulence : bool, False = no phase screens (test of the propagation itself)

    Returns
    -------
    wf_pupil : hci.Wavefront in the telescope aperture (after the aperture), on the pupil grid
    wf_focal : hci.Wavefront in the focal plane, on the focal grid

    An hci.Wavefront is an HCIPy object that bundles the field with its wavelength and
    grid. Its parts are read with a dot (no brackets). The ones used in this script:
        .electric_field  complex field E per pixel (hci.Field, a flat 1D array that knows
                         its grid; .shaped gives the 2D image, e.g. 256 x 256)
        .wavelength      wavelength [m]
        .grid            grid on which the field is defined (pupil or focal grid)
        .amplitude       |E| per pixel
        .phase           phase of E per pixel [rad]
        .intensity       |E|^2 per pixel (plane wave without turbulence = 1)
        .power           |E|^2 * pixel area, i.e. power per pixel
        .total_power     sum of .power over all pixels
    (HCIPy 0.7.1, class Wavefront; amplitude, phase, intensity, power and total_power
    are computed from .electric_field when they are read.)
    """
    wavelength_m = config["wavelength_m"]
    pixel_size_m = config["screen_size_m"] / config["N_px"]              # AOtools needs pixel size of the phase screen grid
    distances_m = layer_distances(config, layer_heights_sorted_m)                     # order top -> bottom
    r0_top_down = r0_layers_m[::-1]                                      # same order

    # field as a 2D numpy array (AOtools works with 2D arrays)
    E = make_source_field(config, optics["pupil_grid"]).shaped           # .shaped turns HCIPy Field into numpy array for AOtools

    for r0_m, distance_m in zip(r0_top_down, distances_m):
        if turbulence:
            # new random screen for each layer, but reproducible for the same run
            phase = make_phase_screen(r0_m, config, seed=rng, pupil_grid=optics["pupil_grid"])
            E = E * np.exp(-1j * phase.shaped)                           # thin phase screen, exp(-i phi) (Schmidt, 2010, Eq. 9.2)
        E = angularSpectrum(E, wavelength_m, pixel_size_m, pixel_size_m, distance_m)

    # back to HCIPy: aperture and lens
    # .ravel() flattens the 2D numpy array into the 1D order of an HCIPy Field
    E_pupil = hci.Field(E.ravel(), optics["pupil_grid"]) * optics["aperture"]
    wf_pupil = hci.Wavefront(E_pupil, wavelength_m)
    wf_focal = optics["prop_focal"](wf_pupil)
    return wf_pupil, wf_focal


# ---------------------------------------------------------------------------
# Realisation loop with saving to .h5
# ---------------------------------------------------------------------------
def run_simulation(config):
    """Run all realisations and save them to one HDF5 file (.h5).

    The file contains the settings (as attributes), the layers, the grid coordinates,
    and per realisation the complex pupil and focal-plane field and their total power.

    Returns
    -------
    path : str, path of the saved .h5 file
    """
    # --- Atmosphere, optics and random generator (computed once per run) ---
    layer_heights_sorted_m, r0_layers_m, cn2_scale = atmosphere_layers(config)    # scaled to the target D/r0 if set
    r0_tot_m = total_r0(r0_layers_m)
    optics = make_optics(config)
    # random generator: each layer draws new numbers from it, but the run stays reproducible
    rng = np.random.default_rng(config["master_seed"])
    M = config["n_realisations"]
    D_r0 = config["D_rx_m"] / r0_tot_m
    n_pupil = config["N_px"]                                       # image size in pixels (pupil plane)
    n_focal = 2 * config["focal_q"] * config["focal_num_airy"]     # image size in pixels (focal plane)

    # --- Output file: folder and descriptive file name ---
    os.makedirs(config["output_dir"], exist_ok=True)    # create folder if needed
    filename = (f"TPSimulation__model_{config['cn2_model']}__Dr0_{D_r0:.2f}__M_{M}__seed_{config['master_seed']}__{datetime.now():%Y%m%d_%H%M}.h5")
    path = os.path.join(config["output_dir"], filename)

    with h5py.File(path, "w") as f:                 # new .h5 file, opened for writing ("w")

        # --- Simulation settings, stored as attributes of the file ---
        for key, value in config.items():
            if isinstance(value, dict):             # store dict entries one by one
                for sub_key, sub_value in value.items():
                    f.attrs[f"{key}_{sub_key}"] = sub_value
            elif value is not None:
                f.attrs[key] = value
        f.attrs["created"] = datetime.now().isoformat()
        f.attrs["cn2_scale"] = cn2_scale
        f.attrs["D_over_r0"] = D_r0                 # D/r0 actually reached (may differ slightly from the target)

        # --- Layers and grid coordinates (identical for all realisations) ---
        f.create_dataset("layer_heights_sorted_m", data=layer_heights_sorted_m)
        f.create_dataset("r0_layers_m", data=r0_layers_m)
        f.create_dataset("r0_total_m", data=r0_tot_m)
        # only x-coordinates are stored; y-coordinates are identical (square grid)
        f.create_dataset("pupil_x_m", data=optics["pupil_grid"].separated_coords[0])
        f.create_dataset("focal_x_m", data=optics["focal_grid"].separated_coords[0])

        # --- Empty datasets for the results, filled one realisation at a time ---
        # complex64: 8 bytes per value instead of the default 16
        # chunks: each realisation is stored as one block, for faster writing and reading
        pupil_field = f.create_dataset("pupil_field", shape=(M, n_pupil, n_pupil),
                                       dtype="complex64", chunks=(1, n_pupil, n_pupil))
        focal_field = f.create_dataset("focal_field", shape=(M, n_focal, n_focal),
                                       dtype="complex64", chunks=(1, n_focal, n_focal))
        pupil_power = f.create_dataset("pupil_power", shape=(M,), dtype="float64")
        focal_power = f.create_dataset("focal_power", shape=(M,), dtype="float64")

        # --- Realisation loop: propagate and write each result directly to the file ---
        for k in range(M):
            # phase screens are switched off when there is no turbulence
            wf_pupil, wf_focal = propagate_once(config, optics, layer_heights_sorted_m, r0_layers_m, rng,
                                                turbulence=(cn2_scale > 0))
            # .shaped turns the flat HCIPy field (1D array) into a 2D image for saving
            pupil_field[k] = wf_pupil.electric_field.shaped
            focal_field[k] = wf_focal.electric_field.shaped
            pupil_power[k] = wf_pupil.total_power
            focal_power[k] = wf_focal.total_power
            if (k + 1) % 10 == 0 or k + 1 == M:     # progress message every 10 realisations and at the end
                print(f"\rrealisation {k + 1}/{M} done", end="", flush=True)
        print()

    print(f"saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_checks(config):
    """Check the simulation with one realisation before a long run.

    Expected for HV5/7, 1550 nm, 60 deg zenith, D = 1 m, D/r0 = 8:
        no turbulence: focal power almost equal to pupil power (diffraction-limited spot)
        no phase screens: |E| = 1 and flat phase in the aperture (propagation adds nothing)
        all layers: pupil scintillation index about 0.25, speckled focal spot
    """
    layer_heights_sorted_m, r0_layers_m, _ = atmosphere_layers(config)
    r0_tot_m = total_r0(r0_layers_m)
    D_r0 = config["D_rx_m"] / r0_tot_m
    optics = make_optics(config)
    rng = np.random.default_rng(config["master_seed"])
    inside = optics["aperture"] > 0.999         # pixels fully inside the aperture
    n_layers = len(layer_heights_sorted_m)
    print(f"r0 = {r0_tot_m * 100:.1f} cm, D/r0 = {D_r0:.1f}")

    # Diffraction-limited spot (no turbulence)
    wf = hci.Wavefront(make_source_field(config, optics["pupil_grid"]) * optics["aperture"],
                       config["wavelength_m"])
    focal_no_turb = optics["prop_focal"](wf)
    peak_no_turb = focal_no_turb.power.max()
    print(f"no turbulence: power in pupil {wf.total_power:.4f}, in focal plane {focal_no_turb.total_power:.4f}")

    # Propagation through all layers without phase screens
    wf_pupil, _ = propagate_once(config, optics, layer_heights_sorted_m, r0_layers_m, rng, turbulence=False)
    E = wf_pupil.electric_field[inside]
    print(f"no phase screens: |E| from {np.abs(E).min():.4f} to {np.abs(E).max():.4f}, "
          f"phase std {np.std(np.angle(E)):.1e} rad")

    # One realisation through all layers
    wf_pupil, wf_focal = propagate_once(config, optics, layer_heights_sorted_m, r0_layers_m, rng)
    I = wf_pupil.intensity[inside]
    scint_index = np.mean(I**2) / np.mean(I)**2 - 1     # Kaushal & Kaddoum (2017), Eq. 10, p.66
    print(f"{n_layers} layers: scintillation index {scint_index:.3f}, "
          f"focal/pupil power {wf_focal.total_power / wf_pupil.total_power:.3f}, "
          f"peak {wf_focal.power.max() / peak_no_turb:.3f} of the diffraction-limited peak")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [
        (focal_no_turb.power / peak_no_turb, 1e-6, "Focal plane, no turbulence", "x [um]",
         "intensity / diffraction-limited peak"),
        (wf_pupil.intensity, 1, f"Pupil plane, {n_layers} layers\nscintillation index {scint_index:.2f}",
         "x [m]", "intensity (no turbulence = 1)"),
        (wf_focal.power / peak_no_turb, 1e-6, f"Focal plane, {n_layers} layers, D/r0 = {D_r0:.1f}",
         "x [um]", "intensity / diffraction-limited peak"),
    ]
    for ax, (image, unit, title, xlabel, cbar_label) in zip(axes, panels):
        im = hci.imshow_field(image, ax=ax, grid_units=unit, cmap="inferno")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("y [um]")
    axes[1].set_ylabel("y [m]")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    #run_checks(config)       # quick check with one realisation and plots
    run_simulation(config)
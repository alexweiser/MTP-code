MTP-code
========

Python code for the Master Thesis Project (Applied Physics, TU Delft):
"Effects of Atmospheric Turbulence on the Detection Performance and Throughput
of an SNSPD Array for Laser Satellite Communication".
Contains simulations, analysis scripts and (later) instrument control.


Owner and point of contact
--------------------------
Alexander Weiser (alexander.weiser@outlook.at)
Supervisors: Dr. Iman Esmaeil Zadeh (QuPSI, Faculty of Applied Sciences),
             Dr.ir. Rudolf Saathof (FSOC, Faculty of Aerospace Engineering)


Directory contents
------------------
MTP-code/
  README.txt            This file
  requirements.txt      Python packages needed to run the code
  Turbulence_Propagation_Simulation/
    TPSimulation.py     Split-step propagation of a plane wave from the satellite through
                        the turbulent atmosphere (HV5/7 Cn2 profile, phase screens,
                        angular-spectrum propagation) onto the telescope focal plane.
                        Saves the pupil and focal field of every realisation to one .h5 file.
    TPSim_Analysis.py   Validation of the simulation: statistics over the realisations,
                        per simulation pixel, compared with theory and with
                        Saathof and Zadeh (2025). Optional tip-tilt correction.
  Plotting/             Common style for all thesis figures (see Plotting/HOWTO.md)
    thesis.mplstyle     matplotlib style
    figstyle.py         use(), figsize(), panel_labels(), save()
    example_figstyle.py template plotting script with dummy data


Usage
-----
1. Create the virtual environment and install the packages (macOS):
     python3 -m venv MTPvenv
     source MTPvenv/bin/activate
     pip install -r requirements.txt

2. Data is not stored in this repository. Measurement data lives on the QuPSI U:drive.
   Simulation output is written to a local folder Data/ (git-ignored). Its path is set
   by DATA_DIR at the top of each script: change it to your own folder before running.

3. All settings are in the dictionary `config` at the top of each script (units in the
   key names); there are no command-line options. Run from the MTP-code folder.

   a) Simulation
        python Turbulence_Propagation_Simulation/TPSimulation.py
      Main settings: target_D_over_r0 (0, 2, 8), n_realisations, master_seed.
      Writes one file per run to Data/OUTPUT_TPSimulation/:
        TPSimulation__model_<Cn2 model>__Dr0_<D/r0>__M_<realisations>__seed_<seed>__<date_time>.h5
      One realisation takes about 0.3-0.4 s; 4000 realisations give about 4 GB.

   b) Analysis
        python Turbulence_Propagation_Simulation/TPSim_Analysis.py
      Reads the newest .h5 file for each D/r0 in D_over_r0_cases (0 is needed as the
      reference). Main settings: n_realisations_max, tip_tilt_correction.
      Prints per D/r0 the pupil and power scintillation index, the Strehl ratio and the
      focal/pupil power, each next to its theory value. Saves the figures and
      results.json to a new subfolder of Data/OUTPUT_TPSim_Analysis/:
        Analysis_Run__model_<model>__Dr0_<D/r0 list>__M_<M>__tiptilt_<on/off>__<date_time>/


Tests
-----
   For the simulation, remove the "#" before run_checks(config) at the end of
   TPSimulation.py. It runs one realisation, checks the diffraction-limited spot, the
   propagation without phase screens (|E| = 1, flat phase) and the pupil scintillation
   index, and shows the images.

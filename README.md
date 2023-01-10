# lelec2710
Files of the simulations made for the course LELEC2710 at [UCLouvain](https://uclouvain.be/) ([EPL](https://uclouvain.be/en/faculties/epl)).  
The goal was to reproduce the simulation results presented in this [paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.075310).  
All the simulations were performed using [Kwant](https://kwant-project.org/).

## How to run a simulation
This is done with the `batch.py` file.
One simply needs to call one (or more) of the already implemented simulations:
* `G_simple`: conductance with respect to tip position (2D)
* `G_1D`: same as `G_simple` but 1D (along y = 0)
* `G_advanced`: same as `G_simple` but with "adaptative" mesh
* `Vg_simple`: same as `G_1D` but also with respect to amplitude of tip potential amplitude
* `Rp_simple`: same as `G_1D` but also with respect to FWHM of tip potential amplitude
* `Ef_simple`: same as `G_1D` but also with respect to Ef
* `dG_simple`: conductance with respect to Ef and tip potential amplitude
* `dG_Rp`: same as dG_simple but with routine to vary Rp

This will generate '.mat' files (in the 'saved_data' folder) containing all the simulation infos (settings and output).

## How to plot the results of a simulation
This is done with the `plot_results.py` file.
One simply needs to call the exact same functions (with the same arguments) as in the simulations.  
Do not forget to call `plt.show()` at the end (and optionnaly some `plt.figure()` in between the different calls).

## Unknown module `tqdm`
This module is just for a loadbar and is not essential to the simulations.
You can comment the following lines in `run_simulations.py` if needed:
* 8: `from tqdm import tqdm`
* 134: `pbar = tqdm(total=N, desc=filename) #loadbar`
* 139: `pbar.update(1)`

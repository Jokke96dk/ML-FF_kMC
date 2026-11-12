# Background

Microkinetic models of catalytic reactions usually rely on a fixed set of rate
constants, computed beforehand for static catalyst structures. This breaks down
when the catalyst structure changes under reaction conditions in ways that affect the performance.
Copper-exchanged chabazite (Cu-CHA), an NH<sub>3</sub>-SCR catalyst used for removal of NOx, is an example of such a case.

This code evaluates the rate constants on-the-fly during the kinetic Monte Carlo (kMC)
simulation, using an machine-learning force field (ML-FF). The method can in principle be applied to
other catalysts where structural changes during reaction affects the performance, but this code written
for the case of Cu-CHA.

See the preprint for further information:

Bjerregaard, J. D.; Votsmeier, M.; Grönbeck, H. On-the-fly Neural Network based Kinetic Monte Carlo Simulations of NH<sub>3</sub>-SCR over Cu-CHA. ChemRxiv 2026. DOI: 10.26434/chemrxiv.15012036/v1 

# How does it work?

The kMC part of the code is built on the MonteCoffee framework. For information on how the kMC code works, we
refer to their Gitlab page https://gitlab.com/ChemPhysChalmers/MonteCoffee or the following publication
M. Jørgensen and H. Grönbeck, *MonteCoffee:
A programmable kinetic Monte Carlo framework*, J. Chem. Phys. 149, 114101 (2018),
https://doi.org/10.1063/1.5046635. 

For the code provided here, the rate constant are evaluated by structural optimization using an ML-FF in LAMMPS. 
Every new configuration the kMC visits is minimised with a DeePMD-kit model in LAMMPS, and the rate constants are calculated from the
resulting energies. The minimisations are spread over MPI ranks, where rank 0 runs the kMC and the other ranks work as workers.
All results are stored and reused if the same configuration appears again.


## Files

| File | Content |
| --- | --- |
| `main.py` | Settings and setup of the simulation. Start here |
| `ML_FF.py` | Everything related to the ML-FF |
| `user_*.py` | The MonteCoffee user layer: sites, system, events and the kMC loop |
| `base/` | The base of the kMC code MonteCoffee |
| `Structure.vasp` | Starting structure. In this case, it is an Cu-CHA structure with the composition Si/Al:30 and Cu/Al: 0.96 |
| `FF_CuCHA_80_32_8.pb` | The DeePMD model used for the production runs (default) |
| `FF_CuCHA_minimal.pb` | A small DeePMD model for tests **only** |
| `kMC_options.cfg` | MonteCoffee options |
| `run_local` | Runs the simulation on a local machine |
| `run_cluster` | Example SLURM script for running on HPC cluster |


## Running a test

```bash
./run_local 3
```

This starts 3 MPI ranks: rank 0 runs the kMC and the other two are ML-FF workers with one core each. 
For a quicker test, set `model_file = 'FF_CuCHA_minimal.pb'` in `main.py`. This model is small and not
accurate, so use it only to check that everything runs.

To continue a simulation, set `Restart_sim = True` in `main.py` and start it in the same directory. It
starts from the last line of `coverages.txt` and reuses the stored minimizations. A fresh start refuses to
run if `data_MLFF_*.pkl` files exist in the directory.


## Output and restart

- `coverages.txt`: the coverage string for each kMC step
- `time.txt`, `mcstep.txt`, `evs_exec.txt`, `kMClog_*.txt`: kMC time, steps and executed events
- `data_MLFF_*.pkl`: all minimizations done so far (energy and structure per coverage). Written after every
  5 kMC steps that needed new minimizations, and at the end of the simulation. This file can be used for post-processing of the sampled structures.


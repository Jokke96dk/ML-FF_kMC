"""
Runs the kMC simulation of [Cu(NH3)2]+ and NH4+ diffusion in Cu-CHA with on-the-fly rates obtained from the ML-FF.

Run with MPI: rank 0 is the kMC master and the other ranks are workers.

"""
import os
import numpy as np
from mpi4py import MPI
from user_sites import Site, assign_atoms, adjust_positions, NH4_site
from user_system import System
from user_kmc import NeighborKMC
from user_events import (CuDiffEvent)
from ML_FF import load_calc


def run_test():
    """Runs the simulation of Cu diffusion in Cu-CHA using on the fly rate estimation with MLFF

    First, constants are defined and old output files cleared.
    Next, the sites, events, system and simulation objects
    are loaded, and the simulation is performed.

    """
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()

    # Define constants.
    # ------------------------------------------
    tend = 1  # End time of simulation (s) - For the diffusion time is arbitary as ea is not calculated
    T = 473.0   # Temperature of simulation in K
    max_kmc_steps = 2  # Stop after this many kMC steps even if tend is not reached. None for no limit
    initial_structure = "Structure.vasp" # Starting structure in vasp format
    # Define ML-FF settings
    # ------------------------------------------
    model_file = 'FF_CuCHA_80_32_8.pb'  # DeePMD model (DPLR). 'FF_CuCHA_minimal.pb' is a small, inaccurate model for testing
    max_min_steps = 50  # Max number of iterations in each ML-FF minimization. Can be set to 1, for testing with FF_CuCHA_minimal.pb.
    cores_per_worker = 1  # MPI ranks per ML-FF worker. Rank 0 is the kMC master, the rest are split into workers
    first_worker_cores = None  # Size of the first worker if different. Since one core is used as kMC master, the remaning number of ranks is uneven
    plugin = None  # DeePMD LAMMPS plugin. None finds the plugin of the installed deepmd-kit (local .venv).
    


    Restart_sim = False
    # ------------------------------------------

    p = parameters = events = reverse_events = None  # Set on the master and sent to all ranks
    if me == 0:
        # base/kmc.py restarts whenever data_MLFF_1.pkl exists, so it has to agree with Restart_sim
        pkl_found = os.path.isfile('data_MLFF_1.pkl')
        if Restart_sim and not (pkl_found and os.path.isfile('coverages.txt')):
            print('Restart_sim is True but data_MLFF_1.pkl or coverages.txt is missing. Aborting')
            comm.Abort(1)
        if not Restart_sim and pkl_found:
            print('Restart_sim is False but data_MLFF_*.pkl files exist. base/kmc.py would restart from them. '
                  'Remove them or set Restart_sim = True. Aborting')
            comm.Abort(1)

        np.savetxt("time.txt", [])
        if not Restart_sim:
            np.savetxt("coverages.txt", [])
        else:
            from datetime import datetime
            import shutil
            timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            src = "coverages.txt"
            dst = f"coverages_{timestamp}.txt"
            shutil.copy(src, dst)
            with open("coverages.txt") as f: # Find last line
                lines = [line for line in f.read().splitlines() if line.strip()]
            if not lines:
                print('coverages.txt is empty. Cannot restart. Aborting')
                comm.Abort(1)
            last_line = "".join(lines[-1].split())

        np.savetxt("evs_exec.txt", [])
        np.savetxt("mcstep.txt", [])



        # Define the sites from ase.Atoms.
        # ------------------------------------------


        # Load zeolite strucutre. 
        if Restart_sim:
            system_data = load_calc()
            pos = system_data['framework'][1].copy() # To save space the framework (Si,O,Al), which is frozen, is only saved on with the key framework
            pos_atoms = system_data[last_line][1].copy()
            pos += pos_atoms

            pos.write('TEST_POSCAR', parallel=False)

        else: # Load from file
            print('reading')
            from ase.io import read
            pos = read(initial_structure,parallel=False)
            print('Finished reading')

        del pos[[atom.symbol == 'Ne' for atom in pos]]
        if not Restart_sim:
            pos = adjust_positions(pos)
        pos, NH4_atom_indice, NH4_cov, NH4_F,F_Ne_block = NH4_site(pos)  # Make Ne atoms as a site for NH4+
        # Assign the Cu atoms to a cage (F site)
        Cu_indice,_,Cu_atom_indice,CuAl,SiAl = assign_atoms(pos)
        print('Starting calculation for a system with Si/Al of ' + str(SiAl) + ' and Cu/Al ratio of ' + str(CuAl))

        sites = []
        F_indices = [i for i, atom in enumerate(pos) if atom.symbol == 'F']
        Ne_indices = [i for i, atom in enumerate(pos) if atom.symbol == 'Ne']
        for i in range(len(F_indices) + len(Ne_indices)): # Define kMC sites
            sites.append(Site(stype=0,
                              covered=0, ind=i,atoms=[[],[]],atom_type=''))


        # Insert Cu into the kMC sites and keep track of indices
        for i in range(len(Cu_indice)):
            sites[i].covered += Cu_indice[i]
            sites[i].atom_type = 'Cu'
        for i in range(len(Cu_atom_indice)):
            if not sites[Cu_atom_indice[i][0]].atoms[0]:
                sites[Cu_atom_indice[i][0]].atoms[0] = [Cu_atom_indice[i][1]]
            else:
                sites[Cu_atom_indice[i][0]].atoms[1] = [Cu_atom_indice[i][1]]

        # Insert NH4+ into the kMC sites and keep track of indices
        for i in range(len(NH4_cov)):
            sites[i+len(Cu_indice)].covered += NH4_cov[i]
            sites[i+len(Cu_indice)].atom_type = 'N'
            if sites[i+len(Cu_indice)].covered > 1:
                print('More than one NH4+ on the same site!!')
                comm.Abort(1)

        for i in range(len(NH4_atom_indice)):
            sites[NH4_atom_indice[i][0]].atoms[0] = [NH4_atom_indice[i][1]]
        for i in range(len(NH4_F)):
            sites[NH4_F[i][0]].atoms[1] = [NH4_F[i][1],NH4_F[i][2]]



        print('Found ' + str(len(Cu_indice)) + ' Cu sites and ' + str(len(NH4_cov)) + ' NH4+ sites')

        # Check if cov is the same as start
        if Restart_sim:
            cov = ''
            for g in sites: # Get coverage
                cov += str(g.covered)
            if cov != last_line:
                print('Restarted simulation. old cov does not match new cov. see Main')
                comm.Abort(1)
            else:
                print('Restarted simulation succesfully initiated!')
        # Instantiate a system, events, and simulation.
        # ---------------------------------------------
        p = System(atoms=pos, sites=sites)
        # Set the global neighborlist based on distances: With PBC. Site given as F (Cu) and Ne (NH4+)
        p.set_neighbors(Ncutoff=11, pbc=True, atom_id=['F', 'Ne'])

        events = [CuDiffEvent]

        reverse_events = {0: 0}

        parameters = {"T": T, 'N_block': F_Ne_block, 'max_kmc_steps': max_kmc_steps,
                      'model_file': model_file, 'max_min_steps': max_min_steps,
                      'cores_per_worker': cores_per_worker, 'first_worker_cores': first_worker_cores,
                      'plugin': plugin}

    p, tend, parameters, events, reverse_events = comm.bcast((p, tend, parameters, events, reverse_events), root=0) # Broadcast the parameters to all ranks
    if me == 0 and Restart_sim:
        p.calc = system_data  # Only the master saves the ML-FF calculations, to avoid memory problem. 

    # Instantiate simulator object.
    sim = NeighborKMC(system=p, tend=tend,
                      parameters=parameters,
                      events=events,
                      rev_events=reverse_events)

    # Run the simulation.
    sim.run_kmc()
    sim.mlff.close()  # Close all ranks. LAMMPS must be closed first


    if me == 0:
        if sim.t < tend:
            print("Simulation stopped after max_kmc_steps = " + str(max_kmc_steps) + " steps")
        else:
            print("Simulation end time reached ! ! !")



    MPI.Finalize()
if __name__ == '__main__':
    run_test()

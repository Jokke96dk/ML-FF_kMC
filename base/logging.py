"""Defines the Log class to log results of a kMC run.

"""
import time
import copy
import numpy as np
from ase.io import write
from ase import Atoms
import h5py
import pickle
import os

class Log:
    """Handles logging of kMC simulations.
            
    Initializes a filename based on the CPU date and time.  
    All passed *parameters* will be written to the log.

    Parameters
    ----------
    parameters: dict
        parameters to dump at the beginning of a log. For example

        >>> dict = {'T':300, 'pCO':1E2}

    Examples
    ---------
    Simply instantiate a Log and write a line as

    >>> log = Log(parameters = _params)
    >>> log.write_line("This is a line!")

    Dump a simulation point to the log:

    >>> log.dump_point(step=100, sim_time=1E-9, ev_called=[10,90,0,0,0])

    See Also
    ---------
    Module: NeighborKMC.base.kmc

    """

    def __init__(self, parameters, accelparams = {"on": False}):
        self.fn = 'kMClog_'+time.strftime('%Y-%m-%d_%H-%M')+'.txt'
        self.f = open(self.fn,'a')
        #with open(self.fn, 'a') as f:
        self.f.write(r'''  __  __             _        ____       __  __           
|  \/  | ___  _ __ | |_ ___ / ___|___  / _|/ _| ___  ___ 
| |\/| |/ _ \| '_ \| __/ _ \ |   / _ \| |_| |_ / _ \/ _ \
| |  | | (_) | | | | ||  __/ |__| (_) |  _|  _|  __/  __/
|_|  |_|\___/|_| |_|\__\___|\____\___/|_| |_|  \___|\___|
                                                         
''')
        self.f.write('MonteCoffee is developed at Chalmers University of Technology with contributions from:\n')
        self.f.write('\nMikkel Jorgensen (2015-2019)')
        self.f.write('\nNoemi Bosio (since 2019)')
        self.f.write('\nElisabeth M. Dietze (since 2020)')
        self.f.write('\n ** This is a modified version for calculating the rates on-the-fly using a ML-FF implemented in deepmd-kit **')
        self.f.write('\n ** See the following preprint for details: **')

        self.f.write('\n'+'-'*80+'\n')
        self.f.write('Simulation parameters\n')
        for p in parameters.keys():
            self.f.write(format(str(p),'<10')+format(':','<5')+
                    str(parameters[p])+'\n')

        self.f.write('\n'+'-'*80+'\n'*3)
        self.f.write('kinetic Monte Carlo Log \n\n')
        self.f.write('{:<10s} {:^20s} {:^20s} {:^30s} {:<10s}'.format('Step',
                'time[hr:min:s]','Sim time [s]','mlff called',str(parameters["Number of events"])+' \n'))
        self.f.flush()
        os.fsync(self.f.fileno())  # Ensure it's written to disk


    def write_line(self, string):
        """Writes a line to the log.
        
        Appends a string to the end of the log
        on its own line.
        
        Parameters
        ----------
        string: str
            string to write to log.

        """
        self.f.write(string)
        self.f.flush()
        os.fsync(self.f.fileno())  # Ensure it's written to disk


    def dump_point(self, step, sim_time, ev_called,mlff_called):
        """Writes a simulation point to the log.
           
        Method writes the Monte Carlo step number *step*,  
        the time in the MC simulation *sim_time*,  
        and the number of event calls *ev_called*.
       
        Parameters
        ----------
        step: int
            The Monte Carlo step number.
        sim_time: float
            The simulation time in seconds.
        ev_called: list(int)
            The number of times each event is called during simulation.
            For example [N_CO_ads,N_CO_des,...].

        """
#        with open(self.fn, 'a') as f:
        time_str = time.strftime('%H:%M:%S')
        #self.f.write('{:<10s} {:^20s} {:^30s} {:<10s}'.format(str(step),
        #            time_str,str(sim_time), str(["%.0f"%item for item in ev_called])+'\n'))
        self.f.write('{:<10s} {:^20s} {:^30s} {:^10d} {:<10s}'.format(
            str(step),
            time_str,
            str(sim_time),
            mlff_called,  # This is the new integer variable
            str(["%.0f" % item for item in ev_called]) + '\n'
        ))
        self.f.flush()
        os.fsync(self.f.fileno())  # Ensure it's written to disk


    def save_atoms(sim,filename):
        """Writes tagged ase.Atoms to file.

        Writes self.atom_cfgs to file with path filename.
        The variable self.atom_cfgs can be tagged with coverages or
        augmented with molecules near the sites to
        visualize the reaction trajectory. This is currently not implemented.

        Parameters
        ------------
        filename: str
            Path to file.

        """
        atoms_write=copy.deepcopy(sim.system.surf_atoms)
        acell = atoms_write.get_cell()
        pos_ext = list(atoms_write.get_positions())
        sym_ext = list(atoms_write.get_chemical_symbols())
        pos_ind = list(sim.system.atoms.get_positions())
        max_z = pos_ext[-1][2] 
        for i in range(sim.system.Nsites):
          #if self.system.sites[i].covered == 0:
          #  pos.append(a_pos[self.system.sites[i].ind][0])
          #  syms.append('Pt')
          if sim.system.sites[i].covered == 1:
            sym_ext.append('C')
            pos_ext.append([pos_ind[sim.system.sites[i].ind][0],pos_ind[sim.system.sites[i].ind][1],max_z+1.5])
          elif sim.system.sites[i].covered == 2:
            sym_ext.append('O')
            pos_ext.append([pos_ind[sim.system.sites[i].ind][0],pos_ind[sim.system.sites[i].ind][1],max_z+1.5])

        a_other = Atoms(sym_ext)
        a_other.set_positions(pos_ext)
        a_other.set_cell(acell)
        a_other.write(filename)

    def save_txt(self, save_step = 1000.):
        """Saves txt files containing the simulation data.

        Saves the number of events executed on
        the different types of sites, the time vs mcstep,
        the site-types, and optionally the coverages if
        *self.covered* is True.

        Growing lists are cleaned from memory.

        """

        if self.verbose:
            print('Saving .txt files ...')

        # Save global neighborlist to one file
        with open("coverages.txt", "ab") as f2:
            np.savetxt(f2, self.covered, delimiter=" ", fmt="%s") # added to make it possible to save strings also

        with h5py.File("detail_site_event_evol.hdf5", "a") as f2:
            size_shape = int(len(self.system_evolution[0]))
            size_resize = int(f2["time"].shape[0] + size_shape)
            f2["time"].resize(size_resize,axis=0)
            f2["time"][-size_shape:] = self.system_evolution[3]
            f2["site"].resize(size_resize,axis=0)
            f2["site"][-size_shape:] = self.system_evolution[0]
            f2["othersite"].resize(size_resize,axis=0)
            f2["othersite"][-size_shape:] = self.system_evolution[1]
            f2["event"].resize(size_resize,axis=0)
            f2["event"][-size_shape:] = self.system_evolution[2]


        with open("mcstep.txt", "ba") as f2:
            np.savetxt(f2, [self.stepNMC])

        with open("evs_exec.txt", "wb") as f2:
            np.savetxt(f2, self.evs_exec)

        with open("time.txt", "ab") as f2:
            np.savetxt(f2, self.times)

        # Write rates as dict
        #with open('data_MLFF.pkl', 'wb') as f2:
        #    pickle.dump(self.system.calc, f2)
        

        base_filename="data_MLFF"
        chunk_size = 100000 # Keep this many entries per file
        items = list(self.system.calc.items())
        #n_chunks = math.ceil(len(items) / chunk_size)
        n_chunks = (len(items) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            chunk_items = items[i*chunk_size:(i+1)*chunk_size]
            chunk_dict = dict(chunk_items)
            filename = f"{base_filename}_{i+1}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(chunk_dict, f)
            print(f"Saved {len(chunk_dict)} items to {filename}")

        if self.write_atoms:
            Log.save_atoms(self,"atoms_"+str(self.stepNMC)+".traj")

        # Clear up lists that grow with time:
        self.times = []
        self.covered = []
        self.system_evolution = [[] for i in range(4)]
#

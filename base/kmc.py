"""Defines the NeighborKMCBase class.

The methods are used to perform kMC 
simulations with the first reaction method.

"""
from __future__ import print_function
from six.moves import configparser
import six
from mpi4py import MPI
import os


if six.PY2:
    ConfigParser = configparser.SafeConfigParser
else:
    ConfigParser = configparser.ConfigParser

import numpy as np
import random
random.seed()
#from bisect import bisect
import bisect
from sortedcontainers import SortedList
from ML_FF import MLFFCalculator,MLFFSettings,load_calc,check_NH4_sites,recompute_neighbors,check_structure,recompute_neighbors_all_site,check_Cu_sites
from user_sites import adjust_positions
import copy

class NeighborKMCBase:
    """Main class for performing MonteCoffee simulations.
          
    Assigns a system to the simulation, stores parameters, 
    and reads in software configuration from the separate  
    file kMC_options.cfg.  
    Then it sets the time equal to zero and prepares to perform
    frm kinetic Monte Carlo simulations.

    Attributes:
    
    system: System
        The system instance to perform the simulation on.

    tend: float
        Simulation end-time, given in seconds.

    parameters: dict
        parameters used to calculate rate-constants and to dump to log files.
        Example: parameters = {'pCO':1E2,'T':700,'Note':'Test simulation'}

    t: float
        Current simulation time in seconds.

    *Attributes used to keep track of events (where and when they happen)*

    siteslist: list(int)
        The list of sites for each specific event.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    other_sitelist: list(int)
        The list of neighbor sites for each specific event.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    lastsel: int
        The int of the last selected site.

    lastother: int
        The int of the last selected neighbor site.

    rindex: list(list(list(int)))):
        The index of the specific events in lists like self.frm_times. For example to find the indices
        of site no i and event no j and neighbor number k to site i, call
        rindex[i][j][k].

    possible_evs: list(int):
        List of events that are possible, used for superbasin algorithms.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    evs: numpy.ndarray(int):
        The event numbers for each specific event.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    rs: numpy.ndarray(float)
        Rate constants of specific events.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    wheres: list(list(int)):
        List of all the positions of the event-types in the lists with length
        len(self.events)*len(self.sites)*len(self.sites). To find all site-indices where event i
        happens, call wheres[i].
  
    involve_other: bool:
        -False if the event happens only on one specific site
        -True if the event modifies two or more sites         


    *Statistics counting attributes used to log and write output*

    SaveSteps: int
        The number of Monte Carlo steps between saving the .txt files.

    LogSteps: int
        The number of Monte Carlo steps between logging steps.

    tinfinity: float
        What time to put impossible events to.

    Nspecies: int
        How many different types of species are in the simulation. Used to
        print and log.
   
    nninter: int
        How deep is the nearest-neighbor interaction (depth of effect of event on neighbor properties)

    verbose: bool
        If True, the code prints verbose information.

    save_coverages: bool
        If True, coverages are saved to coverages.txt and the site, othersite and event evolution in detail_site_event_evol.hdf5. This can result in
        large files.

    write_atoms: bool
        If True, the surface atoms are written with the step number in the filename. It has to be adjusted for adsorption species individually. 

    times: list(float)
        List of times for each logged monte carlo steps in self.MCstep

    MCstep: list(int)
        List of Monte Carlo step numbers logged.

    Nsites: int
        The number of sites in self.system

    Nstypes: int
        The number of distinct site-types.

    covered: list(list(int))
        A list of site-occupations, of each site for each logged step.
        To find the site-occupation at step no self.MCstep[i] and site j, call
        covered[i][j].

    system_evolution: list(list())
        List which contains a list of the fired event with at site, other site and time

    used_ijk: list(tuples(site,event,othersite))
        List of tuples representing the unique neighbor-event pairs avoiding double counting. 

    *Superbasin attributes related to temporal acceleration*

    equilEV: list(int)
        A list of the event-indices that are quasi-equilibrated.

    Suffex: list(int)
        A list of the event-indices that are quasi-equilibrated and sufficiently executed.

    tgen: list(float)
        A list of when each specific event was generated.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    us: list(float)
        A list of random deviates used when each specific event was generated.
        This list has the length: len(self.events)*len(self.sites)*len(self.sites).

    rescale: bool
        Defines if the rates have to be rescaled and frm timelist updated.

    r_S: numpy.ndarray(float)
        The cumulative rates in the current superbasin.

    dt_S: list(float)
        The time-steps taken in the current superbasin.

    nem: numpy.ndarray(int)
        The number of times an event was performed the last self.ne steps

    use_scaling_algorithm: bool
        Defines if the rate constants are scaled in any way or not

    delta: float
        Reversibility tolerance to determine if reactions have become quasi-equilibrated.

    Nf: int
        The average number of steps a quasi-equilibrated event should be observed in each superbasin.

    Ns: int
        The frequency of barrier scaling.

    ne: int
        The minimum number of times to see a quasi-equilibrated event in each superbasin.

    isup: int
        How many steps were taken in the current superbasin.

.. seealso::

   Module :py:mod:`NeighborKMC.base.basin`
      for documentation about the superbasin.

   user_kmc - files in the tutorial/examples folder for aditional specifications.

    """

    def __init__(self, system, tend, parameters={}):

        self.system = system
        self.tend = tend
        self.parameters = parameters
        comm = MPI.COMM_WORLD
        me = comm.Get_rank()
        nprocs = comm.Get_size()
        self.mlff = None # ML-FF of this rank. Created in frm_init
        self.color = None


        self.t = 0.  #Initialize the time
        
        # UPDATE ALL EVENTS EVERY TIME!!
        self.update_all_rates = True # Shall we update all the events or just a part?!
        self.framework = None


        # Load software configuration
        self.load_options()
        if self.verbose and me == 0:
            print('-' * 50, '\n', 'MonteCoffee Simulation Initialized', '\n', '-' * 50, '\n')
            print('kMC simulation loading ...')

        # Variables connected to after analysis.
        self.mlff_called = 0
        self.mlff_performed = False
        self.times = []
        self.covered = []  # Site-occupations
        self.evs_exec = np.zeros(len(self.events))
        self.system_evolution = [[] for i in range(4)]

        self.Nstypes = list(set([s.stype for s in self.system.sites]))

        #self.used_ijk = [] # Avoid double counting events of site 1+2 vs site 2+1.

        # FRM method variables
        # --------------------------------------------------     
        self.frm_times = []  # Times of occourences
        self.frm_arg = None  # args that sort frm times
        if self.verbose and me == 0:
            print('Initializing First Reaction method lists ...')

        self.frm_init()

    def load_options(self):
        """Loads all options set in kMC_options.cfg.
        
        Instantiates a configuration parser, and loads in all
        options from *kMC_options.cfg*.

        """
        config = configparser.RawConfigParser()
        config.read('kMC_options.cfg')
        self.SaveSteps = config.getint('Parameters', 'SaveSteps', fallback = 1000)  # How often to save txt files
        self.LogSteps = config.getint('Parameters', 'LogSteps', fallback = 1)  # How often to log steps
        self.tinfinity = config.getfloat('Parameters', 'tinfinity', fallback = 1e18)  # What is considered infinite time
        self.Nspecies = config.getint('Parameters', 'Nspecies', fallback = 1)  # Number of different species in simulation.
        self.nninter = config.getint('Parameters', 'nninteractions', fallback = 1)  # depth of NN interactions
        self.verbose = config.getboolean('Options', 'Verbose', fallback = True)  # Print verbose information?
        self.write_atoms = config.getboolean('Options', 'Write_atoms', fallback = False)  # Write out the atom object
        self.save_coverages = config.getboolean('Options', 'SaveCovs', fallback = False)  # Save coverages?

    def frm_init(self):
        """Prepare to perform FRM simulation.
            
        Initializes empty rate and event lists to 
        bookkeep the FRM algorithm. The initial times
        of occurrence for each event is also calculated
        and stored.  

        """
        

        from ase.io import read,write
        # Get variables important for MPI4PY. TODO add them as a variable such as self. so it does not need to be called every time
        comm = MPI.COMM_WORLD
        me = comm.Get_rank()
        nprocs = comm.Get_size()
        self.mlff_performed = False # To track when to write data_

        
        # Make sure that the number of proc can be divided
        #if nprocs % self.parameters['nproc'] != 0:
        #    if me == 0:
        #        print("Error: Number of processes should be divisible by " + str(self.parameters['nproc']))
        #    MPI.Finalize()
        #    exit()

        cov = ''
        for g in self.system.sites: # Get coverage
            cov += str(g.covered)

        self.lastsel = 0
        self.lastother = None
        restart = False
        # Initilize storing of caluclations by either creating dictonary or read in pickle file
        if me == 0:
            if os.path.isfile('data_MLFF_1.pkl'): 
                if not getattr(self.system, 'calc', None): # main.py may already have loaded them
                    self.system.calc = load_calc()
                #pos = system_data[last_line][1]

                #with open('data_MLFF.pkl', 'rb') as f:
                    #self.system.calc = pickle.load(f)
                #print('Found a data_MLFF.pickle file!! Loading saved calculations from file')
                

                self.system.atoms = self.system.calc['framework'][1].copy()
                pos_atoms = self.system.calc[cov][1].copy()
                self.system.atoms += pos_atoms

                restart = True
            else:
                self.system.calc = {}
        else:
            self.system.calc = {}

        if os.path.isfile('data_MLFF_1.pkl'):
            restart = True
            if me == 0:
                for i in range(1, nprocs):
                    comm.send(self.system.atoms, dest=i, tag=i)
                    comm.send(self.system.calc['framework'][1].copy(), dest=i, tag=i)
            else:
                self.system.atoms = comm.recv(source=0, tag=me)
                self.framework = comm.recv(source=0, tag=me)




        self.possible_ijk = set()    # possible tubles of all possible events in same order as self.possible_times
        self.possible_times = SortedList()    # Times of all possible events
        l_used_ijk = set()         # Employed site_event_othersite paires (unique) 


        self.mlff = MLFFCalculator(self.system.atoms,comm,MLFFSettings.from_parameters(self.parameters)) # initlize lmp
        self.color = self.mlff.color
        





        if me != 0:
            
            if restart:
                if me == 1:
                    print('Found the starting structure in data_MLFF.pickle! Starting from this')
                self.mlff.load_structure(self.system.atoms) # initlize deepmdKit
                
                if me == 1:
                    comm.send(self.system.atoms, dest=0, tag=728)
                    comm.send(self.system.sites, dest=0, tag=729)

            else:
                if self.color == 1: # Do optimization on color 1
                    self.system.atoms = adjust_positions(self.system.atoms)
                    self.mlff.load_structure(self.system.atoms)
                    self.system.atoms,energy = self.mlff.minimize(self.system.atoms)
                    NH4_site_correct,_ = check_NH4_sites(self.system.atoms,self.system.sites,me)
                    if NH4_site_correct:
                        counter = 0
                        if self.mlff.is_leader:
                            print('The starting structure does not lead to a stable system!! Moving NH4+ atoms to find a stable minimum...')
                        while NH4_site_correct:
                            if self.mlff.is_leader:
                                print('Minimize No. ' + str(counter))
                            self.system.atoms,self.system.sites = recompute_neighbors(self.system.atoms,self.system.sites,me)
                            self.mlff.load_structure(self.system.atoms)
                            self.system.atoms,energy = self.mlff.minimize(self.system.atoms)
                            pos_check = copy.deepcopy(self.system.atoms)
                            check_structure(pos_check,pos_check,pos_check,me,[0,0],self.system.sites)
                            NH4_site_correct,_ = check_NH4_sites(self.system.atoms,self.system.sites,me)


                            if self.mlff.is_leader:
                                self.system.atoms.write('Final_structure.vasp',parallel=False)

                            counter += 1
                            if counter == 50:
                                print('Could not find a minimum after 50 tries')
                                comm.Abort(1)
                    

                else:
                    self.mlff.load_structure(self.system.atoms) # The first load has to be done on all proc. If not lmp(deepmdkit) will hang

                if me ==1: # Send the new structure from color 0 to the other proc
                    comm.send(self.system.atoms, dest=0, tag=728)
                    comm.send(self.system.sites, dest=0, tag=729)
        else:
            self.mlff.load_structure(self.system.atoms)
            self.system.atoms = comm.recv(source=1, tag=728)
            self.system.sites = comm.recv(source=1, tag=729)
            if 'framework' not in self.system.calc: # Fresh start. Stored structures only hold the Al and the ions
                self.system.calc['framework'] = [None, self.system.atoms[[a.symbol in ('O','Si','F','Ne') for a in self.system.atoms]]]
        if me == 0:
           for i in range(1, nprocs):
                comm.send(self.system.atoms, dest=i, tag=i)
                comm.send(self.system.sites, dest=i, tag=i)
        else:
            self.system.atoms = comm.recv(source=0, tag=me)
            self.system.sites = comm.recv(source=0, tag=me)



        if me ==0:
            print('initilizing of deepmdkit and lammps finished!')


        FF_calc = [] # Variable containing the ML-FF calculation that needs to be done
        
        # Find possible events and check if a ML-FF calculation should be carried out. Only performed on proc 0
        if me == 0:
#            print('Starting cov is ' + str(cov))
            for j, e in enumerate(self.events): # Loop through events
                for i, s in enumerate(self.system.sites): # Loop through the sites
                    NNcur = self.system.neighbors[i]
                    if not e.get_involve_other():
                        if e.possible(self.system, i, NNcur[0]):
                            rcur,MLFF = e.get_rate(self.system, i, NNcur[0],False) # Rate is called!!
                            if MLFF: # do a mlff calc
                                FF_calc.append([i,j,NNcur[0],rcur])
                            else: # if not then add time
                                u = random.uniform(0, 1)
                                self.possible_times.add((self.t - np.log(u) / rcur, (i, j, NNcur[0])))
                                self.possible_ijk.add((i, j, NNcur[0]))
                        l_used_ijk.add((i,j,NNcur[0])) 
                    else:
                        for k, other_site in enumerate(NNcur):
                            used_ijk = True if (i,j,other_site) in l_used_ijk or (other_site,j,i) in l_used_ijk else False # Check if the pair is already in list
                            # if e.possible(self.system, i, other_site) and not used_ijk:
                            if e.possible(self.system, i, other_site): # and not used_ijk:
                                rcur,MLFF = e.get_rate(self.system, i, other_site,False) #Rate is called!
                                if MLFF: # do a mlff calc
                                    FF_calc.append([i,j,other_site,rcur])
                                else: # if not then add time
                                    u = random.uniform(0,1)
                                    self.possible_times.add((self.t - np.log(u) / rcur,(i,j,other_site)))
                                    self.possible_ijk.add((i,j,other_site))
        
                            if (i,j,other_site) not in l_used_ijk and (other_site,j,i) not in l_used_ijk: # Create unique pair-event tuple list  
                                l_used_ijk.add((i,j,other_site)) 
        else:
            FF_calc = None
      
        # Transfer variable to other proccessors
        # TO DO: Can I get bcast or some other communcation transfer method to work. Should be faster
        if me == 0:
            cov = ''
            for g in self.system.sites:
                cov += str(g.covered)
            # master process sends data to worker processes by
            # going through the ranks of all worker processes
            for i in range(1, nprocs):
                comm.send(FF_calc, dest=i, tag=i)
                comm.send(cov, dest=i, tag=i)
            # worker processes
        else:
            # each worker process receives data from master process
            FF_calc = comm.recv(source=0, tag=me)
            cov = comm.recv(source=0, tag=me)
        
        
        

        # Start ML-FF calculations if there is any to do
        if FF_calc:
            rates = self.mlff.run_tasks(self.system.atoms,FF_calc,cov,self.system.sites)
            self.mlff_performed = True
        else:
            rates = None

        # Update counter for number of ML-FF calculations done
        self.mlff_called += len(FF_calc)
        

        # Store the calculations, which is the energy and structure
        if me == 0:
            counter = 0
            if FF_calc:
                for g in rates:
                    #self.system.calc[g[0]] = [g[1],g[2],g[3]] # DEBUG
                    # Delte F, Si and O atoms before saving them
                    s = g[2]                      # ASE structure
                    del_idxs = [i for i, a in enumerate(s) if a.symbol in ("F", "Si", "O","Ne")]
                    del s[del_idxs]              # delete all those atoms at once
                    self.system.calc[g[0]] = [g[1],s,g[3]]


            # Compute the rates that was waiting for ml-ff calculation
            for g in FF_calc:
                rcur, MLFF = e.get_rate(self.system, g[0], g[2], True)  # Rate is called!
                u = random.uniform(0, 1)
                self.possible_times.add((self.t - np.log(u) / rcur, (g[0],g[1],g[2])))
                self.possible_ijk.add((g[0],g[1],g[2]))
                counter += 1

            
        
        if me == 0:
            print('Everything is intilized!! starting frm_update')
        

    def find_index(self, check_tuble):
        for ind, element in enumerate(self.possible_times):
            if check_tuble == element[1]:
                return ind               


    def frm_update(self):
        
        comm = MPI.COMM_WORLD
        me = comm.Get_rank()
        nprocs = comm.Get_size()



        # Return all sites
        search = list(range(len(self.system.sites)))

        updated_set = set()

        # UPDATE ALL EVENTS EVERY TIME!
        self.possible_ijk = set()    # possible tubles of all possible events in same order as self.possible_times
        self.possible_times = SortedList()    # Times of all possible events

        FF_calc = [] # Variable containing the ML-FF calculation that needs to be done
        cov = ''
        

        self.mlff_performed = False
        #print('Color ' + str(self.color) + ' has a cov of ' + str(cov))
        # Change to new structure and send it to the other nodes - MAYBE DO THIS ELSEWHERE
        if me == 0:
            for g in self.system.sites:
                cov += str(g.covered)
            print('The coverage is now:' + str(cov))
            
            #self.system.atoms.write('output_structure/beforestep.vasp',parallel=False)
            #self.system.atoms = self.system.calc[cov][1]
            #self.system.atoms.write('output_structure/afterstep.vasp',parallel=False)

            self.system.atoms.write('output_structure/beforestep.vasp',parallel=False)
            self.system.atoms = self.system.calc['framework'][1].copy()
            pos_atoms = self.system.calc[cov][1].copy()
            self.system.atoms += pos_atoms
            self.system.atoms.write('output_structure/afterstep.vasp',parallel=False)


            if self.system.calc[cov][0] > 100:
                print('A step with energy higher than 100 has been chosen. Should not happen!!. See frm_update')
                exit()
            if cov != self.system.calc[cov][2]:
                print('A step that for cov of ' + str(cov) + ' lead to the cov of ' + str(self.system.calc[cov][2]) + '. Updating the the NH4 indice and writing to cov!!')
                

                

                cov = self.system.calc[cov][2]
                with open("coverages.txt", "a") as f:
                    cov_print = spaced = ' '.join(str(cov))
                    f.write("x" + str(cov_print) + '\n')
            
            self.system.sites = recompute_neighbors_all_site(self.system.atoms,self.system.sites,cov)

            error_check,no_error = check_Cu_sites(self.system.atoms,self.system.sites,me)
            if error_check:
                print('Coverage for Cu is wrong!!!')




            for i in range(1, nprocs):
                comm.send(self.system.atoms, dest=i, tag=i)
                #comm.send(self.system.calc, dest=i, tag=i)
                comm.send(cov, dest=i, tag=i)
                comm.send(self.system.sites,dest=i,tag=i)
        else:
            self.system.atoms = comm.recv(source=0, tag=me)
            #self.system.calc = comm.recv(source=0, tag=me)
            #print('Cov for' + str(me) +  ' before sending is ' + str(cov))
            cov = comm.recv(source=0, tag=me)
            #print('Cov for' + str(me) +  ' after sending is ' + str(cov))
            self.system.sites = comm.recv(source=0, tag=me)



        
        if me == 0:
            for j, e in enumerate(self.events):
                if not e.get_involve_other():
                    for i in search:
                        NNcur = self.system.neighbors[i][0]
                        poss_now = e.possible(self.system, i, NNcur)
                        in_poss_list = True if (i,j,NNcur) in self.possible_ijk else False
                        if poss_now and not in_poss_list:
                            rcur,MLFF = e.get_rate(self.system, i, NNcur,False)
                            if MLFF:  # do a mlff calc
                                FF_calc.append([i,j,NNcur,rcur])
                            else:
                                u = random.uniform(0,1)
                                new_time = self.t - np.log(u) / rcur
                                self.possible_times.add((new_time,(i,j,NNcur)))
                                self.possible_ijk.add((i,j,NNcur))
                        if not poss_now and in_poss_list:
                            poss_index = self.find_index((i,j,NNcur))
                            self.possible_ijk.remove((i,j,NNcur))
                            del self.possible_times[poss_index]
    
                else:
                    for i in search:
                        for other in self.system.neighbors[i]:
                            # Modified
                            i_ass = min([i,other])
                            other_ass = max([i,other])
                            i_ass = i
                            other_ass = other
                            updated = True if (i_ass,j,other_ass) in updated_set else False 
                            if not updated:
                                poss_now = e.possible(self.system, i_ass, other_ass)
                                in_poss_list = True if (i_ass,j,other_ass) in self.possible_ijk else False
                                if poss_now and not in_poss_list:
                                    rcur,MLFF = e.get_rate(self.system, i_ass, other_ass,False)
                                    if MLFF:  # do a mlff calc
                                        FF_calc.append([i_ass,j,other_ass,rcur])
                                    else:
                                        u = random.uniform(0,1)
                                        new_time = self.t - np.log(u) / rcur
                                        self.possible_times.add((new_time,(i_ass,j,other_ass)))
                                        self.possible_ijk.add((i_ass,j,other_ass))
            
                                if not poss_now and in_poss_list:
                                    poss_index = self.find_index((i_ass,j,other_ass))
                                    self.possible_ijk.remove((i_ass,j,other_ass))
                                    del self.possible_times[poss_index]
    
                                updated_set.add((i_ass,j,other_ass))
        else:
            FF_calc = None




        # TODO: Can I get bcast or some other communcation transfer method to work
        if me == 0:
            # master process sends data to worker processes by
            # going through the ranks of all worker processes
            for i in range(1, nprocs):
                comm.send(FF_calc, dest=i, tag=i)
            # worker processes
        else:
            # each worker process receives data from master process
            FF_calc = comm.recv(source=0, tag=me)
        # Start calculations
        if FF_calc:
            rates = self.mlff.run_tasks(self.system.atoms,FF_calc,cov,self.system.sites)
            self.mlff_performed = True
        else:
            rates = None
        



        if me == 0:
            # Update counter
            self.mlff_called += len(FF_calc)
            # Write the energy and structure to the dictonary
            counter = 0
            if FF_calc:
                for g in rates:
                    s = g[2]                      # ASE structure
                    del_idxs = [i for i, a in enumerate(s) if a.symbol in ("F", "Si", "O","Ne")]
                    del s[del_idxs]              # delete all those atoms at once
                    self.system.calc[g[0]] = [g[1],s,g[3]]


            # Compute the rates that was waiting for ml-ff calc
            for g in FF_calc:
                rcur, MLFF = e.get_rate(self.system, g[0], g[2], True)  # Rate is called!
                u = random.uniform(0, 1)
                self.possible_times.add((self.t - np.log(u) / rcur, (g[0],g[1],g[2])))
                self.possible_ijk.add((g[0],g[1],g[2]))
                counter += 1


    def frm_step(self):
        comm = MPI.COMM_WORLD
        me = comm.Get_rank()
        nprocs = comm.Get_size()
        if me == 0:
            # Choose the first reaction if possible
            site = self.possible_times[0][1][0]  # The site to do event.
            othersite = self.possible_times[0][1][2]
            self.lastsel = int(site)
            self.lastother = int(othersite)
            evtype = self.possible_times[0][1][1]
            if self.events[evtype].possible(self.system, site, othersite):
                # Event is possible, change state
                
                self.events[evtype].do_event(self.system, site, othersite)
                self.t = self.possible_times[0][0]  # Update time

                #Data logging
                self.evs_exec[evtype] += 1 #Total executed events
                self.system_evolution[0].append(int(self.system.sites[site].ind))
                self.system_evolution[1].append(int(self.system.sites[othersite].ind))
                self.system_evolution[2].append(int(evtype))
                self.system_evolution[3].append(float(self.t))
            
                # Delete the executed event from possible list
                self.possible_ijk.remove((site,evtype,othersite))
                del self.possible_times[0]

            else:
#               del self.possible_times[self.frm_arg]
#               del self.possible_ijk[self.frm_arg]
                # New first reaction must be determined
                print ('Event', evtype, self.possible_times[0])
                print (self.system.sites[site].covered, self.system.sites[othersite].covered)
                raise Warning("Impossible event were next in que and was attempted")
        else:
            self.lastsel = None
            self.lastother = None
            evtype = None
            self.events = None
            self.t = None
            self.evs_exec = None
            self.system_evolution = None
            self.possible_ijk = None
        # Send all information around - TODO - maybe not everything needs to be send? try with BCAST or similiar some point
        
        if me == 0:
            # master process sends data to worker processes by
            # going through the ranks of all worker processes
            for i in range(1, nprocs):
                comm.send(self.lastsel, dest=i, tag=i)
                comm.send(self.lastother,dest=i, tag=i)
                comm.send(evtype, dest=i, tag=i)
                comm.send(self.events, dest=i, tag=i)
                comm.send(self.t, dest=i, tag=i)
                comm.send(self.evs_exec, dest=i, tag=i)
                comm.send(self.system_evolution, dest=i, tag=i)
                comm.send(self.possible_ijk, dest=i, tag=i)

            # worker processes
        else:
            # each worker process receives data from master process
            self.lastsel = comm.recv(source=0, tag=me)
            self.lastother = comm.recv(source=0, tag=me)
            evtype = comm.recv(source=0, tag=me)
            self.events = comm.recv(source=0, tag=me)
            self.t = comm.recv(source=0, tag=me)
            self.evs_exec = comm.recv(source=0, tag=me)
            self.system_evolution = comm.recv(source=0, tag=me)
            self.possible_ijk = comm.recv(source=0, tag=me)
        # Change the atoms in the atoms list
        #print([self.system.sites[self.lastsel].atoms,self.system.sites[self.lastother].atoms])
        self.system.sites[self.lastsel].atoms, self.system.sites[self.lastother].atoms = self.move_numbers(self.system.sites[self.lastsel].atoms,self.system.sites[self.lastother].atoms,self.system.sites[self.lastother].atom_type,me)
        #print([self.system.sites[self.lastsel].atoms,self.system.sites[self.lastother].atoms])


        self.frm_update()

        

    def update_T(self, T_init):
        heat_rate = self.parameters['heating_rate']
        upd_T = T_init + self.t*heat_rate
        self.parameters['T'] = upd_T
        self.update_all_rates = True


    def move_numbers(self, A, B,type_n,me_i): # Update the atom list correctly
        if type_n == 'Cu':
            if not A[0]:
                print('No atom to move... aborting')
                exit()


            # Move the first number from A to B
            if A and A[0]:  # Check if A is not empty and A[0] has a value
                num = A[0].pop(0)  # Remove the first number from A[0]
                if not B[0]:  # If B[0] is empty
                    B[0].append(num)
                elif not B[1]:  # If B[0] has a value but B[1] is empty
                    B[1].append(num)

            # Move the number from A[1] to A[0] if A[1] has a value
            if A and A[1]:  # Check if A is not empty and A[1] has a value
                num = A[1].pop(0)  # Remove the first number from A[1]
                A[0].append(num)  # Move the number to A[0]

            return A, B
        elif type_n == 'N':
            if me_i == 0:
                print(A)
                print(B)
            # Move the first number from A to B
            if not A[0]:
                print('No atom to move... aborting')
                exit()

            B[0].append(A[0].pop(0))
            

            if me_i == 0:
                print(A)
                print(B)
            return A, B

        else:
            print('Atom type ' + str(type_n) + ' not supported. Aborting')
            exit()
        return
 
    def load_events(self):
        """Loads events (abstract method).

        This method must be overridden by the child class in user_kmc.NeighborKMC.

        Raises
        ---------
        NotImplementedError:
            If called.
          
        """
        raise NotImplementedError('''User needs to define load_events
                                 method in derived NeighborKMC class''')

    def run_kmc(self):
        """Runs the kMC simulation (abstract method)

        This method must be overridden by the child class in user_kmc.NeighborKMC.

        Raises
        ---------
        NotImplementedError:
            If called.

        """
        raise NotImplementedError('''User needs to define run_kmc method 
                                          in derived NeighborKMC class''')

"""
Everything related to calculating the rate constants using a machine learning force field (ML-FF).

The ML-FF is a DeePMD-kit model that is evaluated through LAMMPS. When doing the structural optimisation ,the
MPI ranks are split into colors, where rank 0 is the master and every other group of CPU's is an worker doing
ML-FF optimizations with its own LAMMPS instance. The master hands out minimisations to the workers and receives
the relaxed structures and energies back.

Overview
--------
MLFFSettings  Settings read from the kMC parameters dict (model file, minimization steps, ...)
RankLayout    Maps MPI ranks to colors (workers)
MLFFCalculator Owns the LAMMPS instance of a worker and runs the master/worker task farm

Besides the classes, the module contains functions to move the [Cu(NH3)2]+ and NH4+ ions between
sites before a minimization (move_ion, apply_repelling_force) and to move the structure into its 
new position according to the coverages and to check if no complex/molecule broke or flew away (recompute_neighbors*, check_*).

This module was polished and documented with the help of Claude.
"""

import copy
import importlib.util
import os
import pickle
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from lammps import lammps
from mpi4py import MPI
from numpy.linalg import norm


# MPI message tags used by the master/worker task farm
TAG_REQUEST = 1000  # Worker asks the master for a task
TAG_TASK = 2000     # Master sends a task (or None to stop) to a worker
TAG_RESULT = 3000   # Worker sends a result to the master

# LAMMPS atom types. Type 8 is the virtual (Wannier centroid) atom bonded to each Al for DPLR. Have to be changed if another catalyst is studied.
LMP_TYPES = {'O': 1, 'Si': 2, 'Al': 3, 'Cu': 4, 'N': 5, 'C': 6, 'H': 7}
LMP_SYMBOLS = ['O', 'Si', 'Al', 'Cu', 'N', 'C', 'H', 'He']  # Symbols. Virtual atoms are read back as He. Since there are two types of hydrogen, they are set to C and H
LMP_CHARGES = {3: -0.91, 4: 0.53, 5: -1.22, 6: 0.5325, 7: 0.47} # Should be the same as used in the training
LMP_MASSES = {1: 15.999, 2: 28.0855, 3: 26.981539, 4: 63.546, 5: 14.0067, 6: 3, 7: 3, 8: 26.981539}  # H (type 6 and 7) are set to 3 allowing a larger timestep
FREEZE_TYPES = '1 2 3 8'  # The framework (O, Si, Al and virtual atoms) is kept fixed during minimization
MAX_MIN_EVALS = 100000  # Max force evaluations in a minimization. Set high so only max_min_steps limits it.
SITE_SYMBOLS = ('F', 'Ne')  # Dummy atoms for the Cu (F) and NH4+ (Ne) sites. They are not sent to LAMMPS, but only used for structural manipulation by ASE
CU_COMPLEX_SIZE = 9   # Cu, N, N, 6 H
NH4_SIZE = 5          # N, 4 H (written as C)


def abort(message):
    """Print a message and stop all MPI ranks. Calling exit() on one rank would leave the others hanging."""
    print(message, flush=True)
    MPI.COMM_WORLD.Abort(1)


def load_calc(base_filename="data_MLFF"):
    """
    Load the stored ML-FF calculations (system.calc) from data_MLFF_1.pkl, data_MLFF_2.pkl, ...

    :return: dict of coverage -> [energy, structure, coverage after minimization], plus 'framework'
    """
    calc = {}
    i = 1
    while os.path.exists(f"{base_filename}_{i}.pkl"):
        filename = f"{base_filename}_{i}.pkl"
        with open(filename, "rb") as f:
            chunk = pickle.load(f)
        calc.update(chunk)
        print(f"Loaded {len(chunk)} items from {filename}")
        i += 1
    return calc


@dataclass
class MLFFSettings:
    """
    Settings for the ML-FF calculations.

    :param model_file: DeePMD model
    :param max_min_steps: Maximum number of steps for the structural optimisation
    :param force_tol: Force criteria for stopping of the structural optimisation (eV/Å)
    :param cores_per_worker: Number of MPI ranks sharing one LAMMPS instance
    :param first_worker_cores: Number of ranks of the first worker. None means cores_per_worker. 
    :param plugin: DeePMD LAMMPS plugin. None finds the plugin of the installed deepmd-kit
    :param log_dir: Directory for the LAMMPS output files
    :param debug: If True no minimization is done and the energy is a random number between -0.5 and 0.5. Only used for debugging
    """
    model_file: str = 'FF_CuCHA_minimal.pb'
    max_min_steps: int = 2000
    force_tol: float = 0.05
    cores_per_worker: int = 1
    first_worker_cores: int = None
    plugin: str = None
    log_dir: str = 'log'
    debug: bool = False

    @classmethod
    def from_parameters(cls, parameters):
        """Create the settings from the kMC parameters dict. Keys that are not given keep their default."""
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: parameters[k] for k in keys if k in parameters})

    def plugin_path(self):
        if self.plugin is not None:
            return self.plugin
        spec = importlib.util.find_spec('deepmd')
        if spec is not None:
            path = os.path.join(os.path.dirname(spec.origin), 'lib', 'libdeepmd_lmpplugin.so')
            if os.path.isfile(path):
                return path
        return 'libdeepmd_lmp.so'  # Older deepmd-kit installations (e.g. the cluster)


class RankLayout:
    """
    Maps MPI ranks to colors. Rank 0 is the kMC master and has color 0.
    The other ranks are split into workers (colors 1, 2, ...) of cores_per_worker ranks each.
    The first worker can be given a different size with first_worker_cores, due to unveven numbers of cores.
    """

    def __init__(self, nprocs, cores_per_worker, first_worker_cores=None):
        if nprocs < 2:
            abort('The ML-FF needs at least 2 MPI ranks: 1 master and 1 worker')
        self.nprocs = nprocs
        self.cores_per_worker = cores_per_worker
        self.first_worker_cores = first_worker_cores or cores_per_worker

    def color(self, rank):
        if rank == 0:
            return 0
        if rank <= self.first_worker_cores:
            return 1
        return (rank - self.first_worker_cores - 1) // self.cores_per_worker + 2

    def leader(self, color):
        """First rank of a color. It talks to the master on behalf of the color"""
        if color <= 1:
            return color
        return self.first_worker_cores + 1 + (color - 2) * self.cores_per_worker

    def worker_colors(self):
        return list(range(1, self.color(self.nprocs - 1) + 1))


class MLFFCalculator:
    """
    The ML-FF of one MPI rank.

    Must be created on all ranks at the same time since it splits the communicator. The master (rank 0),
    also needs to start a LAMMPS instance on its own communicator. The master only loads the first structure
    and never minimizes, since older deepmd-kit versions hang if not all ranks load the model.

    :param atoms: ASE structure that defines the simulation box
    :param comm: MPI communicator (MPI.COMM_WORLD)
    :param settings: MLFFSettings
    """

    def __init__(self, atoms, comm, settings):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.settings = settings
        self.layout = RankLayout(comm.Get_size(), settings.cores_per_worker, settings.first_worker_cores)
        self.color = self.layout.color(self.rank)
        self.split_comm = comm.Split(color=self.color, key=self.rank)
        self.is_master = self.rank == 0
        self.is_leader = self.split_comm.Get_rank() == 0  # The master is the leader of color 0
        self.lmp = None
        self._model_loaded = False

        for directory in (settings.log_dir, 'output_structure'):
            os.makedirs(directory, exist_ok=True)

        if self.is_master:
            print('Starting initializing ml-ff with ' + str(len(self.layout.worker_colors())) + ' workers. Model: '
                  + settings.model_file + ', max minimization steps: ' + str(settings.max_min_steps)
                  + ', plugin: ' + settings.plugin_path())
        self._start_lammps(atoms.cell)  # Also on the master. See the class docstring
        if self.is_master:
            print('Finished initializing')

    def close(self):
        """Free LAMMPS and the split communicator. Call on all ranks before MPI.Finalize()"""
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None
        if self.split_comm is not None:
            self.split_comm.Free()
            self.split_comm = None

    # ------------------------------------------------------------------
    # LAMMPS
    # ------------------------------------------------------------------
    def _start_lammps(self, ase_cell):
        """Start LAMMPS and create the simulation box. The model is loaded with the first structure."""
        cell, _ = convert_cell(ase_cell)
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-in", "none"], comm=self.split_comm)
        self.lmp.commands_list([
            'units           metal',  # Å, ps, eV, etc.
            'boundary        p p p',
            'atom_style      full',
            'plugin load     ' + self.settings.plugin_path(),
            'neighbor        2.0 bin',
            'neigh_modify    every 10 delay 0 check no exclude type 3 8',
            f'region mybox prism 0.0 {cell[0][0]} 0.0 {cell[1][1]} 0.0 {cell[2][2]} {cell[0][1]} {cell[0][2]} {cell[1][2]} units box',
            'create_box 8 mybox bond/types 1 extra/bond/per/atom 1',
        ])

    def _load_model(self):
        """Load the DeePMD DPLR model. Can only be done once atoms exist."""
        model = self.settings.model_file
        self.lmp.commands_list([
            f'dump            1 all custom 500 {self.settings.log_dir}/Proc{self.color}_zeolite_ase_read.lammpstrj id element x y z',
            'dump_modify     1 element O Si Al Cu N H H He',
            'group           real_atom type 1 2 3 4 5 6 7',
            'group           virtual_atom type 8',
            f'pair_style      deepmd {model}',
            'pair_coeff      * *',
            'bond_style      zero',
            'bond_coeff      *',
            'special_bonds   lj/coul 1 1 1 angle no',
            'kspace_style    pppm/dplr 1e-5',
            'kspace_modify   gewald 0.066 diff ik mesh 32 32 32',
            f'fix             0 all dplr model {model} type_associate 3 8 bond_type 1',
            'fix_modify      0 virial yes',
            'thermo 25',
            'thermo_style custom step pe fnorm fmax',
            'thermo_modify  lost ignore',
        ])
        self._model_loaded = True

    def load_structure(self, pos_i):
        """
        Load an ASE structure into LAMMPS, replacing any atoms already there.

        Adds a virtual atom bonded to each Al atom and sets charges and masses. The site atoms (F, Ne)
        are skipped. The first call also loads the model. On the master only the first call does something,
        since the master only needs to load the model.

        :param pos_i: ASE structure
        """
        if self.lmp is None or (self.is_master and self._model_loaded):
            return
        if self.lmp.get_natoms() != 0:  # The model delete all atoms and create them again. Would be faster to move the atoms instead, however this step is not the bottleneck of the code anyway.
            self.lmp.command('delete_atoms group all bond yes compress no')

        pos_i.wrap()  # LAMMPS does not accept atoms outside the box
        cell, trans = convert_cell(pos_i.cell)
        box = self.lmp.extract_box()
        if not np.allclose([box[1][0], box[1][1], box[1][2]], [cell[0][0], cell[1][1], cell[2][2]]):
            abort('Simulation box is not the same!! \nOld box: ' + str(box) + '\nNew cell: ' + str(cell))

        lammps_coor = np.dot(pos_i.positions, trans.T)
        atom_cmds = []
        bond_cmds = []
        atom_id = 1
        for symbol, (x, y, z) in zip(pos_i.get_chemical_symbols(), lammps_coor):
            if symbol in SITE_SYMBOLS:
                continue
            atom_cmds.append(f'create_atoms {LMP_TYPES[symbol]} single {x} {y} {z}')
            if symbol == 'Al':  # Virtual atom on top of the Al, bonded to it
                atom_cmds.append(f'create_atoms 8 single {x} {y} {z}')
                bond_cmds.append(f'create_bonds single/bond 1 {atom_id} {atom_id + 1}')
                atom_id += 1
            atom_id += 1
        self.lmp.commands_list(atom_cmds)
        self.lmp.commands_list(bond_cmds)
        self.lmp.commands_list([f'set type {t} charge {q}' for t, q in LMP_CHARGES.items()])
        self.lmp.commands_list([f'mass {t} {m}' for t, m in LMP_MASSES.items()])

        if not self._model_loaded:
            self._load_model()

    def minimize(self, pos):
        """
        Minimize the structure in LAMMPS with the framework frozen. load_structure must be called first.

        :param pos: The ASE structure that was loaded. Used for the cell and the site atoms
        :return: minimized ASE structure (site atoms first) and the potential energy
        """
        s = self.settings
        self.lmp.commands_list([
            'group freeze type ' + FREEZE_TYPES,
            'fix 1 freeze setforce 0.0 0.0 0.0',
            'min_style cg',
            'min_modify dmax 0.2',
            f'minimize 0 {s.force_tol} {s.max_min_steps} {MAX_MIN_EVALS}',
        ])
        return self._extract_structure(pos), self.lmp.get_thermo('pe')

    def _extract_structure(self, pos):
        """
        Convert the structure in LAMMPS to an ASE structure. The site atoms (F, Ne) are taken from pos and
        put first, followed by the LAMMPS atoms in the order of their ids. Virtual atoms are removed.
        """
        coords = np.array(self.lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
        types = self.lmp.gather_atoms("type", 0, 1)
        symbols = [LMP_SYMBOLS[t - 1] for t in types]

        _, trans = convert_cell(pos.cell)
        atoms = Atoms(symbols, positions=np.dot(coords, trans), cell=pos.cell, pbc=True)
        del atoms[[a.symbol == 'He' for a in atoms]]
        return pos[[a.symbol in SITE_SYMBOLS for a in pos]] + atoms

    # ------------------------------------------------------------------
    # Master-worker setup
    # ------------------------------------------------------------------
    def run_tasks(self, pos, movement, coverage, sites):
        """
        Minimize the structures of all coverages that is requested. This function must be called on all ranks.

        The master hands out one structure at a time to each worker and collects the results. A worker
        moves the ion from the current position to the new position, minimizes and sends back the result.

        :param pos: Current ASE structure
        :param movement: List of [site, event, other_site, coverages]. The coverages is the new structure, which are the tasks
        :param coverage: Current coverage string
        :param sites: Current sites
        :return: On the master a list consiting of [coverage, energy, structure, coverage after minimization], that it recived from the workers.
            None on the workers
        """
        if self.is_master:
            tasks = list(dict.fromkeys(cov for m in movement for cov in m[3]))  # Unique, keeping the order
            return self._master_loop(tasks)
        self._worker_loop(pos, coverage, sites)
        return None

    def _master_loop(self, tasks):
        print('There are ' + str(len(tasks)) + ' minimizations that has to be done')
        queue = list(tasks)
        results = {}
        active_leaders = {self.layout.leader(c) for c in self.layout.worker_colors()}
        status = MPI.Status()
        # A leader always asks for a new task after sending a result, so when every leader has been
        # told to stop, all results have arrived.
        while active_leaders:
            message = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            if status.Get_tag() == TAG_REQUEST:
                if queue:
                    self.comm.send(queue.pop(0), dest=source, tag=TAG_TASK)
                else:
                    self.comm.send(None, dest=source, tag=TAG_TASK)
                    active_leaders.discard(source)
            else:  # TAG_RESULT
                results.setdefault(message[0], message)
                left = len(tasks) - len(results)
                if left and left % 10 == 0:
                    print('There is ' + str(left) + ' tasks left')
        print('All ML-FF tasks finished')
        return list(results.values())

    def _worker_loop(self, pos_org, coverage, sites):
        while True:
            task = None
            if self.is_leader:
                self.comm.send(None, dest=0, tag=TAG_REQUEST)
                task = self.comm.recv(source=0, tag=TAG_TASK)
            task = self.split_comm.bcast(task, root=0)
            if task is None:
                break

            pos_final, energy, move = self._relax(pos_org, task, coverage, sites)

            if self.is_leader:
                _, cov_new = recompute_neighbors_all(pos_final, sites)
                if cov_new != task:
                    print('a opt relaxed to another minimum! cov_old is ' + str(task) + ' and new is ' + str(cov_new))
                result = [task, energy, pos_final, cov_new]
                if cov_new == coverage:
                    print('Found a minimum that relaxes to the original pos. Seeting energy to 10**10')
                    result[1] = 10**10 # By setting the energy to a very high value, the time for this step would never be the lowest and therefor never excucuted
                self.comm.send(result, dest=0, tag=TAG_RESULT)
                # Do checks to see if everything is ok and no atoms/complexes is exploded
                error_check, no_error = check_Cu_sites(pos_final, sites_after_move(sites, move), self.rank)
                if error_check:
                    print(str(no_error) + ' Cu ions did not match their description. Task ' + str(task))
                check_structure(pos_final, pos_final, pos_org, self.rank, move, sites)

    def _relax(self, pos_org, task, coverage, sites):
        """Move the ion from its initial position to requested position and minimize. Returns (structure, energy, move)"""
        pos = copy.deepcopy(pos_org)
        move = None
        if task != coverage: 
            move = find_digit_difference(coverage, task)
            start, end = move[0], move[1]
            pos = move_ion(pos, [start, end], [sites[start].atoms, sites[end].atoms], sites[start].atom_type,
                           self.is_leader, sites)
        if self.settings.debug:
            return pos, np.random.uniform(-0.5, 0.5), move
        self.load_structure(pos)
        pos_final, energy = self.minimize(pos) # Start minimsation
        return pos_final, energy, move


def convert_cell(ase_cell):
    """
    Adapted from ASE's LAMMPS coordinate transformation
    implementation:
    https://gitlab.com/ase/ase

    Convert a parallelepiped (forming right hand basis)
    to lower triangular matrix LAMMPS can accept. This
    function transposes cell matrix so the bases are column vectors

    .. deprecated:: 3.23.0
        Please use
        :func:`~ase.calculators.lammps.coordinatetransform.calc_rotated_cell`.
    """
    cell = ase_cell.T

    # rotate bases into triangular matrix
    tri_mat = np.zeros((3, 3))
    A = cell[:, 0]
    B = cell[:, 1]
    C = cell[:, 2]
    tri_mat[0, 0] = norm(A)
    Ahat = A / norm(A)
    AxBhat = np.cross(A, B) / norm(np.cross(A, B))
    tri_mat[0, 1] = np.dot(B, Ahat)
    tri_mat[1, 1] = norm(np.cross(Ahat, B))
    tri_mat[0, 2] = np.dot(C, Ahat)
    tri_mat[1, 2] = np.dot(C, np.cross(AxBhat, Ahat))
    tri_mat[2, 2] = norm(np.dot(C, AxBhat))

    # create and save the transformation for coordinates
    volume = np.linalg.det(ase_cell)
    trans = np.array([np.cross(B, C), np.cross(C, A), np.cross(A, B)])
    trans /= volume
    coord_transform = np.dot(tri_mat, trans)

    return tri_mat, coord_transform


# ----------------------------------------------------------------------
# Moving ions before a minimization
# ----------------------------------------------------------------------
# Positions relative to the site atom. Cu complex: Cu, N in NH3_1, N in NH3_2, 3 H in NH3_1, 3 H in NH3_2
CU_TEMPLATE = np.array([
    [0.0, 0.0, 0.0],
    [2.1039964980873522, -0.0273411537525956, 0.2068627931803984],
    [-2.0230930026026397, 0.2152016173712335, 0.2584719485902820],
    [2.3217850446840433, 0.9657452529124215, 0.2215750478840622],
    [2.4182585653125823, -0.4022621064855209, 1.0982337638206616],
    [2.7019686660405631, -0.4407437655526302, -0.5040493968299049],
    [-2.146830110431101, 1.1777348296322845, -0.0449878609573300],
    [-2.5343575311401532, -0.3616695373628993, -0.4046244453751968],
    [-2.5107150509413096, 0.1304522287907650, 1.1466784865668014],
])
NH4_TEMPLATE = np.array([
    [0.0, 0.0, 0.0],
    [0.9428, 0.0000, 0.3333],
    [-0.4714, 0.8165, 0.3333],
    [-0.4714, -0.8165, 0.3333],
    [0.0000, 0.0000, -1.0000],
])

REPULSION_MIN_DIST = 1.7     # Minimum distance between atoms (Å)
REPULSION_MIN_DIST_CU = 2.5  # Minimum distance between a Cu and the moved NH4+ N (Å)
REPULSION_MAX_STEPS = 30
REPULSION_STEP = 1.0         # Step length (Å) per close pair and step


def _neighbor_list(pos, cutoff):
    nl = NeighborList(cutoffs=[cutoff / 2] * len(pos), self_interaction=False, bothways=True)
    nl.update(pos)
    return nl


def _neighbors(pos, index, cutoff):
    return _neighbor_list(pos, cutoff).get_neighbors(index)[0]


def _ion_group(pos, first, size):
    """[[index, symbol], ...] of the size atoms of an ion starting at index first"""
    return [[first + i, pos[first + i].symbol] for i in range(size)]


def _place(pos, group, template, center, dz=0.0):
    """Put the atoms of group at template positions around atom center, shifted dz in z"""
    for (index, _), offset in zip(group, template):
        pos.positions[index] = offset + [0, 0, dz] + pos.positions[center]


def _site_of_atom(sites, atom_index):
    """Index of the site that lists atom_index among its atoms"""
    for i, site in enumerate(sites):
        if any(atom_index in atom_list for atom_list in site.atoms):
            return i
    abort('Atom ' + str(atom_index) + ' is not assigned to any site. See ML_FF._site_of_atom')


def apply_repelling_force(pos_i, atom_indices_1, atom_indices_2, atoms_list, N_atom_index=None):
    """
    Move the complexes until no atom is closer than REPULSION_MIN_DIST to another atom. This is required to avoid
    the ML-FF optimisation exploding due to atoms being to close. 

    Is done for either one Cu complex or two.

    For atoms closer than the minimum distance, both atoms get a push of REPULSION_STEP along the pair
    vector. Each group is moved by the average push of its atoms. If a group's Cu is closer than
    REPULSION_MIN_DIST_CU to the N of N_atom_index (the moved NH4+), only the push of that Cu is used.
    This is to avoid the algorithm to get stuck and being pushed into a different site.
    Atoms not involved in the optimisation is frozen

    :param pos_i: ASE structure.
    :param atom_indices_1: First group as [[index, symbol], ...]. 
    :param atom_indices_2: Second group. May be empty if only one Cu ion are present in the zeolite cage
    :param atoms_list: Indices of the atoms to check distances between
    :param N_atom_index: Group of the moved NH4+. Only its first atom (N) is used
    :return: pos_i and True if the distances are fine, False if not within REPULSION_MAX_STEPS steps.
    If the optimisation did not succed, the optimisation will still take place and the structure will anyway
    be checked afterwards to see if something went wrong.
    """
    finished = True
    group_1 = {g[0] for g in atom_indices_1}
    group_2 = {g[0] for g in atom_indices_2}
    cu_1 = atom_indices_1[0][0] if atom_indices_1 else None
    cu_2 = atom_indices_2[0][0] if atom_indices_2 else None
    n_atom = N_atom_index[0][0] if N_atom_index else None
    moving = group_1 | group_2
    symbols = pos_i.get_chemical_symbols()
    atoms_list = list(atoms_list)

    for step in range(REPULSION_MAX_STEPS):
        forces = np.zeros((len(pos_i), 3))
        goodsound = True
        only_Cu_1 = False
        only_Cu_2 = False
        for i, a in enumerate(atoms_list[:-1]):
            others = atoms_list[i + 1:]
            vectors = pos_i.get_distances(a, others, mic=True, vector=True)
            for b, difference in zip(others, vectors):
                if (a not in moving and b not in moving) or 'F' in (symbols[a], symbols[b]) \
                        or (a in group_1 and b in group_1) or (a in group_2 and b in group_2):
                    continue
                if 'Ne' in (symbols[a], symbols[b]):
                    continue
                Rij = np.linalg.norm(difference)
                pair = (a, b)
                if (cu_1 in pair or cu_2 in pair) and n_atom in pair and Rij < REPULSION_MIN_DIST_CU:
                    if cu_1 in pair:
                        only_Cu_1 = True
                    else:
                        only_Cu_2 = True
                elif Rij >= REPULSION_MIN_DIST:
                    continue
                goodsound = False
                push = REPULSION_STEP / Rij * difference
                forces[a] -= push
                forces[b] += push

        if goodsound:
            break

        if step == REPULSION_MAX_STEPS - 1:
            print('WARNING: Did not ensure a distance of ' + str(REPULSION_MIN_DIST) + 'Å within '
                  + str(REPULSION_MAX_STEPS) + ' steps. N is ' + str(n_atom))
            finished = False

        for group, cu, only_Cu in ((atom_indices_1, cu_1, only_Cu_1), (atom_indices_2, cu_2, only_Cu_2)):
            if not group:
                continue
            indices = [g[0] for g in group]
            translation = forces[cu] if only_Cu else forces[indices].sum(axis=0)
            pos_i.positions[indices] += translation / len(group)
        pos_i.wrap()
    return pos_i, finished


def move_ion(pos_i, movement_i, atoms_indice, atom_type, verbose, sites):
    """
    Move an ion from one site (cage) to another as a starting guess for the minimization.

    For Cu complexes: If there already is one, the two complexes are put 2 Å above and

    After they are place the function applies the apply_repelling_force function
    
    :param pos_i: ASE structure. Changed in place
    :param movement_i: [start site, final site]. The site index is also the index of the site atom
    :param atoms_indice: [atoms of the start site, atoms of the final site]
    :param atom_type: 'Cu' or 'N'
    :param verbose: Print diagnostics (True on the worker leader)
    :param sites: All sites
    :return: pos_i
    """
    start, end = movement_i
    start_atoms, end_atoms = atoms_indice

    if not start_atoms[0]:
        name = 'output_structure/POSCAR-' + str(start) + str(end) + '.vasp'
        pos_i.write(name, parallel=False)
        abort('No ' + str(atom_type) + ' atoms in the starting cage! ' + str(movement_i) + '. Structure written to ' + name)

    if atom_type == 'Cu':
        atom_indices_1 = _ion_group(pos_i, start_atoms[0][0], CU_COMPLEX_SIZE)
        atom_indices_2 = _ion_group(pos_i, end_atoms[0][0], CU_COMPLEX_SIZE) if end_atoms[0] else []
        if end_atoms[1]:
            print('Two Cu ions in the final cage... Should not be possible  ', movement_i)

        if atom_indices_2:  # A Cu in the final cage. Move both of them
            _place(pos_i, atom_indices_1, CU_TEMPLATE, end, dz=2.0)
            _place(pos_i, atom_indices_2, CU_TEMPLATE, end, dz=-2.0)
        else:
            _place(pos_i, atom_indices_1, CU_TEMPLATE, end)

        # Neighbor list so it does not have to find distance between all atoms. Must be done after moving the Cu
        atoms_list = _neighbors(pos_i, end, 9)  # TODO: Pre store this
        pos_i, _ = apply_repelling_force(pos_i, atom_indices_1, atom_indices_2, atoms_list)

    elif atom_type == 'N':
        atom_indices_N = _ion_group(pos_i, start_atoms[0][0], NH4_SIZE)

        # Find [Cu(NH3)2]+ close to the final site: look back from their N and H atoms to their Cu
        neigh_Ne = _neighbors(pos_i, end, 2.5)
        atom_indices_check = set()
        atom_indices_1 = []
        for g in neigh_Ne:
            if pos_i[g].symbol in ('H', 'N') and g not in atom_indices_check:
                for back in range(1, CU_COMPLEX_SIZE + 1):
                    if pos_i[g - back].symbol == 'Cu':
                        group = _ion_group(pos_i, g - back, CU_COMPLEX_SIZE)
                        atom_indices_1 += group
                        atom_indices_check.update(i for i, _ in group)

        atom_indices_2 = []
        if len(atom_indices_1) == 2 * CU_COMPLEX_SIZE:
            atom_indices_2 = atom_indices_1[CU_COMPLEX_SIZE:]
            atom_indices_1 = atom_indices_1[:CU_COMPLEX_SIZE]
        elif atom_indices_1 and len(atom_indices_1) != CU_COMPLEX_SIZE:
            print('WARNING: atom_indices_1 in move_ion is not equal to zero, 9 or 18. Something is wrong')
            print(atom_indices_1)

        _place(pos_i, atom_indices_N, NH4_TEMPLATE, end)

        # Atoms of the two cages next to the NH4+ site. TODO Only add atoms from the specific cage if only one Cu is optimized
        nl = _neighbor_list(pos_i, 7.2)
        cage_1, cage_2 = end_atoms[1][0], end_atoms[1][1]
        atoms_list = np.concatenate((nl.get_neighbors(cage_1)[0], nl.get_neighbors(cage_2)[0]))

        groups = [g for g in (atom_indices_1, atom_indices_2) if g]
        if groups:
            pos_i, finished = apply_repelling_force(pos_i, atom_indices_1, atom_indices_2, atoms_list, atom_indices_N)
            # Check that the Cu are still in their cage
            cages = [_site_of_atom(sites, g[0][0]) for g in groups]
            distances = [pos_i.get_distance(c, g[0][0], mic=True) for c, g in zip(cages, groups)]
            incorrect_cage = any(d > 7 for d in distances)
            if incorrect_cage and verbose:
                print('A Cu moved to another cage during pre opt!!. Distance are ' + ' and '.join(map(str, distances)))

            if incorrect_cage or not finished:  # Something went wrong. Move the Cu near the center of its cage and try again
                if verbose:
                    print('Did not preopt correct for ' + str(len(groups)) + ' Cu. Moving it to center of its cage and trying again')
                    print('The F are ' + str(cage_1) + ' and ' + str(cage_2))
                    print(atoms_list)
                    print(sites[cage_1].atoms)
                    print(sites[cage_2].atoms)
                if len(groups) == 2:
                    _place(pos_i, atom_indices_1, CU_TEMPLATE, cages[0], dz=1.5)
                    _place(pos_i, atom_indices_2, CU_TEMPLATE, cages[1], dz=-1.5)
                else:
                    _place(pos_i, atom_indices_1, CU_TEMPLATE, cages[0], dz=1.0)
                pos_i, finished = apply_repelling_force(pos_i, atom_indices_1, atom_indices_2, atoms_list, atom_indices_N)
                if not finished and verbose:
                    print('WARNING: ' + str(len(groups)) + ' Cu ion(s) could not find a proper place with '
                          + str(REPULSION_MIN_DIST) + ' Å inbetween')
    else:
        abort('Atom type ' + str(atom_type) + ' not implemented')

    pos_i.wrap()
    return pos_i


def find_digit_difference(num1, num2):
    """
    Function that takes two coverages and identify which numbers are changed to be used in moving a atom for ML-FF.
    The number that changed is the site which is involved in the kMC step

    param: num1: initial coverage
    param: num2: final coverage

    return: list with initial site, final site and the new coverage of the final site

    """
    str_num1 = str(num1)
    str_num2 = str(num2)
    max_len = max(len(str_num1), len(str_num2))
    str_num1 = str_num1.zfill(max_len)
    str_num2 = str_num2.zfill(max_len)

    subtracted = None
    added = None
    no_Cu = None
    for i, (digit1, digit2) in enumerate(zip(str_num1, str_num2)):
        if int(digit1) - int(digit2) == 1:
            subtracted = i
        elif int(digit2) - int(digit1) == 1:
            added = i
            no_Cu = int(digit2)

    if subtracted is None or added is None:
        abort('Could not find the move between coverage ' + str(num1) + ' and ' + str(num2))
    return [subtracted, added, no_Cu]


# ----------------------------------------------------------------------
# Site assignment
# These functions helps processing the optimised structure and check if anything has gone wrong
# ----------------------------------------------------------------------
def _site_atom_counts(pos_i):
    """
    Counts the number of sites
    """
    return sum(1 for a in pos_i if a.symbol == 'F'), sum(1 for a in pos_i if a.symbol == 'Ne')


def _find_NH4(pos_i, caller):
    """
    Finds the indices of the N atoms of the NH4+ ions:
    This is done by looking at the neighbors. If 4 C atoms are found within 4 Å then it is a NH4+ ion
    The rest of the N atoms belong to the [Cu(NH3)2]+ complex.
    """
    symbols = pos_i.get_chemical_symbols()
    N_indices = [i for i, s in enumerate(symbols) if s == 'N']
    no_NH4 = len(N_indices) - 2 * symbols.count('Cu')
    nl = _neighbor_list(pos_i, 1.10)

    NH4_indices = []
    for k in N_indices:
        neigh = nl.get_neighbors(k)[0]
        neigh_symbols = {symbols[g] for g in neigh}
        if (len(neigh) == 4 and neigh_symbols <= {'C'}) or (len(neigh) == 5 and neigh_symbols <= {'C', 'Ne', 'F'}):
            NH4_indices.append(k)

    if len(NH4_indices) != no_NH4:
        abort('Did not find all NH4 atoms in ' + caller + '!! Found ' + str(len(NH4_indices)) + ' of ' + str(no_NH4) + '. See ML_FF')
    return NH4_indices


def _closest_site_atom(pos_i, indices, site_symbol, cutoff=8):
    """
    For each atom in indices, find the closest site atom (F or Ne) within cutoff. This is to assign the different
    atoms to their site

    :return: list of [site atom, atom, distance]
    """
    symbols = pos_i.get_chemical_symbols()
    nl = _neighbor_list(pos_i, cutoff)
    closest = []
    for k in indices:
        candidates = [g for g in nl.get_neighbors(k)[0] if symbols[g] == site_symbol]
        if not candidates:
            abort('Atom ' + str(k) + ' (' + symbols[k] + ') could not be assigned to a ' + site_symbol + ' site. See ML_FF')
        distances = pos_i.get_distances(k, candidates, mic=True)
        best = int(np.argmin(distances))
        closest.append([candidates[best], k, distances[best]])
    return closest


def _occupation(pos_i, caller):
    """Check which site the complexes belong to"""
    NH4_atom = _closest_site_atom(pos_i, _find_NH4(pos_i, caller), 'Ne')
    Cu_indices = [i for i, a in enumerate(pos_i) if a.symbol == 'Cu']
    Cu_atom = _closest_site_atom(pos_i, Cu_indices, 'F')
    return NH4_atom, Cu_atom


def _coverage_string(sites):
    return ''.join(str(s.covered) for s in sites)


def recompute_neighbors(pos_i, site_i, me):
    """
    Reassign the NH4+ to the closest NH4+ site and move NH4+ that are more than 2 Å from their site onto it.
    Only used the starting structure does not lead to a stable system.

    :return: pos_i and site_i, both changed in place
    """
    _, Ne_length = _site_atom_counts(pos_i)
    NH4_atom = _closest_site_atom(pos_i, _find_NH4(pos_i, 'recompute_neighbors'), 'Ne')

    for g in site_i[-Ne_length:]:  # Reset the NH4+ sites
        g.atoms[0] = []
        g.covered = 0

    for site, N, distance in NH4_atom:
        site_i[site].atoms[0] = [N]
        site_i[site].covered = 1
        if distance > 2:  # Move the NH4+ ion onto its site
            _place(pos_i, _ion_group(pos_i, N, NH4_SIZE), NH4_TEMPLATE, site)

    check_NH4_sites(pos_i, site_i, me)
    return pos_i, site_i


def recompute_neighbors_all(pos_i, site_i):
    """
    Compute the position of the complexes

    :return: coverage of site_i and coverage of pos_i as strings. site_i is not changed
    """
    NH4_atom, Cu_atom = _occupation(pos_i, 'recompute_neighbors_all')
    covered = [0] * len(site_i)
    for site, _, _ in NH4_atom:
        covered[site] = 1
    for site, _, _ in Cu_atom:
        covered[site] += 1
    return _coverage_string(site_i), ''.join(map(str, covered))


def recompute_neighbors_all_site(pos_i, site_i, cov_new):
    """
    Update the coverage and atoms of all sites from the structure. 

    :return: site_i, changed in place
    """
    NH4_atom, Cu_atom = _occupation(pos_i, 'recompute_neighbors_all_site')

    for g in site_i:
        g.atoms[0] = []
        if g.atom_type == 'Cu':  # For NH4+ sites atoms[1] holds the two blocked Cu sites and is kept
            g.atoms[1] = []
        g.covered = 0

    for site, N, _ in NH4_atom:
        site_i[site].atoms[0] = [N]
        site_i[site].covered = 1

    for site, Cu, _ in Cu_atom:
        site_i[site].covered += 1
        if site_i[site].covered == 1:
            site_i[site].atoms[0] = [Cu]
        elif site_i[site].covered == 2:
            site_i[site].atoms[1] = [Cu]
        else:
            print('Something went wrong in recomputation of sites. More than 2 Cu on site ' + str(site))

    if _coverage_string(site_i) != cov_new:
        print('cov mismatch!!! See ML_FF compute neighbor NH4 site')

    return site_i


# ----------------------------------------------------------------------
# Checks of the optimized structure
# ----------------------------------------------------------------------
def _wrong_site_assignment(pos_i, sites, site, my_proc):
    print('The coverage is :' + _coverage_string(sites))
    print('The covered and indice does not match!! Site is ' + str(site) + ' and indice is ' + str(sites[site].atoms[0][0]))
    debug_structures(pos_i, [site, sites[site].atoms[0][0]], my_proc, 'WRONG_covered_indice', 'S')
    abort('Site and atom assignment does not match. See ML_FF')


def check_structure(pos_i, pos_min, pos_main, my_proc, move, site_data):
    """
    Checks the neighbors of all N atoms: an N with 4 neighbors must be NH4+ (all C) and an N with 3 neighbors
    must be NH3 in [Cu(NH3)2]+ (all H). Writes debug structures and aborts if not.

    :param pos_i: Structure to check
    :param pos_min: Minimized structure (only written for debugging)
    :param pos_main: Structure before the move (only written for debugging)
    :param move: The move from find_digit_difference, or None
    """
    pos_i = pos_i[[a.symbol not in SITE_SYMBOLS for a in pos_i]]
    pos_min = pos_min[[a.symbol not in SITE_SYMBOLS for a in pos_min]]
    symbols = pos_i.get_chemical_symbols()
    moved = 'when moving from ' + str(move[0]) + ' to ' + str(move[1]) if move else 'for the current coverage'
    nl = _neighbor_list(pos_i, 1.10)

    for k in [i for i, s in enumerate(symbols) if s == 'N']:
        neigh = nl.get_neighbors(k)[0]
        neigh_symbols = {symbols[g] for g in neigh}
        if len(neigh) == 4 and neigh_symbols <= {'C'}:
            continue
        if len(neigh) == 3 and neigh_symbols <= {'H'}:
            continue
        if len(neigh) in (3, 4):
            print('Not correct ' + str(len(neigh)) + ' neighbors of wrong type ' + moved + ': ' + str(neigh))
        else:
            print('Not correct, to few or many neighbors. Happen ' + moved)
        debug_structures(pos_i, [k], my_proc, 'WRONGSTRUCTURE_FINAL', 'Ne')
        debug_structures(pos_min, [k], my_proc, 'WRONGSTRUCTURE_MIN', 'Ne')
        debug_structures(pos_main, [], my_proc, 'WRONGSTRUCTURE_MAIN', 'Ne')
        abort('Wrong N neighbors in check_structure. Debug structures written to output_structure/')


def check_NH4_sites(pos_i, sites_N, my_proc):
    """
    Checks if the NH4+ ions are on the correct site in the ML-FF structure compared to where the kMC has it.

    :return: True if any NH4+ is more than 2.5 Å from its site, and the number of such NH4+
    """
    No_error = 0
    _, Ne_length = _site_atom_counts(pos_i)
    first = len(sites_N) - Ne_length
    for site, g in enumerate(sites_N[first:], start=first):
        if not g.atoms[0]:
            continue
        if g.covered != 1:
            _wrong_site_assignment(pos_i, sites_N, site, my_proc)
        if pos_i.get_distance(g.atoms[0][0], site, mic=True) > 2.5:
            No_error += 1
    return No_error > 0, No_error


def sites_after_move(sites, move):
    """
    Copy of the sites as the kMC will have them after the move. This is used for the kMC code and not ML-FF!

    :param move: The move from find_digit_difference, or None for no move
    :return: sites if there is no move, otherwise a changed copy
    """
    if move is None:
        return sites
    start, end = move[0], move[1]
    new_sites = copy.deepcopy(sites)
    A, B = new_sites[start], new_sites[end]
    A.covered -= 1
    B.covered += 1
    if A.atom_type == 'Cu':
        Cu = A.atoms[0].pop(0)
        if not B.atoms[0]:
            B.atoms[0].append(Cu)
        else:
            B.atoms[1].append(Cu)
        if A.atoms[1]:  # The second Cu of the start site becomes the first
            A.atoms[0].append(A.atoms[1].pop(0))
    else:  # NH4+. atoms[1] holds the blocked Cu sites and is kept
        B.atoms[0].append(A.atoms[0].pop(0))
    return new_sites


def check_Cu_sites(pos_i, sites_N, my_proc):
    """
    Checks if the Cu ions are on the correct site in the ML-FF structure compared to where the kMC has it.

    :return: True if any Cu is more than 8 Å from its site, and the number of such Cu
    """
    No_error = 0
    for site, g in enumerate(sites_N):
        if g.atom_type != 'Cu' or not g.atoms[0]:
            continue
        if g.covered == 0:
            _wrong_site_assignment(pos_i, sites_N, site, my_proc)
        for Cu in g.atoms[0] + g.atoms[1]:  # atoms[1] holds the second Cu of a site with 2
            if pos_i.get_distance(Cu, site, mic=True) > 8:
                No_error += 1
    return No_error > 0, No_error


def debug_structures(pos_i, list_Ne, me, name, atom_type):
    """Write pos_i to output_structure/ with the atoms in list_Ne changed to atom_type, so they are easy to spot"""
    pos_a = copy.deepcopy(pos_i)
    for g in list_Ne:
        pos_a[g].symbol = atom_type
    os.makedirs('output_structure', exist_ok=True)
    pos_a.write('output_structure/POSCAR-' + str(me) + '-' + str(name) + '.vasp', parallel=False)

"""Defines the Site Class derived from base.site.SiteBase.

The site class is defined here as an interface to the base
class in base.site.SiteBase, where the user can add custom tags.
Custom tags can be used to evaluate the rate-constants in user_events.py

See Also
---------
Module: user_events

"""

from base.sites import SiteBase
from ase.neighborlist import NeighborList
import numpy as np
from ase import Atoms


class Site(SiteBase):
    """A kMC site.

    The sites are dummy atoms in the structure: F atoms is the Cu sites (cage centers) and
    Ne atoms mark the NH4+ sites (the eight-membered rings).
    

    Attributes
    -------------
    covered: int
        Number of ions on the site. Cu sites: 0, 1 or 2 [Cu(NH3)2]+. NH4+ sites: 0 or 1 NH4+.

    ind: int
        Index of the F or Ne atom of the site.

    atom_type: str
        'Cu' for a Cu site and 'N' for a NH4+ site.

    atoms: list
        Cu sites: [[first Cu], [second Cu]]. NH4+ sites: [[N], [F1, F2]], where F1 and F2 are the
        two Cu sites the NH4+ blocks.

    See Also
    -----------
    Module: base.sites

    """

    def __init__(self, stype=0, covered=0, ind=[], lattice_pos=None,atoms=None,atom_type=None):
        SiteBase.__init__(self, stype=stype, covered=covered, ind=ind, atoms=atoms,
                          lattice_pos=lattice_pos,atom_type=atom_type)

def assign_atoms(pos):
    """
    This function assign atoms to kMC by figuring out which site they belong to.
    
    
    :param pos: ASE structure
    :return: 
    """

    def Find_nearest_F(structure):

        # Find the nearest F atoms for Cu species. Cut off is 8 Å. the cutoff just have to be large enough to find the Cu ions
        # The code will always look for the F that is closest even though the cut off is set higher
        cutoff_radius = 8  # Å
        nl_f = NeighborList(cutoffs=[cutoff_radius / 2] * len(structure), self_interaction=False, bothways=True)
        nl_f.update(structure)
        Cu_indices = [i for i, atom in enumerate(structure) if atom.symbol == 'Cu']
        F_indices = [i for i, atom in enumerate(structure) if atom.symbol == 'F']
        Cu_F = []
        Cu_F_d = []
        Cu_atoms = []
        for k in range(len(Cu_indices)):
            F_atoms_found = []
            F_atoms_found_index = []
            Cu_neighbor = nl_f.get_neighbors(Cu_indices[k])
            for f in Cu_neighbor[0]:
                if structure.symbols[f] == 'F':
                    distance_i = structure.get_distances(a=Cu_indices[k], indices=f, mic=True)
                    F_atoms_found.append(distance_i[0])
                    F_atoms_found_index.append(f)
            if not F_atoms_found:
                print('Error - A Cu atom could not be assigned to a F atom')
                exit(1)
            Cu_F.append([F_atoms_found_index[F_atoms_found.index(min(F_atoms_found))]][0])
            Cu_F_d.append(min(F_atoms_found))
            Cu_atoms.append([[F_atoms_found_index[F_atoms_found.index(min(F_atoms_found))]][0],Cu_indices[k]])
        return Cu_F, Cu_F_d, Cu_atoms
    
    # Report Cu/Al and Si/Al
    Cu_indices = [i for i, atom in enumerate(pos) if atom.symbol == 'Cu']
    Al_indices = [i for i, atom in enumerate(pos) if atom.symbol == 'Al']
    Si_indices = [i for i, atom in enumerate(pos) if atom.symbol == 'Si']
    

    CuAl = len(Cu_indices) / len(Al_indices)
    SiAl = len(Si_indices) / len(Al_indices)
    



    # Find all F
    F_indices = [i for i, atom in enumerate(pos) if atom.symbol == 'F']


    Cu_F_nl, Cu_F_d,Cu_atoms = Find_nearest_F(pos)

    max_val = len(F_indices)

    # Initialize result list with zeros
    result = [0] * (max_val)

    # Track occurrences
    for idx in Cu_F_nl:
        index = int(idx)
        if result[index] == 0:
            result[index] = 1
        elif result[index] == 1:
            result[index] = 2


    
    return result,pos,Cu_atoms,CuAl,SiAl


def adjust_positions(pos_i):



    F_indices = [i for i, atom in enumerate(pos_i) if atom.symbol == 'F']
    # Find the atoms around each cage (F site)
    cutoff = 7.2  # bond cutoff distance in ångstroms
    nl = NeighborList(cutoffs=[cutoff / 2] * len(pos_i), self_interaction=False, bothways=True)
    nl.update(pos_i)
    for g in F_indices:
        neigh_F = nl.get_neighbors(g)
        atoms_list = neigh_F[0]
        atom_indices_1 = []
        atom_indices_2 = []
        for f in atoms_list:
            if pos_i[f].symbol == 'Cu':
                if not atom_indices_1:
                    atom_indices_1.append([f,'Cu'])
                    for i in range(8):
                        atom_indices_1.append([f + i + 1, pos_i[f + i + 1].symbol])
                elif not atom_indices_2:
                    atom_indices_2.append([f,'Cu'])
                    for i in range(8):
                        atom_indices_2.append([f + i + 1, pos_i[f + i + 1].symbol])
                else:
                    print('More than two Cu in a cage number ' + str(g))
                    exit(1)
        if atom_indices_1:
            pos_i = apply_repelling_force(pos_i, atom_indices_1, atom_indices_2, atoms_list)

    return pos_i



def NH4_site(pos_i):
    def combinations(lst, r):
        # Base case: if r is 0, return an empty list
        if r == 0:
            return [[]]

        # Base case: if the list is empty, return an empty list
        if len(lst) == 0:
            return []

        # Recursive case: take the first element and generate combinations with and without it
        first = lst[0]
        rest = lst[1:]

        # Generate combinations that include the first element
        with_first = [[first] + combo for combo in combinations(rest, r - 1)]

        # Generate combinations that do not include the first element
        without_first = combinations(rest, r)

        # Return the combination of both lists
        return with_first + without_first

    F_indices = [i for i, atom in enumerate(pos_i) if atom.symbol == 'F']

    combinations_F = combinations(F_indices, 2)
    length_structure = len(pos_i)


    # Find the coverage of the NH4+. Do this before putting in Ne!
    atom_indices = []
    atom_indices_i = [i for i, atom in enumerate(pos_i) if atom.symbol == 'N']
    cutoff_N = 1.1  # bond cutoff distance in ångstroms
    nl = NeighborList(cutoffs=[cutoff_N / 2] * len(pos_i), self_interaction=False, bothways=True)
    nl.update(pos_i)
    for k in atom_indices_i:
        neigh = nl.get_neighbors(k)
        n_neigh = neigh[0]
        n_neigh = n_neigh[n_neigh > 80]
        if len(n_neigh) == 4:
            for g in n_neigh:
                if pos_i[g].symbol != 'C':
                    print('A N has 4 neighbors but they are not all C. see user_sites')
                    pos_i[g].symbol = 'S'
                    pos_i.write('output_structure/POSCAR_Neighbor.vasp',parallel=False)
                    exit(1)
            atom_indices.append(k)
        elif len(n_neigh) == 3: # Just to make sure that the correct N are taken
            for g in n_neigh:
                if pos_i[g].symbol != 'H':
                    print('A N has 3 neighbors but they are not all H. See user_site')
                    exit(1)
        else:
            print('A N does not have 3 or 4 neighbors!! see user_site')
            exit(1)


    length_F = len(F_indices)
    Ne_length = length_F * 3
    # Get the current positions and atomic numbers
    positions = pos_i.get_positions()
    numbers = pos_i.get_atomic_numbers()

    # Insert the Neon atom at the specified index


    # Create a new Atoms object with the updated positions and atomic numbers
    counter_Ne = Ne_length -1
    F_neighbors = []
    F_Ne_block = {} # Used to which Ne site is between two cages
    for g in combinations_F:
        # Check distance
        distance = pos_i.get_distance(g[0], g[1], mic=True)

        if distance < 11:  # When they are neighbors!
            distance = pos_i.get_distance(g[0], g[1], mic=True, vector=True)
            distance /= 2
            neon_position = pos_i[g[0]].position + distance

            # Insert the Neon atom at the specified index
            positions = np.insert(positions, length_F, neon_position, axis=0)
            numbers = np.insert(numbers, length_F, 10)  # Atomic number of Neon is 10
            F_neighbors.append([counter_Ne+length_F,g[0],g[1]])
            F_Ne_block[min(g),max(g)] = counter_Ne+length_F
            counter_Ne -= 1

            length_structure += 1
    pos_i = Atoms(numbers=numbers, positions=positions, cell=pos_i.get_cell(), pbc=pos_i.get_pbc())
    Ne = len([i for i, atom in enumerate(pos_i) if atom.symbol == 'Ne'])

    correct_sum = Ne / len([i for i, atom in enumerate(pos_i) if atom.symbol == 'F'])
    if correct_sum != 3:
        print('The number of sites for NH4+ (or for Cu) is not correct!! See user_site')

    for g in range(len(atom_indices)):
        atom_indices[g] += Ne
    # Find the closest Ne atom to each NH4+
    cutoff_radius = 8  # Å
    nl_f = NeighborList(cutoffs=[cutoff_radius / 2] * len(pos_i), self_interaction=False, bothways=True)
    nl_f.update(pos_i)
    Ne_indices = [i for i, atom in enumerate(pos_i) if atom.symbol == 'Ne']
    NH4_F = []
    NH4_F_d = []
    NH4_atoms = []
    for k in range(len(atom_indices)): # This is for Ne atoms even though the variable says F
        F_atoms_found = []
        F_atoms_found_index = []
        Cu_neighbor = nl_f.get_neighbors(atom_indices[k])
        for f in Cu_neighbor[0]:
            if pos_i.symbols[f] == 'Ne':
                distance_i = pos_i.get_distances(a=atom_indices[k], indices=f, mic=True)
                F_atoms_found.append(distance_i[0])
                F_atoms_found_index.append(f)
        if not F_atoms_found:
            print('Error - A NH4+ ion could not be assigned to a Ne atom. see user_sites')
        NH4_F.append([F_atoms_found_index[F_atoms_found.index(min(F_atoms_found))]][0])
        NH4_F_d.append(min(F_atoms_found))
        NH4_atoms.append([[F_atoms_found_index[F_atoms_found.index(min(F_atoms_found))]][0], atom_indices[k]])

    pos_i.wrap()


    # Get coverage
    max_val = Ne

    # Initialize result list with zeros
    result = [0] * (max_val)

    # Track occurrences
    for idx in NH4_atoms:
        index = int(idx[0]) - length_F
        if result[index] == 0:
            result[index] = 1
        elif result[index] == 1:
            print('Two NH4 in the same site!! See user_sites')
            exit(1)
    return pos_i, NH4_atoms,result,F_neighbors,F_Ne_block

def apply_repelling_force(pos_i, atom_indices_1, atom_indices_2,atoms_list,N_atom_index = [[362222253]]):
    min_distance = 1.5
    min_distance_Cu = 2.3
    stepmax = 50
    only_Cu_1 = False
    only_Cu_2 = False
    total_atoms = len(pos_i)
    frozen = [True] * total_atoms

    empty_1 = False
    empty_2 = False




    for idx in atom_indices_1 + atom_indices_2:
        frozen[idx[0]] = False


    for step in range(stepmax):

        forces = np.zeros((total_atoms, 3))
        goodsound = True
        if not atom_indices_1:
            atom_indices_1 = [[11111111111111111111111111111]]
            empty_1 = True
        if not atom_indices_2:
            atom_indices_2 = [[11111111111111111111111111111]]
            empty_2 = True
        for i in range(len(atoms_list) - 1):
            for j in range(i + 1, len(atoms_list)):

                if frozen[atoms_list[i]] and frozen[atoms_list[j]] or pos_i[atoms_list[i]].symbol == 'F' or pos_i[atoms_list[j]].symbol == 'F' or all(any(sublist[0] == number for sublist in atom_indices_1) for number in [atoms_list[i], atoms_list[j]]) or all(any(sublist[0] == number for sublist in atom_indices_2) for number in [atoms_list[i], atoms_list[j]]):
                    continue
                if pos_i[atoms_list[i]].symbol == 'Ne' or pos_i[atoms_list[j]].symbol == 'Ne':
                    continue

                difference = pos_i.get_distance(atoms_list[i], atoms_list[j], mic=True,vector=True)
                Rij = np.linalg.norm(difference)

                if (atoms_list[j] == atom_indices_1[0][0] or atoms_list[i] == atom_indices_1[0][0] or atoms_list[j] == atom_indices_2[0][0] or atoms_list[i] == atom_indices_2[0][0]) and (atoms_list[j] == N_atom_index[0][0] or atoms_list[i] == N_atom_index[0][0] or atoms_list[j] == N_atom_index[0][0] or atoms_list[i] == N_atom_index[0][0]) and Rij < min_distance_Cu: # This is to ensure that it move the atom the correct way

                    if atoms_list[j] == atom_indices_1[0][0] or atoms_list[i] == atom_indices_1[0][0]:
                        only_Cu_1 = True
                    elif atoms_list[j] == atom_indices_2[0][0] or atoms_list[i] == atom_indices_2[0][0]:
                        only_Cu_2 = True
                    else:
                        print('Something went wrong in apply_repeling_force function')
                        exit(1)

                    goodsound = False
                    fij, phiij = linear_repulsion_pot(difference, min_distance_Cu)
                    fij /= Rij
                    phiij = fij * difference
                    forces[atoms_list[i]] -= phiij
                    forces[atoms_list[j]] += phiij
                elif Rij < min_distance:
                    goodsound = False
                    fij, phiij = linear_repulsion_pot(difference, min_distance)
                    fij /= Rij
                    phiij = fij * difference
                    forces[atoms_list[i]] -= phiij
                    forces[atoms_list[j]] += phiij
        if goodsound:
            break

        if step == stepmax - 1:
            print('WARNING: Did not ensure a distance of ' + str(min_distance) + 'Å within ' + str(stepmax) + ' steps')

        if empty_1:
            atom_indices_1 = []

        if empty_2:
            atom_indices_2 = []

        translation_vector_1 = np.zeros(3)
        translation_vector_2 = np.zeros(3)

        if only_Cu_1:
            translation_vector_1 += forces[atom_indices_1[0][0]]
        else:
            for g in atom_indices_1:
                translation_vector_1 += forces[g[0]]

        if only_Cu_2:
            translation_vector_2 += forces[atom_indices_2[0][0]]
        else:
            for g in atom_indices_2:
                translation_vector_2 += forces[g[0]]
        only_Cu_1 = False
        only_Cu_2 = False

        if atom_indices_1:
            translation_vector_1 /= len(atom_indices_1)

        if atom_indices_2:
            translation_vector_2 /= len(atom_indices_2)

        for g in atom_indices_1:
            pos_i.positions[g[0]] += translation_vector_1

        for g in atom_indices_2:
            pos_i.positions[g[0]] += translation_vector_2
        pos_i.wrap()
    return pos_i

def linear_repulsion_pot(r, cutoff):
    slope = 1.0 # How big steps are taken
    intercept = slope * cutoff
    energy = slope * r - intercept
    fij = slope
    return fij, energy



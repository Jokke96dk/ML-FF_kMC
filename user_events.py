"""Contains all user-defined event types.

All user-defined events are defined here, which
must be derived from the parent class EventBase.  

See also
---------
Module: base.events for documentation about the methods possible(), get_rate(), and do_event().

"""

import numpy as np
from base.events import EventBase

KB = 8.617333262e-5     # Boltzmann constant (eV/K)
PREFACTOR = 1e8         # Prefactor of the rate constant (1/s)
BARRIER_OFFSET = 0.4    # Added to the energy difference to get the barrier (eV)


class CuDiffEvent(EventBase):
    """A diffusion event class.

    The event is Cu-Cage_a + Cage_b -> Cage_a + Cu-Cage_b .

    This event is possible if:
    1) Starting cage has a minimum of 1 Cu complex - system.sites[site].covered > 0
    2) Next cage has a maximum of 1 Cu complexes - system.sites[other_site].covered < 2
    3) No NH4+ ion blocks the eight-membered ring between the two cages (params['N_block'])

    3 Cu complexes in the same cage are thus not allowed

    The same event also moves a NH4+ ion from an occupied NH4+ site to a neighboring empty NH4+ site.

    Performing this event will lead to a loss of one Cu complex in starting cage and one more in the final cage.

    """

    def __init__(self, params):
        name = 'CuDiffEvent'
        EventBase.__init__(self, params, name)

    def possible(self, system, site, other_site):

        if system.sites[site].covered > 0 and system.sites[other_site].covered < 2 and system.sites[site].atom_type == 'Cu' and system.sites[other_site].atom_type == 'Cu': # For Cu
            block_N = self.params['N_block'] 
            Site_N = block_N[min(site,other_site),max(site,other_site)]
            if system.sites[Site_N].covered == 0: # Check if a NH4+ ion is blocking the eight-membered ring connecting the two cages
                return True
            else:
                return False
        elif system.sites[site].covered == 1 and system.sites[other_site].covered == 0 and system.sites[site].atom_type == 'N' and system.sites[other_site].atom_type == 'N': # For NH4:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site,mlff_info):

        do_MLFF = False  # Should we do a ML-FF optimisation
        R = []
        # Convert to coverage
        cov_start = ''
        cov_end = ''
        counter = 0

        for g in system.sites:
            if counter == site:
                cov_end += str(g.covered - 1)
            elif counter == other_site:
                cov_end += str(g.covered + 1)
            else:
                cov_end += str(g.covered)
            cov_start += str(g.covered)
            counter += 1

        if mlff_info == False:
            # Check if the calculation are present in the database

            key_start = cov_start
            key_end = cov_end
            if key_start not in system.calc:
                do_MLFF = True # Not present. Do a ML-FF optimisation
                R.append(key_start)
            if key_end not in system.calc:
                do_MLFF = True
                R.append(key_end)

        if do_MLFF == False:
            # Extract energy
            try:
                energy = system.calc[cov_end][0] - system.calc[cov_start][0]
            except:
                print('Could not find the energies. See User_event')
                print('Sites are ' + str(cov_end) + ' and ' + str(cov_start))
                exit(1)
            if system.calc[cov_end][0] > 100: # Higher energy barrier than 100. These are events not supposed to happen
                R = 5e-324
            elif energy > 0: # Set barrier as highest energy plus BARRIER_OFFSET
                Ea = np.exp(-(energy + BARRIER_OFFSET) / (KB * self.params['T']))
                R = PREFACTOR * Ea
            else: # Set barrier to BARRIER_OFFSET
                Ea = np.exp(-BARRIER_OFFSET / (KB * self.params['T']))
                R = PREFACTOR * Ea

        return R,do_MLFF

    def do_event(self, system, site, other_site):
        system.sites[site].covered -= 1
        system.sites[other_site].covered += 1
        print('Site ' + str(site) + ' now has a coverage of ' + str(system.sites[site].covered))
        print('Site ' + str(other_site) + ' now has a coverage of ' + str(system.sites[other_site].covered))



    def get_involve_other(self):
        return True

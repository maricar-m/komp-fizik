from copy import deepcopy
from math import pi, cos, sin, sqrt
import numpy as np
import operator
import random

nearest_neighbor_delta = [(-1, -1, 0), (-1, 0, -1), (-1, 0, 1), (-1, 1, 0),
                          (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                          (1, -1, 0), (1, 0, -1), (1, 0, 1), (1, 1, 0)]

next_nearest_neighbor_delta = [(-2, 0, 0), (2, 0, 0),
                               (0, -2, 0), (0, 2, 0),
                               (0, 0, -2), (0, 0, 2)]

class Site:
    def __init__(self, index, coords, theta=pi, phi=0):
        self.index = index
        self.coords = coords
        self.theta = theta  # spin
        self.phi = phi  # spin
        self.nearest = []  # neighbors
        self.next_nearest = []  # neighbors


    def neighbors(self):
        n = list(self.nearest)
        n.extend(self.next_nearest)
        return n

    def with_attr(self, attr, value):
        s = deepcopy(self)
        setattr(s, attr, value)
        return s

    def __repr__(self):
        return "Site({}, {}): {} | {}".format(
            self.index, self.coords,
            map(lambda n: n.index, self.nearest),
            map(lambda n: n.index, self.next_nearest))



class Simulator:
    def __init__(self, size):
        self.size = size
        self.num_sites = self.get_num_sites(size)
        self.sites = []
        self.coord_to_site = {}
        self.magnetizations = []
        self.records = {}

        # Parameters
        self.J_NN = 1
        self.J_NNN = 1
        self.H_EXTERNAL = 1
        self.MAX_DELTA_THETA = 1
        self.MAX_DELTA_PHI = 1
        self.NUM_STEPS = 50
        self.SIM_THRESHOLD = 0
        self.SIM_SAMPLE_RES = 5
        self.TEMP_LOWER = 0
        self.TEMP_UPPER = 5
        self.TEMP_STEP_SIZE = 1

        # Derived
        self.num_temp_steps = ((self.TEMP_UPPER - self.TEMP_LOWER) /
                               self.TEMP_STEP_SIZE) + 1
        # Setup
        self.populate_sites()

    def reset(self):
        """
        Resets this simulator to a ready state.
        """
        self.sites = []
        self.coord_to_site = {}
        self.populate_sites()


    def run(self):
        """
        Runs the simulation using the initialization parameters.
        """
        self.reset()

        for temp in np.linspace(self.TEMP_LOWER, self.TEMP_UPPER, self.num_temp_steps):
            print 'evaluating temp: {}'.format(temp)
            for step in range(self.NUM_STEPS): # num steps of simulation
                beta = self.J_NN / temp

                # Evaluate each site
                for i in range(self.num_sites):
                    self.evaluate_and_update_site_spin(self.sites[i], beta)

                # Sample from every nth step after the threshold.
                if step > self.SIM_THRESHOLD and step % self.SIM_SAMPLE_RES == 0:
                    self.magnetizations.append(self.get_magnetization())

            # Record state for each step of temperature
            self.records[temp] = self.get_state()

    def get_magnetization(self):
        """
        Calculate the magnetization value for the system and
        save it to self.magnetizations.
        """
        M_CONST = 7.0 / 2
        x = reduce(lambda a, site: a + sin(site.theta) + cos(site.phi),
                   self.sites, 0)
        y = reduce(lambda a, site: a + sin(site.theta) + sin(site.phi),
                   self.sites, 0)
        z = reduce(lambda a, site: a + cos(site.theta), self.sites, 0)

        x *= M_CONST
        y *= M_CONST
        z *= M_CONST

        return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))


    def get_state(self):
        """
        Calculate moment and cumulant with errors
        """
        mags = self.magnetizations
        num_mags = len(self.magnetizations)

        # averages
        m_avg = reduce(operator.add, mags, 0) / num_mags
        m_avg_2 = reduce(lambda a, b: a + pow(b, 2), mags, 0) / num_mags
        m_avg_4 = reduce(lambda a, b: a + pow(b, 4), mags, 0) / num_mags

        # average errors
        m_error = reduce(lambda a, m: a + pow(m - m_avg, 2), mags, 0)
        m_error_2 = reduce(lambda a, m: a + pow(pow(m, 2) - m_avg, 2), mags, 0)
        m_error_4 = reduce(lambda a, m: a + pow(pow(m, 4) - m_avg, 2), mags, 0)

        m_error = sqrt(m_error / (num_mags * (num_mags - 1)))
        m_error_2 = sqrt(m_error_2 / (num_mags * (num_mags - 1)))
        m_error_4 = sqrt(m_error_4 / (num_mags * (num_mags - 1)))

        # cumulant
        u = 1.0 - m_avg_4 / 3.0 / pow(m_avg_2, 2)

        # cumulant error
        u_error = (1.0 - u) * \
                  sqrt(pow(m_error_4, 2) / pow(m_avg_4, 2) +
                       (4 * pow(m_error_2, 2) / pow(m_avg_2, 2)))

        return dict(u=u, u_error=u_error, mag_avg=m_avg, m_error=m_error)



    def evaluate_and_update_site_spin(self, site, beta):
        """
        Changes the given site's spin slightly and calculates the
        change in hamiltonian. If the change falls within the threshold,
        the site's new spin is saved.
        """
        new_site = deepcopy(site)

        # Create a new site with only its theta changed
        delta_theta = self.MAX_DELTA_THETA * random.uniform(-1, 1)
        site_theta = site.with_attr('theta', site.theta + delta_theta)

        if random.random() < -beta * self.delta_hamiltonian(site, site_theta):
            new_site.theta = site_theta.theta

        # Create a new site with only its phi changed
        delta_phi = self.MAX_DELTA_PHI * random.uniform(-1, 1)
        site_phi = site.with_attr('phi', delta_phi)

        if random.random() < -beta * self.delta_hamiltonian(site, site_phi):
            new_site.phi = site_theta.phi

        self.sites[site.index] = new_site

    def delta_hamiltonian(self, old_site, new_site):
        """
        Calculate the difference in Hamiltonian values
        for the two systems.
        """
        H_CONST = 63.0/4
        old_hamiltonian = 0.0
        new_hamiltonian = 0.0

        def hamiltonian(site, b):
            def helper(accum, neighbor):
                return accum - H_CONST * b * \
                    (sin(site.theta) *
                     sin(neighbor.theta) *
                     cos(site.phi - neighbor.phi) +
                     cos(site.theta) *
                     cos(neighbor.theta))

            return helper

        old_hamiltonian = reduce(hamiltonian(old_site, 1), # reduction function
                                 site.nearest,  # target neighbors
                                 0)  # starting value
        old_hamiltonian = reduce(hamiltonian(old_site, self.J_NNN), # reduction function
                                 site.next_nearest,  # target neighbors
                                 old_hamiltonian)  # starting value

        new_hamiltonian = reduce(hamiltonian(new_site, 1), site.nearest, 0)
        new_hamiltonian = reduce(hamiltonian(new_site, self.J_NNN),
                                  site.next_nearest, new_hamiltonian)

        # Calculate contribution from external H-field
        h_ext = self.H_EXTERNAL * 7.0 / 2.0
        old_hamiltonian -= h_ext * cos(old_site.theta)
        new_hamiltonian -= h_ext * cos(new_site.theta)

        return new_hamiltonian - old_hamiltonian


    def populate_sites(self):
        """
        Creates a map of site indices to arrays of site indices
        that represent the nearest and next-nearest neighbors for
        each site.
        """
        origin = Site(0, (0, 0, 0))
        self.save_site(origin)

        seen = [origin.index]
        queue = [origin]

        while queue:
            s = self.add_neighbors(queue.pop(0))
            s = self.save_site(s)
            for n in s.neighbors():
                if n.index not in seen:
                    seen.append(n.index)
                    queue.append(n)


    def add_neighbors(self, site):
        """
        Updates the site with lists of its nearest and next-nearest
        neighbor sites.
        """
        def tuple_add(a, b):
            """
            Add two tuples together
            Ex: (0, 0, 1) + (1, 0, 1) = (1, 0, 2)
            See http://stackoverflow.com/a/497894
            """
            return tuple(map(operator.add, a, b))

        def add_neighbor_site(neighbor_delta, neighbor_sites_list):
            """
            Create the neighboring site using the delta coords
            and add it to the given list
            """
            coords = tuple_add(neighbor_delta, site.coords)
            neighbor_site = self.get_site(coords)
            neighbor_sites_list.append(neighbor_site)

        for neighbor_delta in nearest_neighbor_delta:
            add_neighbor_site(neighbor_delta, site.nearest)

        for neighbor_delta in next_nearest_neighbor_delta:
            add_neighbor_site(neighbor_delta, site.next_nearest)

        return site


    def get_site(self, coords):
        """
        Either fetches the site with the given coords
        or creates one.
        Makes sure coords are within the bounds of the fcc
        and wraps if they are not.
        """
        coords = tuple(map(lambda c: c % (self.size * 2), coords))
        if coords in self.coord_to_site:
            return self.coord_to_site[coords]
        site = Site(len(self.sites), coords)
        return self.save_site(site)


    def save_site(self, site):
        # Saves the site to our list of sites
        if site.index >= len(self.sites):
            site.index = len(self.sites)
            self.sites.append(site)
        else:
            self.sites[site.index] = site
            self.coord_to_site[site.coords] = site
        return site


    def get_num_sites(self, size):
        """
        Get the number of sites in a (n x n x n) fcc,
        where n = size.

        Explanation: When retrieving the nearest neighbor
        for a given site, indexing out of the bounds of the
        cube should wrap around and always provide a valid
        site. Therefore, when looking at a 1x1x1 fcc, only
        a single corner site along with 3 face sites are
        necessary to model, since the rest of the sites
        are effectively redundant, since all of the corner
        sites are actually the same site, just wrapped.
        """
        return pow(size, 3) * 4


s = Simulator(1)
s.run()
from pprint import pprint
pprint(s.records)

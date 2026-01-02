import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import random

# from simulator.spatial.field.freeField import FreeField
from simulator.spatial_vec.utils.random_generators import generate_uniform
from simulator.spatial_vec.logger.meshLogger import DataLogger
from simulator.spatial_vec.models.reaction import Reaction
from simulator.spatial_vec.layers.chemical import (
    calc_zero_order,
    calc_first_order,
    calc_second_order,
    calc_reaction_change,
)
from simulator.spatial_vec.layers.force import calc_vel


class FreeMesh:
    """
    Create a mesh of grid cells for 3D space. Performs flux and diffusion calculation, reaction, updates cells during simulation and other mesh and cell related functions.
    """

    def __init__(self, cfg, random_key=42):
        """

        :param self: GridMesh
        :param height: Height of the grid mesh - Y Axis. Must be >0
        :param width: Width of the grid mesh - X Axis. Must be >0
        :param depth: Depth of the grid mesh - Z Axis. Must be >0
        :param D: Diffusion constant
        :param cfg: uv Config
        :param random_key: Random key value for JAX random functions
        """

        self.width = cfg.width
        self.height = cfg.height
        self.depth = cfg.depth
        assert self.width > 0 and self.height > 0 and self.depth > 0, (
            "Spatial dimensions must be greater than 0"
        )
        self.dims = [self.width, self.height, self.depth]
        self.D = cfg.D
        self.n_chemicals = len(cfg["chemical"]["name"])

        self.key, self.sub_key = random.split(random.key(random_key))

        x, y, z = jnp.mgrid[0 : self.width, 0 : self.height, 0 : self.depth]
        self.positions = jnp.vstack([x.ravel(), y.ravel(), z.ravel()]).T

        self.cell_density = cfg.get("cell_density", 10)

        self.mesh_vol = self.height * self.width * self.depth
        self.n_cells = int(self.mesh_vol * self.cell_density)
        self.cell_vol = self.mesh_vol / self.n_cells

        ## Cell positions, sizes, mass

        self.key, self.sub_key, self.positions = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(self.n_cells, 3),
        )
        self.positions = jnp.round(self.positions, decimals=2)

        self.cells = {}
        self.key, self.sub_key, self.cell_sizes = generate_uniform(
            key=self.key, sub_key=self.sub_key, shape=(self.n_cells, 1)
        )

        self.cell_vel = jnp.zeros(shape=(self.n_cells, 3))

        self.cell_mass = jnp.ones_like(self.cell_sizes)

        self.reaction_bool = cfg.reaction_bool
        self.diffusion_bool = cfg.diffusion_bool

        self.repulsion_coeff = cfg.repulsion_coeff
        self.attraction_coeff = cfg.attraction_coeff
        self.drift_vel_coeff = cfg.drift_vel_coeff
        self.random_vel_coeff = cfg.random_vel_coeff

        self.debug_plot = cfg.debug_plot

        curr_id = 0
        self.delta_m = 0  # Adjust mass of chemicals if there is a mismatch in previous and current mass

        self.n_neighbours = 3  # TODO: Use config to set value

        ## Diffusion
        self.field_pos = []
        self.field_D = []
        self.field_vol = []
        self.field_id = []
        self.field_keys = []
        self.field_neighbours = []
        self.field_flux = []
        self.key, self.sub_key, self.field_chem = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(len(self.positions), self.n_chemicals, 1),
        )

        for [i, j, k] in self.positions:
            self.field_D.append(self.D)
            self.field_pos.append(tuple([i, j, k]))
            self.field_vol.append(self.cell_vol)
            self.field_keys.append(random.split(random.key(curr_id)))
            self.field_id.append(curr_id)
            self.field_neighbours.append(self.get_neighbours(jnp.array([i, j, k])))
            self.field_flux.append({})
            curr_id += 1

        self.field_keys = jnp.array(self.field_keys)
        self.field_id = jnp.array(self.field_id)

        ## Reaction

        self.chemicals = []
        self.chem_names = cfg["chemical"]["name"]
        self.reactions = []
        self.mol_masses = cfg["chemical"]["mol_mass"]
        self.delta = cfg["delta"]
        self.key, self.sub_key = random.split(self.key)
        self.use_prob = cfg["reaction_prob"]

        reaction_names = list(cfg["reaction"].keys())

        self.reaction_order = []
        self.reaction_order_sum = jnp.zeros((3,))
        self.reaction_matrix = []
        self.reactant_ids = []
        self.prod_ids = []

        for i in range(len(cfg["reaction"])):
            reaction_i = cfg["reaction"][reaction_names[i]]
            reaction_i_obj = Reaction(
                name=reaction_names[i],
                id=i,
                k=reaction_i.rate_coeff,
                order=reaction_i.order,
                products=[x for x in reaction_i.products]
                if reaction_i.products is not None
                else [],
                products_exp=reaction_i.products_exponent
                if reaction_i.products is not None
                else [],
                reactants=[x for x in reaction_i.reactants]
                if reaction_i.reactants is not None
                else [],
                reactants_exp=reaction_i.reactants_exponent
                if reaction_i.reactants is not None
                else [],
                chemicals=self.chem_names,
            )
            self.reaction_order.append(reaction_i_obj.order)
            self.reactions.append(reaction_i_obj)
            self.reaction_matrix.append(reaction_i_obj._generate_reaction_matrix())
            self.reaction_order_sum = self.reaction_order_sum.at[reaction_i.order].set(
                self.reaction_order_sum[reaction_i.order] + reaction_i.rate_coeff
            )
        self.n_reactions = len(self.reactions)
        self.reaction_order = jnp.array(self.reaction_order)
        self.reaction_matrix = jnp.array(self.reaction_matrix)
        self.reaction_prob = jnp.zeros((self.n_reactions,))
        for i in range(len(self.reactions)):
            if self.use_prob:
                if self.reaction_order_sum[self.reactions[i].order] > 0.0:
                    prob_i = (
                        self.reactions[i].k
                        * (
                            1
                            - jnp.exp(
                                -self.delta
                                * self.reaction_order_sum[self.reactions[i].order]
                            )
                        )
                    ) / self.reaction_order_sum[self.reactions[i].order]
                else:
                    prob_i = 0.0
            else:
                prob_i = 1.0
            self.reaction_prob = self.reaction_prob.at[i].set(prob_i)

        self.reaction_table = {
            0: calc_zero_order,
            1: calc_first_order,
            2: calc_second_order,
        }

        ## JIT and Vec functions

        self.reaction_change_vec = jax.vmap(
            calc_reaction_change,
            in_axes=(None, 0, None, 0, None, 0, None, None, None, None, None),
        )

    def calc_flux(self, cell_id):
        """
        Calculate flux for the cell at position _pos_.
        Also checks if part of the flux is already calculated by another cell before.

        :param pos: Position of cell to calculate the flux
        """
        curr_cell_id = cell_id.item()
        neighbour_cells = []
        neigh_pos = []
        cell_mass = []
        cell_vol = []
        flux = jnp.zeros((self.n_chemicals, 1))
        for cell_i_id in self.field_neighbours[curr_cell_id]:
            cell_i = self.field_id[int(cell_i_id)].item()
            if self.field_flux[curr_cell_id].get(cell_i) is not None:
                flux += self.field_flux[curr_cell_id].get(cell_i)
                continue
            neighbour_cells.append(cell_i)
            cell_mass.append(self.field_chem[cell_i])
            cell_vol.append(self.field_vol[cell_i])
            neigh_pos.append(jnp.array(self.field_pos[cell_i]))

        if len(neigh_pos) == 0:
            return flux

        neigh_pos = jnp.vstack(neigh_pos)
        cell_mass = jnp.array(cell_mass)
        cell_vol = jnp.array(cell_vol)
        distances = jnp.linalg.norm(
            neigh_pos - jnp.array(self.field_pos[curr_cell_id]), axis=1
        )

        def calc_flux_i(D, cell1_mass, cell1_vol, cell2_mass, cell2_vol, dist_i):
            """Helper function for auto vectorization for flux calculation"""
            flux_i = -D * (cell2_mass * cell2_vol - cell1_mass * cell1_vol) / dist_i
            return flux_i

        auto_vec_flux = jax.vmap(calc_flux_i, in_axes=(None, 0, 0, None, None, 0))

        flux_list = auto_vec_flux(
            self.field_D[curr_cell_id],
            cell_mass,
            cell_vol,
            self.field_chem[curr_cell_id],
            self.field_vol[curr_cell_id],
            distances,
        )

        for i in range(len(flux_list)):
            self.field_flux[neighbour_cells[i]][curr_cell_id] = -1 * flux_list[i]
            self.field_flux[curr_cell_id][neighbour_cells[i]] = flux_list[i]

        flux_list = jnp.append(flux_list, flux.reshape(-1, self.n_chemicals, 1), axis=0)
        total_flux = jnp.sum(flux_list, axis=0)
        return total_flux

    def calc_conc_change(self, delta):
        """
        Calculate change in chemical concentration due to diffusion

        :param self: GridMesh
        :param delta: Simulation delta
        """

        # TODO: Assuming area of boundary is 1. Change to dynamic
        def compute_delta_m(cell_id):
            flux = self.calc_flux(cell_id)
            area = 1
            delta_m = flux * area * delta
            if jnp.any(self.field_chem[cell_id] + delta_m < 0):
                fix_flux(cell_id, self.field_chem[cell_id] + delta_m)
            return delta_m

        def fix_flux(cell_id, diff_mass):
            neighbours = self.field_neighbours[cell_id]
            neg_idx = jnp.where(diff_mass < 0)
            diff_i = (diff_mass[neg_idx] / neighbours.shape[0]).reshape(-1)
            for i in neighbours:
                flux_i = self.field_flux[i.item()][cell_id][neg_idx] - diff_i
                self.field_flux[i.item()][cell_id] = (
                    self.field_flux[i.item()][cell_id].at[neg_idx].set(flux_i)
                )
                self.field_flux[cell_id][i.item()] = (
                    self.field_flux[cell_id][i.item()].at[neg_idx].set(-1 * flux_i)
                )

        delta_m_l = []
        ## Initial loop to fix negative masses
        self.vec_compute_delta_m = jax.vmap(compute_delta_m, in_axes=(0))
        # self.vec_compute_delta_m(self.field_id)

        ## TODO: Complete vectorization
        for i in self.field_id:
            compute_delta_m(cell_id=i)

        ## Final loop to set fixed masses
        ## TODO: Complete vectorization
        for i in self.field_id:
            delta_m_l.append(compute_delta_m(cell_id=i))

        prev_mass = 0
        new_mass = 0

        for i in self.field_id:
            chem_mass_i = self.field_chem[i]
            prev_mass += jnp.sum(chem_mass_i)
            chem_mass_i += delta_m_l[i]
            self.field_chem = self.field_chem.at[i].set(chem_mass_i)
            new_mass += jnp.sum(chem_mass_i)
            self.field_flux[i] = {}

        self.delta_m = (prev_mass - new_mass) / self.n_cells

    def calc_movement(self):
        # Calculate velocities
        for cell_id in range(self.n_cells):
            radial_neighs, radial_neigh_dists = self.get_radial_limits(
                self.positions[cell_id], radius=3
            )
            ret_vel = calc_vel(
                radial_neigh_dists,
                self.cell_sizes[cell_id],
                self.cell_sizes[radial_neighs],
                self.cell_mass[cell_id],
                self.cell_mass[radial_neighs],
                self.positions[cell_id],
                self.positions[radial_neighs],
                self.cell_vel[cell_id],
                attraction_coeff=self.attraction_coeff,
                repulsion_coeff=self.repulsion_coeff,
                drift_vel_coeff=self.drift_vel_coeff,
                delta=self.delta,
            )

            # Add in random velocity direction
            self.key, self.sub_key, random_vel = generate_uniform(
                key=self.key, sub_key=self.sub_key, shape=(3)
            )
            random_vel = self.random_vel_coeff * random_vel
            ret_vel = ret_vel + random_vel

            self.cell_vel = self.cell_vel.at[cell_id].set(ret_vel)

        # Compute cell movement
        for cell_id in range(self.n_cells):
            self.positions = self.positions.at[cell_id].set(
                self.positions[cell_id] + self.cell_vel[cell_id] * self.delta
            )

    def step(self, step_i, delta, logger: DataLogger):
        """
        Perform simulation step for the mesh and the cells within.

        :param self: GridMesh
        :param step_i: Step index
        :param delta: Simulation delta
        :param logger: MeshLogger object
        """

        # Perform diffusion
        if self.diffusion_bool:
            self.calc_conc_change(delta)
            logger is not None and logger.log_chem_state(
                step=step_i, field_chem=self.field_chem
            )
            logger is not None and print(
                f"Step - {step_i}, Delta M - {self.delta_m:.4e}"
            )

        # Perform reactions
        if self.reaction_bool:
            field_keys_i, chem_mass_i = self.reaction_change_vec(
                step_i,
                self.field_id,
                logger,
                self.field_keys,
                self.n_reactions,
                self.field_chem,
                self.delta,
                (calc_zero_order, calc_first_order, calc_second_order),
                self.reaction_prob,
                self.reaction_matrix,
                self.reaction_order,
            )
            self.field_chem = chem_mass_i
            self.field_keys = field_keys_i

        # Perform movement/force calculation
        self.calc_movement()

        if self.debug_plot:
            print("Step - ", step_i)
            plt.figure()
            ax = plt.subplot(projection="3d")
            ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
            plt.show()

    def get_cell_id(self, pos):
        """
        Returns an ID of a cell, useful for an order of cells to compute flux i->j and uniquely identify cells

        :param pos: Position of the cell (i,j,k)
        """
        return pos[0] + self.height * pos[1] + (self.height * self.depth) * pos[2]

    def get_neighbours(self, pos, norm_ord=1):
        """
        Get neighbours of the cell at pos[idx].

        :param self: GridMesh
        :param idx: index of cell within pos[idx]
        """
        l1_norm = jnp.linalg.norm(pos - self.positions, axis=1, ord=norm_ord)
        neigh_idxs = jnp.argsort(l1_norm)[1 : self.n_neighbours + 1]
        return neigh_idxs

    def get_radial_limits(self, pos, radius=1, norm_ord=1):
        l1_norm = jnp.linalg.norm(pos - self.positions, axis=1, ord=norm_ord)
        radial_neighs = jnp.where((l1_norm < radius) & (l1_norm > 0))
        return radial_neighs, l1_norm[radial_neighs]

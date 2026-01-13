import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import random
import math

# from simulator.spatial.field.freeField import FreeField
from simulator.spatial_vec.utils.random_generators import (
    generate_uniform,
    generate_normal,
)
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

        self.cell_concentration = cfg.get("cell_concentration", 10)

        self.mesh_vol = self.height * self.width * self.depth
        self.n_cells = int(self.mesh_vol * self.cell_concentration)
        self.field_vol_singular = self.mesh_vol / self.n_cells

        ## Cell positions, sizes, mass

        ## Random positions within the limits of Height, Width, Depth
        self.key, self.sub_key, self.positions = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(self.n_cells, 3),
            minval=jnp.array(
                [[-self.width / 2.0, -self.height / 2.0, -self.depth / 2.0]]
            ),
            maxval=jnp.array([[self.width / 2.0, self.height / 2.0, self.depth / 2.0]]),
        )
        self.positions = jnp.round(self.positions, decimals=2)

        ## Cellular-level attributes

        self.cell_vel = jnp.zeros(shape=(self.n_cells, 3))
        self.cell_states = jnp.zeros(shape=(self.n_cells, 1), dtype=jnp.int8)
        self.cell_time = jnp.zeros(shape=(self.n_cells, 1), dtype=jnp.int16)

        self.cycle_len = cfg.cycle_len
        self.interphase_len = self.cycle_len * 0.9
        self.mitosis_len = self.cycle_len * 0.1
        self.cell_death_prob = cfg.cell_death_prob

        self.key, self.sub_key, self.interphase_chkpt = generate_normal(
            key=self.key,
            sub_key=self.sub_key,
            mean=self.interphase_len,
            shape=(self.n_cells, 1),
            dtype=jnp.float16,
        )
        self.interphase_chkpt = jnp.round(self.interphase_chkpt).astype(jnp.int16)

        self.key, self.sub_key, self.mitosis_chkpt = generate_normal(
            key=self.key,
            sub_key=self.sub_key,
            mean=self.mitosis_len,
            shape=(self.n_cells, 1),
            dtype=jnp.float16,
        )
        self.mitosis_chkpt = self.mitosis_chkpt.at[self.mitosis_chkpt < 1].set(1)
        self.mitosis_chkpt = (
            jnp.round(self.mitosis_chkpt).astype(jnp.int16) + self.interphase_chkpt
        )

        self.cell_density = cfg.get("cell_density", 1.0)

        self.cell_target_vol_param = cfg.get("cell_target_vol", 1.0)
        self.cell_vol_growth_rate_param = cfg.get("cell_vol_growth_rate", 1.0)

        self.key, self.sub_key, self.cell_target_vol = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(self.n_cells, 1),
            minval=max(1e-4, self.cell_target_vol_param * 0.9),
            maxval=self.cell_target_vol_param * 1.1,
        )
        self.key, self.sub_key, self.cell_vol_growth_rate = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(self.n_cells, 1),
            minval=max(1e-4, self.cell_vol_growth_rate_param * 0.9),
            maxval=self.cell_vol_growth_rate_param * 1.1,
        )
        self.key, self.sub_key, self.cell_vol = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(self.n_cells, 1),
            minval=max(1e-4, self.cell_target_vol_param * 0.1),
            maxval=self.cell_target_vol_param,
        )
        self.cell_radius = jnp.pow((self.cell_vol * 3) / (4.0 * math.pi), 1 / 3)

        self.cell_mass = self.cell_density * self.cell_vol

        self.reaction_bool = cfg.reaction_bool
        self.diffusion_bool = cfg.diffusion_bool
        self.movement_bool = cfg.movement_bool
        self.cycle_bool = cfg.cycle_bool

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
            self.field_vol.append(self.field_vol_singular)
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
        Calculate flux for the field at position _pos_.
        Also checks if part of the flux is already calculated by another field before.

        :param pos: Position of field to calculate the flux
        """
        ## TODO: Rename cell_* to field_*
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
                self.cell_radius[cell_id],
                self.cell_radius[radial_neighs],
                self.cell_mass[cell_id],
                self.cell_mass[radial_neighs],
                self.positions[cell_id],
                self.positions[radial_neighs],
                self.cell_vel[cell_id],
                boundary_vals=(self.width, self.height, self.depth),
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

    def add_cell(self, pos, cell_state, new_radius, parent_growth_rate, target_vol):
        ## Arrays updated - Positions, state, time, radius, vol, mass
        self.positions = jnp.append(self.positions, pos, axis=0)
        self.cell_states = jnp.append(self.cell_states, cell_state, axis=0)
        self.cell_time = jnp.append(self.cell_time, jnp.array([[0]]), axis=0)
        new_vol = (4.0 * math.pi * new_radius**3) / 3.0
        self.cell_radius = jnp.append(self.cell_radius, new_radius, axis=0)
        self.cell_vol = jnp.append(self.cell_vol, new_vol, axis=0)
        self.cell_mass = jnp.append(
            self.cell_mass, jnp.array([self.cell_density * new_vol])
        ).reshape(-1, 1)

        self.key, self.sub_key, new_growth_rate = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(1, 1),
            minval=max(1e-4, 0.9 * parent_growth_rate),
            maxval=1.1 * parent_growth_rate,
        )
        self.key, self.sub_key, new_target_vol = generate_uniform(
            key=self.key,
            sub_key=self.sub_key,
            shape=(1, 1),
            minval=max(1e-4, 0.9 * target_vol),
            maxval=1.1 * target_vol,
        )
        self.key, self.sub_key, new_interphase_chkpt = generate_normal(
            key=self.key,
            sub_key=self.sub_key,
            mean=self.interphase_len,
            shape=(1, 1),
            dtype=jnp.float16,
        )
        self.key, self.sub_key, new_mitosis_chkpt = generate_normal(
            key=self.key,
            sub_key=self.sub_key,
            mean=self.mitosis_len,
            shape=(1, 1),
            dtype=jnp.float16,
        )

        self.interphase_chkpt = jnp.append(
            self.interphase_chkpt, new_interphase_chkpt, axis=0
        )
        self.mitosis_chkpt = jnp.append(self.mitosis_chkpt, new_mitosis_chkpt, axis=0)
        self.cell_target_vol = jnp.append(self.cell_target_vol, new_target_vol, axis=0)
        self.cell_vol_growth_rate = jnp.append(
            self.cell_vol_growth_rate, new_growth_rate, axis=0
        )
        self.n_cells += 1

    def calc_cycle(self):
        split_cells_mask = jnp.array(list(range(len(self.cell_states)))).reshape(-1)[
            (self.cell_states == 2).reshape(-1)
        ]
        for cell_id in split_cells_mask:
            cell_pos_i = self.positions[cell_id]

            self.key, self.sub_key, rand_point = generate_uniform(
                key=self.key, sub_key=self.sub_key, shape=(3)
            )
            t_sqrt = math.sqrt(
                self.cell_radius[cell_id][0] ** 2
                / (
                    (rand_point[0] - cell_pos_i[0]) ** 2
                    + (rand_point[1] - cell_pos_i[1]) ** 2
                    + (rand_point[2] - cell_pos_i[2]) ** 2
                )
            )
            cell_boundary_pt_1 = t_sqrt * (rand_point - cell_pos_i) + cell_pos_i
            cell_boundary_pt_2 = (
                -t_sqrt * (rand_point - cell_pos_i) + cell_pos_i
            ).reshape(1, 3)

            ## TODO: Check and Update all parameters of parent cell post split
            ## Parameters being updated - Radius, position, cell_state, cell_vol, cell_mass, cell_time

            new_radius = self.cell_radius[cell_id][0] / math.sqrt(2)
            self.positions = self.positions.at[cell_id].set(cell_boundary_pt_1)
            self.cell_states = self.cell_states.at[cell_id].set(0)

            self.cell_radius = self.cell_radius.at[cell_id].set(new_radius)
            self.cell_vol = self.cell_vol.at[cell_id].set(
                (4.0 * math.pi * new_radius**3) / 3.0
            )
            self.cell_mass = self.cell_mass.at[cell_id].set(
                self.cell_density * self.cell_vol[cell_id]
            )
            self.cell_time = self.cell_time.at[cell_id].set(0)

            self.add_cell(
                pos=cell_boundary_pt_2,
                cell_state=jnp.array([[0]]),
                new_radius=(new_radius).reshape(1, 1),
                parent_growth_rate=self.cell_vol_growth_rate[cell_id],
                target_vol=self.cell_target_vol[cell_id],
            )

        ## Check interphase
        transition_cell_mask = (self.cell_time >= self.interphase_chkpt) & (
            self.cell_states == 0
        )
        self.cell_states = self.cell_states.at[transition_cell_mask].set(1)

        ## Check mitosis
        interphase_cell_mask = (self.cell_time >= self.mitosis_chkpt) & (
            self.cell_states == 1
        )
        self.cell_states = self.cell_states.at[interphase_cell_mask].set(2)

        ## Update cell death
        self.key, self.sub_key, cell_death_prob = generate_uniform(
            key=self.key, sub_key=self.sub_key, shape=(self.n_cells)
        )
        cell_death_mask = jnp.any(
            (self.cell_states == 2) | (self.cell_states == 1) | (self.cell_states == 0)
        ) & (cell_death_prob <= self.cell_death_prob)
        self.cell_states = self.cell_states.at[cell_death_mask].set(-1)

        # Update cell times
        self.cell_time = self.cell_time.at[:].set(self.cell_time + 1)

    def calc_cell_growth(self):
        diff_target = self.cell_target_vol - self.cell_vol
        vol_inc = self.cell_vol_growth_rate * diff_target * self.delta
        self.cell_vol = self.cell_vol + vol_inc
        self.cell_radius = jnp.pow((self.cell_vol * 3) / (4.0 * math.pi), 1 / 3)
        self.cell_mass = self.cell_vol * self.cell_density

    def step(self, step_i, delta, logger: DataLogger):
        """
        Perform simulation step for the mesh and the cells within.

        :param self: GridMesh
        :param step_i: Step index
        :param delta: Simulation delta
        :param logger: MeshLogger object
        """
        if self.debug_plot:
            print("Step - ", step_i)

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
        if self.movement_bool:
            self.calc_movement()

        # Cell Cycling
        if self.cycle_bool:
            self.calc_cycle()

        # Cell Growth
        self.calc_cell_growth()

        if self.debug_plot:
            plt.figure()
            ax = plt.subplot(projection="3d")
            ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
            plt.show()
            print("\n\n")
        return self.cell_vol

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

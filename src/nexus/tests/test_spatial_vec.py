from nexus.tests.utils import log_cleanup, get_config
import jax.numpy as jnp
import jax
from nexus.simulator.spatial.spatialSim import SpatialSim
import numpy as np
import random


class TestSpatial:
    @log_cleanup
    def test_diffusion_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.reaction_bool = False
        base_config.spatial_sim.cycle_bool = False

        self.sim = SpatialSim(base_config.spatial_sim)
        self.sim.run_sim()
        if self.sim.mesh.n_fields < 2:
            assert False
        field_id_1 = 0
        field_id_2 = self.sim.mesh.field_neighbours[field_id_1][0]

        chem_before = self.sim.logger.retrieve_chem_data(
            step=0, field_id=[field_id_1, field_id_2]
        )["conc"]
        chem_after = self.sim.logger.retrieve_chem_data(
            step=-1, field_id=[field_id_1, field_id_2]
        )["conc"]

        min_chem_idx = np.argmin(chem_before)
        max_chem_idx = np.argmax(chem_after)

        if (
            chem_before[min_chem_idx] <= chem_after[min_chem_idx]
            and chem_before[max_chem_idx] >= chem_after[max_chem_idx]
        ):
            assert True
        else:
            assert False

    @log_cleanup
    def test_mass_preserving_freemesh(self):
        base_config = get_config("test_config")
        base_config.spatial_sim.reaction_bool = False
        base_config.spatial_sim.cycle_bool = False
        base_config.spatial_sim.n_steps = 30

        self.sim = SpatialSim(base_config.spatial_sim)

        def calc_mass():
            mass = 0
            for i in self.sim.mesh.field_id:
                mass += np.sum(self.sim.mesh.field_chem[i])
            return mass

        mass_before = calc_mass()
        self.sim.run_sim()
        mass_after = calc_mass()

        print("Before - ", mass_before)
        print("After - ", mass_after)
        if self.sim.mesh.n_fields < 2:
            assert False
        assert (mass_after - mass_before) <= 0.01 * mass_before

    @log_cleanup
    def test_reaction_freemesh(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.diffusion_bool = False
        base_config.spatial_sim.cycle_bool = False

        self.sim = SpatialSim(base_config.spatial_sim)
        self.sim.run_sim()
        chem_id = [1, 2]
        field_id = random.randint(a=0, b=self.sim.mesh.n_fields - 1)
        chem_before = self.sim.logger.retrieve_chem_data(
            step=0, field_id=field_id, chem_id=chem_id
        )["conc"]
        chem_after = self.sim.logger.retrieve_chem_data(
            step=-1, field_id=field_id, chem_id=chem_id
        )["conc"]

        if chem_before[0] >= chem_after[0] and chem_before[1] <= chem_after[1]:
            assert True
        else:
            assert False

    @log_cleanup
    def test_consistent_noise(self):
        base_config = get_config("freemesh_config")
        base_config.spatial_sim.diffusion_bool = False
        base_config.spatial_sim.cycle_bool = False
        base_config.spatial_sim.random_key = 100

        self.sim = SpatialSim(base_config.spatial_sim)
        self.sim.run_sim()

        self.sim2 = SpatialSim(base_config.spatial_sim)
        self.sim2.run_sim()

        for attr in self.sim.mesh.__dict__:
            if isinstance(getattr(self.sim.mesh, attr), jax.Array):
                arr1 = getattr(self.sim.mesh, attr)
                arr2 = getattr(self.sim2.mesh, attr)
                assert jnp.all(arr1 == arr2)

    @log_cleanup
    def test_resource_limit_hard(self):
        # Manually set the field chemical concentration and verifies the simulator accurately marks the cell for split or no-split
        cell_type = 0
        base_config = get_config("test_config")
        base_config.spatial_sim.cycle_bool = True
        base_config.spatial_sim.logging = False
        sim = SpatialSim(base_config.spatial_sim)

        cell_idx = jnp.where(sim.mesh.cell_type_mask == cell_type)[0][0]
        cell_fields = sim.mesh.get_assigned_fields()
        cell_field = cell_fields[cell_idx]
        chem1_idx = sim.mesh.chem_name_map["chem1"]
        sim.mesh.cell_contact_limit = {}
        sim.mesh.cell_resource_limit[cell_type] = (
            [["chem1", 0.1], ["chem2", 0.2]],
            ("hard", 1, "min"),
        )
        assert sim.mesh.cell_resource_limit[cell_type][1][0] == "hard"

        sim.mesh.field_chem = sim.mesh.field_chem.at[cell_field, chem1_idx, 0].set(0.05)
        assert not sim.mesh.check_cell_constraints(
            cell_id=cell_idx, cell_fields=cell_fields
        )

        sim.mesh.field_chem = sim.mesh.field_chem.at[cell_field, chem1_idx, 0].set(0.15)
        assert sim.mesh.check_cell_constraints(
            cell_id=cell_idx, cell_fields=cell_fields
        )

    @log_cleanup
    def test_resource_limit_hill(self):
        # Test the resource limit using hill function computation for cell split
        cell_type = 0
        base_config = get_config("test_config")
        base_config.spatial_sim.cycle_bool = True
        base_config.spatial_sim.logging = False
        sim = SpatialSim(base_config.spatial_sim)
        cells = jnp.where(sim.mesh.cell_type_mask == cell_type)[0]
        cell_fields = sim.mesh.get_assigned_fields()
        chem1_idx = sim.mesh.chem_name_map["chem1"]
        chem2_idx = sim.mesh.chem_name_map["chem2"]
        sim.mesh.cell_contact_limit = {}
        assert sim.mesh.cell_resource_limit[cell_type][1][0] == "hill"

        sim.mesh.field_chem = sim.mesh.field_chem.at[:, chem1_idx, 0].set(10.0)
        sim.mesh.field_chem = sim.mesh.field_chem.at[:, chem2_idx, 0].set(10.0)
        splits_prob_highchem = jnp.array(
            [
                sim.mesh.check_cell_constraints(
                    cell_id=cell_id, cell_fields=cell_fields
                )
                for cell_id in cells
            ],
        )
        sim.mesh.field_chem = sim.mesh.field_chem.at[:, chem1_idx, 0].set(0.01)
        sim.mesh.field_chem = sim.mesh.field_chem.at[:, chem2_idx, 0].set(0.01)
        splits_prob_lowchem = jnp.array(
            [
                sim.mesh.check_cell_constraints(
                    cell_id=cell_id, cell_fields=cell_fields
                )
                for cell_id in cells
            ],
        )
        assert jnp.sum(splits_prob_lowchem) < jnp.sum(splits_prob_highchem)

    @log_cleanup
    def test_contact_limit_hard(self):
        cell_type = 0
        base_config = get_config("test_config")
        base_config.spatial_sim.cycle_bool = True
        base_config.spatial_sim.logging = False
        sim = SpatialSim(base_config.spatial_sim)
        cell_fields = sim.mesh.get_assigned_fields()
        sim.mesh.cell_resource_limit = {}
        sim.mesh.cell_contact_limit[0] = ([3, 0.5], ("hard", 1))
        assert sim.mesh.cell_contact_limit[cell_type][1][0] == "hard"

        cells = jnp.where(sim.mesh.cell_states == cell_type)[0]
        sim.mesh.cell_positions = (
            jnp.zeros_like(sim.mesh.cell_positions).at[:].set(10.0)
        )
        sim.mesh.cell_positions = sim.mesh.cell_positions.at[cells[0]].set(0.0)
        assert sim.mesh.check_cell_constraints(
            cell_id=cells[0], cell_fields=cell_fields
        )

        sim.mesh.cell_positions = sim.mesh.cell_positions.at[cells[1]].set(
            jnp.array([0.1, 0.0, 0.0])
        )
        sim.mesh.cell_positions = sim.mesh.cell_positions.at[cells[2]].set(
            jnp.array([0.0, 0.1, 0.0])
        )
        sim.mesh.cell_positions = sim.mesh.cell_positions.at[cells[3]].set(
            jnp.array([0.0, 0.0, 0.1])
        )
        assert not sim.mesh.check_cell_constraints(
            cell_id=cells[0], cell_fields=cell_fields
        )

    @log_cleanup
    def test_contact_limit_hill(self):
        cell_type = 0
        base_config = get_config("test_config")
        base_config.spatial_sim.cycle_bool = True
        base_config.spatial_sim.logging = False
        sim = SpatialSim(base_config.spatial_sim)
        cell_fields = sim.mesh.get_assigned_fields()
        sim.mesh.cell_resource_limit = {}
        sim.mesh.cell_contact_limit[cell_type] = ([3, 0.5], ("hill", 1))
        cells = jnp.where(sim.mesh.cell_type_mask == cell_type)[0]
        assert sim.mesh.cell_contact_limit[cell_type][1][0] == "hill"

        sim.mesh.cell_positions = (
            jnp.zeros_like(sim.mesh.cell_positions).at[:].set(10.0)
        )
        sim.mesh.cell_positions = sim.mesh.cell_positions.at[cells[0]].set(0.0)
        splits_low_density = jnp.array(
            [
                sim.mesh.check_cell_constraints(
                    cell_id=cells[x], cell_fields=cell_fields
                )
                for x in cells
            ]
        )

        for i in cells[1:]:
            sim.mesh.cell_positions = sim.mesh.cell_positions.at[i].set(
                jnp.array([0.05 * i, 0.0, 0.0])
            )
        splits_high_density = jnp.array(
            [
                sim.mesh.check_cell_constraints(
                    cell_id=cells[x], cell_fields=cell_fields
                )
                for x in cells
            ]
        )
        assert jnp.sum(splits_high_density) < jnp.sum(splits_low_density)

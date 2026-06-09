from nexus.tests.utils import log_cleanup
import random
import copy
from tqdm import tqdm
import jax.numpy as jnp
from nexus.simulator.run_sim import check_config
import os
from hydra import initialize_config_dir, compose
from nexus.simulator.grn.grnSim import GRNSim
from nexus.simulator.spatial.spatialSim import SpatialSim
from nexus.simulator.intervention.intervention import InterventionManager


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg


class TestInterventions:
    base_config = get_config("test_config")
    cell_params = [
        "cell_positions",
        "cell_death_prob",
        "cell_prg_death_prob",
        "cell_density",
        "cell_target_vol",
        "cell_vol_growth_rate",
        "cell_attraction_coeff",
        "cell_repulsion_coeff",
        "cell_drift_vel_coeff",
        "cell_random_vel_coeff",
        "cell_death_decay_coeff",
        "cell_vel",
        # "cell_states", ## TODO: Create seperate test
        # "cell_time",  ## Should not be modified
        "cell_radius",
        "cell_mass",
    ]
    grn_params = [
        "decay",
        "prot_decay",
        "prot_tran_rates",
        "prot_half_lives",
        "ki_matrix",
    ]
    n_steps = 5
    interven_step = 1
    n_cell = 3

    base_config["grn"]["n_steps"] = n_steps
    base_config["spatial_sim"]["n_steps"] = n_steps
    base_config["grn"]["logging"] = False
    base_config["spatial_sim"]["logging"] = False
    base_config["intervention"]["spatial_sim"] = []
    base_config["intervention"]["grn"] = []
    base_config["intervention"]["other"] = []
    config = base_config
    random.seed(42)

    @log_cleanup
    def test_cell_scheduled_do_interventions(self):
        val_set = random.random()
        cell_range = list(range(3))
        base_config = copy.deepcopy(self.config)

        interventions = []
        for i in self.cell_params:
            interventions.append(
                [
                    i,
                    ["hard", val_set],
                    ["scheduled", self.interven_step],
                    ["index", cell_range],
                ]
            )
        base_config["intervention"]["spatial_sim"] = interventions

        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        for i in tqdm(range(self.interven_step + 1)):
            intervention_flag and interven_manager.check(i)
            if i == self.interven_step:
                for i in self.cell_params:
                    _param = getattr(self.spatial_sim.mesh, i)
                    assert jnp.all(
                        _param[jnp.array(cell_range)] == val_set
                    ) and not jnp.all(
                        _param[jnp.array(range(self.n_cell + 1, 8))] == val_set
                    )
                break
            else:
                self.grn_sim.run_sim(step=i)
                self.spatial_sim.run_sim(step=i)

    @log_cleanup
    def test_cell_pulse_do_intervention(self):
        base_config = copy.deepcopy(self.config)
        interven_start = 1
        interven_end = 3

        interventions = []
        pulse_val = random.random()
        cell_range = [0, 1]

        for i in self.cell_params:
            interventions.append(
                [
                    i,
                    ["hard", pulse_val],
                    ["pulse", interven_start, interven_end],
                    ["index", cell_range],
                ]
            )

        base_config["intervention"]["spatial_sim"] = interventions

        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True

        for t in tqdm(range(interven_end + 1)):
            intervention_flag and interven_manager.check(t)
            if t == interven_start or t == interven_end or t < interven_start:
                for i in self.cell_params:
                    current_param = getattr(self.spatial_sim.mesh, i)
                    vals_at_indices = current_param[jnp.array(cell_range)]
                    if t == interven_start:
                        assert jnp.all(vals_at_indices == pulse_val), (
                            f"Failed to start pulse at t={t}"
                        )
                        assert not jnp.all(
                            current_param[jnp.array(range(self.n_cell + 1, 8))]
                            == pulse_val
                        ), f"Modified other cell states at t={t}"
                    elif t == interven_end or t < interven_start:
                        assert jnp.all(vals_at_indices != pulse_val), (
                            f"Failed to revert pulse at t={t}"
                        )

    @log_cleanup
    def test_grn_do_interventions(self):
        val_set = random.random()
        cell_range = list(range(3))
        base_config = copy.deepcopy(self.config)

        interventions = []
        for i in self.grn_params:
            interventions.append(
                [
                    i,
                    ["hard", val_set],
                    ["scheduled", self.interven_step],
                    ["index", cell_range],
                ]
            )
        base_config["intervention"]["grn"] = interventions

        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        for i in tqdm(range(self.interven_step + 1)):
            intervention_flag and interven_manager.check(i)
            if i == self.interven_step:
                for i in self.grn_params:
                    _param = getattr(self.grn_sim, i)
                    assert jnp.all(
                        _param[jnp.array(cell_range)] == val_set
                    ) and not jnp.all(
                        _param[jnp.array(range(self.n_cell + 1, 8))] == val_set
                    )
                break
            else:
                self.grn_sim.run_sim(step=i)
                self.spatial_sim.run_sim(step=i)

    @log_cleanup
    def test_add_cell(self):
        base_config = copy.deepcopy(self.config)

        add_cell_step = 2
        base_config["intervention"]["other"] = [
            ["add_cell", ["scheduled", add_cell_step], [[1, 1, 1], 0, 0.4, 0]]
        ]
        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        prev_num_cells = 0
        for i in tqdm(range(add_cell_step + 1)):
            intervention_flag and interven_manager.check(i)
            if i == add_cell_step:
                interven_cells = self.spatial_sim.mesh.cell_positions.shape[0]
                assert interven_cells > prev_num_cells
                new_cell_pos = self.spatial_sim.mesh.cell_positions[-1]
                new_cell_state = self.spatial_sim.mesh.cell_states[-1].item()
                assert (
                    jnp.all(jnp.equal(new_cell_pos, jnp.array([1, 1, 1])))
                    and new_cell_state == 0
                )
            elif i < add_cell_step:
                prev_num_cells = self.spatial_sim.mesh.cell_positions.shape[0]
            self.grn_sim.run_sim(step=i)
            self.spatial_sim.run_sim(step=i)

    @log_cleanup
    def test_ablate_cell(self):
        base_config = copy.deepcopy(self.config)

        add_cell_step = 2
        base_config["intervention"]["other"] = [
            ["ablate", ["scheduled", add_cell_step], [[2], -1]],
            ["ablate", ["scheduled", add_cell_step], [[3], -2]],
        ]
        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        prev_num_cells = 0
        for i in tqdm(range(add_cell_step + 1)):
            intervention_flag and interven_manager.check(i)
            if i == add_cell_step:
                new_cell_state_1 = self.spatial_sim.mesh.cell_states[2].item()
                new_cell_state_2 = self.spatial_sim.mesh.cell_states[3].item()
                assert new_cell_state_1 == -1 and new_cell_state_2 == -2
            self.grn_sim.run_sim(step=i)
            self.spatial_sim.run_sim(step=i)

    @log_cleanup
    def test_chem_particle_interventions(self):
        base_config = copy.deepcopy(self.config)
        base_config["grn"]["n_steps"] = 10
        base_config["spatial_sim"]["n_steps"] = 10

        add_particle_step = 2
        remove_particle_step = 5
        add_chem_id = 0
        base_config["intervention"]["other"] = [
            [
                "add_particle",
                ["scheduled", add_particle_step],
                [[0.1, 0.1, 0.1], add_chem_id, 0.1],
            ],
            [
                "remove_particle",
                ["scheduled", remove_particle_step],
                [[0.1, 0.1, 0.1], add_chem_id],
            ],
        ]
        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")

        prev_chem = None
        new_chem = None

        remove_chem_conc = None
        assigned_field = self.spatial_sim.mesh.get_closest_field(
            pos=jnp.array([[0.1, 0.1, 0.1]])
        ).item()
        for i in tqdm(range(base_config.spatial_sim.n_steps)):
            intervention_flag and interven_manager.check(i)
            if i < add_particle_step:
                prev_chem = self.spatial_sim.mesh.field_chem[
                    assigned_field, add_chem_id
                ].item()
            elif i == add_particle_step:
                new_chem = self.spatial_sim.mesh.field_chem[
                    assigned_field, add_chem_id
                ].item()
            elif i > add_particle_step and i < remove_particle_step:
                prev_chem = new_chem
                new_chem = self.spatial_sim.mesh.field_chem[
                    assigned_field, add_chem_id
                ].item()
                assert prev_chem < new_chem
            else:
                if remove_chem_conc is None:
                    remove_chem_conc = self.spatial_sim.mesh.field_chem[
                        assigned_field, add_chem_id
                    ].item()
                assert (
                    remove_chem_conc
                    == self.spatial_sim.mesh.field_chem[
                        assigned_field, add_chem_id
                    ].item()
                )

            self.grn_sim.run_sim(step=i)
            self.spatial_sim.run_sim(step=i)

    @log_cleanup
    def test_modify_edge(self):
        base_config = copy.deepcopy(self.config)

        modify_edge_step = 2
        new_weight = -5.0
        target_id_1 = 0
        target_id_2 = 2
        reg_id_1 = 1
        reg_id_2 = 0
        base_config["intervention"]["other"] = [
            [
                "modify_edge",
                ["scheduled", modify_edge_step],
                [target_id_1, reg_id_1, new_weight, 1.0],
            ],
            [
                "block_edge",
                ["scheduled", modify_edge_step],
                [target_id_2, reg_id_2, 1.0],
            ],
        ]
        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        for i in tqdm(range(modify_edge_step + 1)):
            intervention_flag and interven_manager.check(i)
            if i == modify_edge_step:
                assert jnp.all(
                    self.grn_sim.ki_matrix[:, target_id_1, reg_id_1] == new_weight
                ) and jnp.all(self.grn_sim.ki_matrix[:, target_id_2, reg_id_2] == 0.0)
            self.grn_sim.run_sim(step=i)
            self.spatial_sim.run_sim(step=i)

    @log_cleanup
    def test_add_reaction(self):
        base_config = copy.deepcopy(self.config)
        base_config["grn"]["n_steps"] = 10
        base_config["spatial_sim"]["n_steps"] = 10
        base_config["spatial_sim"]["reaction"] = {}

        modify_reaction_step = 3
        reactant_id = 1
        product_id = 0
        base_config["intervention"]["other"] = [
            [
                "add_reaction",
                ["scheduled", modify_reaction_step],
                [
                    {
                        "reac_4": {
                            "order": 2,
                            "rate_coeff": 1,
                            "reactants": ["chem2"],
                            "reactants_exponent": [3],
                            "products": ["chem1"],
                            "products_exponent": [1],
                        }
                    }
                ],
            ]
        ]
        self.grn_sim = GRNSim(base_config.grn)
        self.spatial_sim = SpatialSim(base_config.spatial_sim)
        intervention_flag = False
        interven_manager = InterventionManager(
            cfg=base_config,
            spatial_obj=self.spatial_sim,
            grn_obj=self.grn_sim,
            key=base_config.intervention.get("random_key", 42),
        )
        intervention_flag = True
        check_config(spatial_sim=self.spatial_sim, cfg=base_config)

        ## Running sim
        print("Running simulators...")
        reactant_conc = None
        product_conc = None
        for i in tqdm(range(base_config["grn"]["n_steps"])):
            intervention_flag and interven_manager.check(i)
            print(self.spatial_sim.mesh.field_chem[:, product_id].reshape(-1))
            print(self.spatial_sim.mesh.field_chem[:, reactant_id].reshape(-1))
            if i > modify_reaction_step + 1:
                assert jnp.sum(
                    self.spatial_sim.mesh.field_chem[:, reactant_id]
                ) < jnp.sum(reactant_conc) and jnp.sum(
                    self.spatial_sim.mesh.field_chem[:, product_id]
                ) > jnp.sum(product_conc)

                reactant_conc = self.spatial_sim.mesh.field_chem[:, reactant_id]
                product_conc = self.spatial_sim.mesh.field_chem[:, product_id]
            elif i == modify_reaction_step:
                reactant_conc = self.spatial_sim.mesh.field_chem[:, reactant_id]
                product_conc = self.spatial_sim.mesh.field_chem[:, product_id]
            self.grn_sim.run_sim(step=i)
            self.spatial_sim.run_sim(step=i)

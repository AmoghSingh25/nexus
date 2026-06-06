import math
from nexus.simulator.spatial.utils.random_generators import generate_choices
from nexus.simulator.intervention.interventionVerify import check_intervene_params
from beartype.typing import TypeVar, Union, Tuple, Literal
import copy
from nexus.simulator.spatial.spatialSim import SpatialSim
from nexus.simulator.grn.grnSim import GRNSim
import jax.numpy as jnp
import jax
from beartype import beartype
from jaxtyping import jaxtyped
from omegaconf.listconfig import ListConfig

ARR = TypeVar("ARR", int, float)

ScheduledInterven = Tuple[Literal["scheduled"], ARR]
LoopInterven = Tuple[Literal["loop"], ARR, ARR, ARR]
PulseInterven = Tuple[Literal["pulse"], ARR, ARR]

TemporalIntervention = Union[
    ScheduledInterven[ARR], LoopInterven[ARR], PulseInterven[ARR]
]


@jaxtyped(typechecker=beartype)
class InterventionManager:
    def __init__(self, cfg, spatial_obj: SpatialSim, grn_obj: GRNSim, key: int = 42):
        self.spatial_interventions = cfg.intervention.get("spatial_sim", [])
        self.grn_interventions = cfg.intervention.get("grn", [])
        self.other_interventions = cfg.intervention.get("other", [])

        self.key, self.sub_key = jax.random.split(jax.random.key(key))

        self.spatial_checkpoints = {}
        self.grn_checkpoints = {}
        self.checkpoints = {}

        self.grn_obj = grn_obj
        self.spatial_obj = spatial_obj

        self.process_sim_interventions(
            cfg.spatial_sim,
            self.spatial_interventions,
            self.add_spatial_checkpoint,
            sim=self.spatial_obj.mesh,
        )
        self.process_sim_interventions(
            cfg.grn, self.grn_interventions, self.add_grn_checkpoint, sim=self.grn_obj
        )

        self.process_interventions(self.other_interventions, self.add_checkpoint)

    def add_spatial_checkpoint(self, t, data):
        if self.spatial_checkpoints.get(t) is None:
            self.spatial_checkpoints[t] = [data]
        else:
            self.spatial_checkpoints[t].append(data)

    def add_grn_checkpoint(self, t, data):
        if self.grn_checkpoints.get(t) is None:
            self.grn_checkpoints[t] = [data]
        else:
            self.grn_checkpoints[t].append(data)

    def add_checkpoint(self, t, data):
        if self.checkpoints.get(t) is None:
            self.checkpoints[t] = [data]
        else:
            self.checkpoints[t].append(data)

    def generate_temporal_checkpoints(
        self,
        temporal_params: TemporalIntervention,
        add_checkpoint_func,
        interven_i,
        reverse_intervention=None,
        n_steps=None,
    ):
        if temporal_params[0] == "scheduled":
            add_checkpoint_func(temporal_params[1], interven_i)

        elif temporal_params[0] == "pulse":
            start_pulse = temporal_params[1]
            end_pulse = temporal_params[2]

            add_checkpoint_func(start_pulse, interven_i)
            add_checkpoint_func(end_pulse, reverse_intervention)

        elif temporal_params[0] == "loop":
            start_loop = temporal_params[1]
            loop_length = temporal_params[2]
            gap = temporal_params[3]

            start_i = start_loop
            while start_i < n_steps:
                add_checkpoint_func(start_i, interven_i)
                if reverse_intervention is not None:
                    add_checkpoint_func(start_i + loop_length, reverse_intervention)

                start_i = start_i + loop_length + gap

    def generate_values(self, intervention_type):
        if intervention_type[0] == "hard":
            return ("hard", intervention_type[1])
        elif intervention_type[0] == "soft":
            match intervention_type[1]:
                case "shift":
                    return ("soft", intervention_type[2])
                case _:
                    raise ValueError("Invalid intervention type")

    def process_interventions(self, intervention_dict, add_checkpoint_func):
        for intervention_i in intervention_dict:
            intervention_type_i = intervention_i[0]
            temporal_params = intervention_i[1]
            interven_params = intervention_i[2]
            interven_params.insert(0, intervention_type_i)
            self.generate_temporal_checkpoints(
                temporal_params=tuple(temporal_params),
                add_checkpoint_func=add_checkpoint_func,
                interven_i=interven_params,
                reverse_intervention=None,
            )

    def process_sim_interventions(
        self, cfg, intervention_dict, add_checkpoint_func, sim
    ):
        for interven_i_conf in intervention_dict:
            interven_i = list(interven_i_conf)
            interven_param = interven_i[0]
            intervention_type = interven_i[1]
            temporal_params: TemporalIntervention = tuple(interven_i[2])
            spatial_params = interven_i[3]
            curr_val = getattr(sim, interven_param)
            if spatial_params[0] == "index":
                curr_val = curr_val[jnp.array(spatial_params[1])].tolist()

            ## Only reverse the intervention if it is a hard intervention and is a looped or pulsed intervention
            reverse_intervention = None
            if intervention_type[0] == "hard" and (
                temporal_params[0] == "loop" or temporal_params[0] == "pulse"
            ):
                reverse_intervention = copy.deepcopy(interven_i)
                reverse_intervention[1][1] = curr_val

            self.generate_temporal_checkpoints(
                temporal_params=tuple(temporal_params),
                add_checkpoint_func=add_checkpoint_func,
                interven_i=interven_i,
                reverse_intervention=reverse_intervention,
                n_steps=cfg.get("n_steps"),
            )

    def perform_intervention(self, sim_obj, interven_type, spatial_scope, param_name):
        curr_val = getattr(sim_obj, param_name)
        if spatial_scope[0] == "index":
            cell_range = jnp.array(spatial_scope[1])
        else:
            cell_range = jnp.array(range(len(curr_val)))

        if interven_type[0] == "hard":
            interven_val = interven_type[1]
        elif interven_type[0] == "soft":
            match interven_type[1]:
                case "shift":
                    interven_val = curr_val[cell_range] + interven_type[2]
        if isinstance(interven_val, ListConfig):
            interven_val = jnp.array(interven_val)

        if isinstance(curr_val, jax.Array):
            curr_val = curr_val.at[cell_range].set(interven_val)
        else:
            curr_val[cell_range] = interven_val
        check_intervene_params(
            sim_obj=sim_obj, curr_val=curr_val, param_name=param_name
        )
        setattr(sim_obj, param_name, curr_val)

    def check(self, t):
        ## TODO: Fix getting values before intervention

        # Perform interventions on the spatial sim
        if t in self.spatial_checkpoints:
            for i in self.spatial_checkpoints[t]:
                interven_param_name = i[0]
                interven_type = i[1]
                # temporal_scope = i[2]
                spatial_scope = i[3]

                self.perform_intervention(
                    self.spatial_obj.mesh,
                    interven_type=interven_type,
                    spatial_scope=spatial_scope,
                    param_name=interven_param_name,
                )

        # Perform interventions on the grn sim
        if t in self.grn_checkpoints:
            for i in self.grn_checkpoints[t]:
                interven_param_name = i[0]
                interven_type = i[1]
                # temporal_scope = i[2]
                spatial_scope = i[3]

                self.perform_intervention(
                    self.grn_obj,
                    interven_type=interven_type,
                    spatial_scope=spatial_scope,
                    param_name=interven_param_name,
                )

        if t in self.checkpoints:
            for intervention_i in self.checkpoints[t]:
                intervention_type = intervention_i[0]
                if intervention_type == "add_cell":
                    self.spatial_obj.mesh.add_cell(
                        pos=jnp.array(intervention_i[1]).reshape(1, -1),
                        cell_state=jnp.array([[intervention_i[2]]]),
                        new_radius=jnp.array([[intervention_i[3]]]),
                        parent_cell_id=intervention_i[4],
                    )
                elif (
                    intervention_type == "remove_cell" or intervention_type == "ablate"
                ):
                    selected_cells = jnp.zeros_like(
                        self.spatial_obj.mesh.cell_states
                    ).astype(jnp.bool)
                    selected_cells = selected_cells.at[
                        jnp.array(intervention_i[1])
                    ].set(1)
                    if intervention_i[2] == -1:
                        self.spatial_obj.mesh.kill_cells(selected_cells)
                    else:
                        self.spatial_obj.mesh.prg_death_cell(selected_cells)
                elif intervention_type == "add_particle":
                    self.spatial_obj.mesh.control_chem_generator(
                        pos=jnp.array(intervention_i[1]).reshape(1, -1),
                        compound_id=intervention_i[2],
                        rate=intervention_i[3],
                    )
                elif intervention_type == "remove_particle":
                    self.spatial_obj.mesh.modify_chem_generator(
                        pos=jnp.array(intervention_i[1]).reshape(1, -1),
                        compound_id=intervention_i[2],
                        new_rate=None,
                        delete_generator=True,
                    )
                elif intervention_type == "set_particle":
                    self.spatial_obj.mesh.modify_chem_generator(
                        pos=jnp.array(intervention_i[1]).reshape(1, -1),
                        compound_id=intervention_i[2],
                        new_rate=intervention_i[3],
                        delete_generator=False,
                    )
                elif (
                    intervention_type == "modify_edge"
                    or intervention_type == "block_edge"
                ):
                    target_gene = int(intervention_i[1])
                    regulator_gene = int(intervention_i[2])
                    if intervention_type == "block_edge":
                        weight = 0.0
                    else:
                        weight = (
                            float(intervention_i[3]) if len(intervention_i) > 3 else 1.0
                        )
                    ki_matrix = self.grn_obj.ki_matrix
                    is_mr = self.grn_obj.is_mr
                    n_cells = ki_matrix.shape[0]
                    k_prob = intervention_i[-1]
                    n_random_cells = math.ceil(n_cells * k_prob)

                    self.key, self.sub_key, selected_cells = generate_choices(
                        key=self.key,
                        sub_key=self.sub_key,
                        shape=(n_random_cells,),
                        a=n_cells,
                        replace=False,
                        p=jnp.ones(n_cells) * k_prob,
                    )
                    self.grn_obj.ki_matrix = ki_matrix.at[
                        selected_cells, target_gene, regulator_gene
                    ].set(weight)
                    if jnp.any(is_mr[selected_cells, target_gene]) and weight != 0.0:
                        self.grn_obj.is_mr = self.grn_obj.is_mr.at[
                            selected_cells, target_gene
                        ].set(False)
                    # if is_mr[target_gene] and weight == 0.0
                elif intervention_type == "modify_reaction":
                    reaction_names = list(intervention_i[1].keys())
                    self.spatial_obj.mesh.modify_reaction(
                        reaction_names=reaction_names, reaction_obj=intervention_i[1]
                    )
                elif intervention_type == "add_reaction":
                    reaction_names = list(intervention_i[1].keys())
                    self.spatial_obj.mesh.add_reaction(
                        reaction_names=reaction_names, reaction_obj=intervention_i[1]
                    )
        return self.spatial_obj

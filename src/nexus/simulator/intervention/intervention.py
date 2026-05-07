import copy
from nexus.simulator.spatial.spatialSim import SpatialSim
from nexus.simulator.grn.grnSim import GRNSim
import jax.numpy as jnp
import jax


class InterventionManager:
    def __init__(self, cfg, spatial_obj: SpatialSim, grn_obj: GRNSim):
        self.spatial_interventions = cfg.intervention.get("spatial_sim", [])
        self.grn_interventions = cfg.intervention.get("grn", [])

        self.spatial_checkpoints = {}
        self.grn_checkpoints = {}

        self.grn_obj = grn_obj
        self.spatial_obj = spatial_obj

        self.process_interventions(
            cfg.spatial_sim,
            self.spatial_interventions,
            self.add_spatial_checkpoint,
            sim=self.spatial_obj.mesh,
        )
        self.process_interventions(
            cfg.grn, self.grn_interventions, self.add_grn_checkpoint, sim=self.grn_obj
        )

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

    def process_interventions(self, cfg, intervention_dict, add_checkpoint_func, sim):
        for interven_i_conf in intervention_dict:
            interven_i = list(interven_i_conf)
            interven_param = interven_i[0]
            # interven_val = interven_i[1]
            temporal_params = interven_i[2]
            spatial_params = interven_i[3]
            curr_val = getattr(sim, interven_param)
            if spatial_params[0] == "index":
                curr_val = curr_val[jnp.array(spatial_params[1])].tolist()
            reverse_intervention = copy.deepcopy(interven_i)
            reverse_intervention[1] = curr_val

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
                while start_i < cfg.get("n_steps"):
                    add_checkpoint_func(start_i, interven_i)
                    add_checkpoint_func(start_i + loop_length, reverse_intervention)

                    start_i = start_i + loop_length + gap

    def perform_intervention(
        self, sim_obj, interven_param, spatial_scope, interven_val, param
    ):
        if spatial_scope[0] == "index":
            cell_range = jnp.array(spatial_scope[1])
        else:
            cell_range = jnp.array(range(len(param)))
        if isinstance(param, jax.Array):
            param = param.at[cell_range].set(interven_val)
        else:
            param[cell_range] = interven_val
        setattr(sim_obj, interven_param, param)

    def check(self, t):
        ## TODO: Fix getting values before intervention
        if t in self.spatial_checkpoints:
            # Perform interventions on the spatial sim
            for i in self.spatial_checkpoints[t]:
                interven_param = i[0]
                interven_val = i[1]
                # temporal_scope = i[2]
                spatial_scope = i[3]

                param = getattr(self.spatial_obj.mesh, interven_param)
                self.perform_intervention(
                    self.spatial_obj.mesh,
                    interven_param=interven_param,
                    spatial_scope=spatial_scope,
                    interven_val=interven_val,
                    param=param,
                )

        if t in self.grn_checkpoints:
            for i in self.grn_checkpoints[t]:
                interven_param = i[0]
                interven_val = i[1]
                # temporal_scope = i[2]
                spatial_scope = i[3]

                param = getattr(self.grn_obj, interven_param)
                self.perform_intervention(
                    self.grn_obj,
                    interven_param=interven_param,
                    spatial_scope=spatial_scope,
                    interven_val=interven_val,
                    param=param,
                )

            # Perform interventions on the grn sim
        return self.spatial_obj

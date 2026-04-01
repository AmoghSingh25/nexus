from simulator.spatial import spatialSim
from simulator.grn import grnSim
import jax.numpy as jnp
import jax


class InterventionManager:
    def __init__(self, cfg, spatial_obj: spatialSim, grn_obj: grnSim):
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
        ## TODO: Fix getting previously set attributes for looped/pulse changes
        for interven_i in intervention_dict:
            interven_param = interven_i[0]
            # interven_val = interven_i[1]
            temporal_params = interven_i[2]
            # spatial_params = interven_i[3]

            if temporal_params[0] == "scheduled":
                add_checkpoint_func(temporal_params[1], interven_i)

            elif temporal_params[0] == "pulse":
                start_pulse = temporal_params[1]
                end_pulse = temporal_params[2]
                curr_val = cfg.get(interven_param)

                add_checkpoint_func(start_pulse, interven_i)
                interven_i[1] = curr_val
                add_checkpoint_func(end_pulse, interven_i)

            elif temporal_params[0] == "loop":
                start_loop = temporal_params[1]
                loop_length = temporal_params[2]
                gap = temporal_params[3]

                curr_val = cfg.get(interven_param)

                start_i = start_loop
                while start_i < cfg.get("n_steps"):
                    add_checkpoint_func(start_i, interven_i)
                    interven_i[1] = curr_val
                    add_checkpoint_func(start_i + loop_length, interven_i)

                    start_i = start_i + loop_length + gap

    def check(self, t):
        if t in self.spatial_checkpoints:
            # Perform interventions on the spatial sim
            for i in self.spatial_checkpoints[t]:
                interven_param = i[0]
                interven_val = i[1]
                # temporal_scope = i[2]
                spatial_scope = i[3]
                param = getattr(self.spatial_obj.mesh, interven_param)

                if interven_param.startswith("cell_"):
                    if spatial_scope[0] == "index":
                        cell_range = jnp.array(spatial_scope[1])
                    else:
                        cell_range = jnp.array(range(len(param)))
                    if isinstance(param, jax.Array):
                        param = param.at[cell_range].set(interven_val)
                    else:
                        param[cell_range] = interven_val
                    setattr(self.spatial_obj.mesh, interven_param, param)
                else:
                    setattr(self.spatial_obj.mesh, interven_param, interven_val)

        if t in self.grn_checkpoints:
            for i in self.grn_checkpoints[t]:
                interven_param = i[0]
                interven_val = i[1]
                # temporal_scope = i[2]
                # spatial_scope = i[3]

                param = getattr(self.grn_obj, interven_param)
                if isinstance(param, jax.Array):
                    param = param.at[:].set(interven_val)
                else:
                    param = interven_val
                setattr(self.spatial_obj, interven_param, param)

            # Perform interventions on the grn sim
        return self.spatial_obj

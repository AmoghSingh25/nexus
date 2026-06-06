import jax


def check_intervene_params(sim_obj, curr_val, param_name):
    assert isinstance(getattr(sim_obj, param_name), type(curr_val)), (
        "Invalid configuration for intervention"
    )
    if isinstance(getattr(sim_obj, param_name), jax.Array):
        assert getattr(sim_obj, param_name).shape == curr_val.shape, (
            f"Incorrect shape of {param_name} parameter after intervention"
        )

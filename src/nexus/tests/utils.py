import functools
import os
from hydra import initialize_config_dir, compose


def log_cleanup(test_func):
    @functools.wraps(test_func)
    def test_wrapper(self, *args, **kwargs):
        try:
            return test_func(self, *args, **kwargs)
        finally:
            if hasattr(self, "sim") and self.sim is not None:
                self.sim.cleanup()
                self.sim = None
            if hasattr(self, "grn_sim") and self.grn_sim is not None:
                self.grn_sim.cleanup()
                self.grn_sim = None
            if hasattr(self, "spatial_sim") and self.spatial_sim is not None:
                self.spatial_sim.cleanup()
                self.spatial_sim = None
            if hasattr(self, "sim2") and self.sim2 is not None:
                self.sim2.cleanup()
                self.sim2 = None

    return test_wrapper


def get_config(config_name="test_config"):
    conf_path = os.path.join(os.getcwd(), "configs")
    with initialize_config_dir(version_base=None, config_dir=conf_path):
        cfg = compose(config_name=config_name)
    return cfg

from simulator.grnSim import GRNSim
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib

matplotlib.style.use("ggplot")


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print("Configuration being used - ")
    print(OmegaConf.to_yaml(cfg))
    s = GRNSim(cfg=cfg.grn)
    s.run_sim()


if __name__ == "__main__":
    main()

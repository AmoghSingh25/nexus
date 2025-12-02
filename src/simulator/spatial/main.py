from simulator.spatial.spatialSim import SpatialSim
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("ggplot")


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config.yaml"
)
def main(cfg: DictConfig) -> None:
    print("Configuration being used - ")
    print(OmegaConf.to_yaml(cfg))
    s = SpatialSim(cfg)
    conc_before = []
    conc_after = []
    for i in s.cells.reshape(-1):
        conc_before.append(i.chemical.mass)
    for i in s.cells.reshape(-1):
        s.calc_conc_change(i.pos)
    for i in s.cells.reshape(-1):
        conc_after.append(i.chemical.mass)
    plt.plot(conc_before, label="Before")
    plt.plot(conc_after, label="After")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

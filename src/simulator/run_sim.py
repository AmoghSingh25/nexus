import jax.numpy as jnp
import time
from simulator.grn.grnSim import GRNSim
from simulator.spatial_vec.spatialSim import SpatialSimVec
import hydra
from omegaconf import DictConfig, open_dict
import matplotlib.pyplot as plt


@hydra.main(
    version_base=None, config_path="../../configs", config_name="freemesh_config.yaml"
)
def run_sim(cfg: DictConfig) -> None:
    ## Checking configs
    timestep = str(int(time.time()))
    print("Time step - ", timestep)
    with open_dict(cfg):
        cfg.spatial_sim["log_file_name"] = timestep
        cfg.grn["log_file_name"] = timestep
    grn_sim = GRNSim(cfg.grn)
    prot_sim = grn_sim.protein_sim

    if prot_sim:
        # Modify the chemicals config in spatial simulation

        n_prots = grn_sim.n_genes
        chemicals = cfg.spatial_sim.chemical
        chem_names = chemicals.name
        chem_mol_mass = chemicals.mol_mass

        for i in range(n_prots):
            chem_names.append(f"Protein-{i}")
            chem_mol_mass.append(
                1
            )  ## TODO: Modify mol mass to protein mol mass, currently set to 1

        cfg.spatial_sim.chemical.name = chem_names
        cfg.spatial_sim.chemical.mol_mass = chem_mol_mass

    spatial_sim = SpatialSimVec(cfg.spatial_sim)
    check_config(spatial_sim=spatial_sim, cfg=cfg)

    associated_field = {}
    cell_mass = {}
    cell_vols = {}
    for i in range(spatial_sim.mesh.n_cells):
        cell_mass[i] = grn_sim.prot_conc[:, i]
        cell_vols[i] = spatial_sim.mesh.cell_vol[i]

        print("Cell mass - ", cell_mass[i].reshape(-1))
        closest_field_i = spatial_sim.mesh.get_closest_field(i).item()
        if associated_field.get(closest_field_i) is None:
            associated_field[closest_field_i] = [i]
        else:
            associated_field[closest_field_i].append(i)

    before_field_concs = []
    before_cell_concs = []
    after_field_concs = []
    after_cell_concs = []
    cell_delta_m = {}
    for i in associated_field:
        before_field_concs.append(spatial_sim.mesh.field_chem[i, -n_prots:])
        for j in associated_field[i]:
            before_cell_concs.append(cell_mass[j])
            cell_flux_field = 0
            flux_field_i = -spatial_sim.mesh.D * (
                cell_mass[j] / cell_vols[j]
                - spatial_sim.mesh.field_chem[i, -n_prots:]
                / spatial_sim.mesh.field_vol[i]
            )
            cell_delta_m[j] = flux_field_i * spatial_sim.delta
            grn_sim.prot_conc = grn_sim.prot_conc.at[:, j].set(
                grn_sim.prot_conc[:, j] + flux_field_i * spatial_sim.delta
            )
            after_cell_concs.append(grn_sim.prot_conc[:, j])

            cell_flux_field += cell_delta_m[j]

        spatial_sim.mesh.field_chem = spatial_sim.mesh.field_chem.at[i, -n_prots:].set(
            spatial_sim.mesh.field_chem[i, -n_prots:]
            - spatial_sim.delta * cell_flux_field
        )
        after_field_concs.append(spatial_sim.mesh.field_chem[i, -n_prots:])

    before_cell_concs = jnp.array(before_cell_concs)
    after_cell_concs = jnp.array(after_cell_concs)

    before_field_concs = jnp.array(before_field_concs)
    after_field_concs = jnp.array(after_field_concs)

    # print(before_field_concs[0, :].reshape(-1))
    # print(after_field_concs[0, :].reshape(-1))
    # print(before_cell_concs[0, :].reshape(-1))
    # print(after_cell_concs[0, :].reshape(-1))

    plt.show()
    ## Running sim
    print("Running spatial sim")
    # spatial_sim.run_sim()
    print("Running GRN sim")
    # grn_sim.run_sim()
    print("Completed running simulators")


def check_config(spatial_sim, cfg):
    if not cfg.grn.n_cells == spatial_sim.mesh.n_cells:
        raise ValueError(
            "Config values incorrect, no. of cells in spatial sim and GRN sim to be run together"
        )


if __name__ == "__main__":
    run_sim()

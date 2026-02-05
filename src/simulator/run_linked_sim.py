import jax.numpy as jnp
import time
from simulator.grn.grnSim import GRNSim
from simulator.spatial_vec.spatialSim import SpatialSimVec
import hydra
from omegaconf import DictConfig, open_dict
from simulator.utils.file_manager import _save_file


@hydra.main(
    version_base=None, config_path="../../configs", config_name="freemesh_config.yaml"
)
def run_sim(cfg: DictConfig) -> None:
    ## Checking configs
    timestep = str(int(time.time()))
    print("Time step file - ", timestep)
    with open_dict(cfg):
        cfg.spatial_sim["log_file_name"] = timestep
        cfg.grn["log_file_name"] = timestep
    grn_sim = GRNSim(cfg.grn)

    n_prots = grn_sim.n_genes
    chemicals = cfg.spatial_sim.chemical
    chem_names = []
    chem_mol_mass = []
    if chemicals is not None:
        chem_names = chemicals.name
        chem_mol_mass = chemicals.mol_mass

    for i in range(n_prots):
        chem_names.append(f"Protein-{i}")
        chem_mol_mass.append(
            1
        )  ## TODO: Modify mol mass to protein mol mass, currently set to 1

    with open_dict(cfg):
        cfg.spatial_sim["chemical"] = {}
        cfg.spatial_sim.chemical["name"] = chem_names
        cfg.spatial_sim.chemical["mol_mass"] = chem_mol_mass

    spatial_sim = SpatialSimVec(cfg.spatial_sim)
    check_config(spatial_sim=spatial_sim, cfg=cfg)

    cell_concs = [grn_sim.prot_conc]
    field_concs = [spatial_sim.mesh.field_chem]
    field_cell_assgns = []

    for i in range(20):
        associated_field = {}
        cell_mass = {}
        cell_vols = {}
        for i in range(spatial_sim.mesh.n_cells):
            cell_mass[i] = grn_sim.prot_conc[:, i]
            cell_vols[i] = spatial_sim.mesh.cell_vol[i]

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
            before_field_concs.append(spatial_sim.mesh.field_chem[i, :])
            cell_flux_field = 0
            for j in associated_field[i]:
                before_cell_concs.append(cell_mass[j])
                flux_field_i = -spatial_sim.mesh.D * (
                    cell_mass[j] / cell_vols[j]
                    - spatial_sim.mesh.field_chem[i, :] / spatial_sim.mesh.field_vol[i]
                )
                cell_delta_m[j] = flux_field_i * spatial_sim.delta
                grn_sim.prot_conc = grn_sim.prot_conc.at[:, j].set(
                    grn_sim.prot_conc[:, j] + flux_field_i * spatial_sim.delta
                )
                after_cell_concs.append(grn_sim.prot_conc[:, j])

                cell_flux_field += cell_delta_m[j]

            spatial_sim.mesh.field_chem = spatial_sim.mesh.field_chem.at[i, :].set(
                spatial_sim.mesh.field_chem[i, :] - cell_flux_field
            )
            after_field_concs.append(spatial_sim.mesh.field_chem[i, :])

        field_concs.append(spatial_sim.mesh.field_chem)
        cell_concs.append(grn_sim.prot_conc)
        field_cell_assgns.append(associated_field)

        before_cell_concs = jnp.array(before_cell_concs)
        after_cell_concs = jnp.array(after_cell_concs)

        before_field_concs = jnp.array(before_field_concs)
        after_field_concs = jnp.array(after_field_concs)

        spatial_sim.run_sim()
        grn_sim.run_sim()

    _save_file("src/simulator/spatial_vec/logs/cell_concs.pkl", cell_concs)
    _save_file("src/simulator/spatial_vec/logs/field_concs.pkl", field_concs)
    _save_file("src/simulator/spatial_vec/logs/field_cell_assgn.pkl", field_cell_assgns)

    # plt.show()
    ## Running sim


def check_config(spatial_sim, cfg):
    if not cfg.grn.n_cells == spatial_sim.mesh.n_cells:
        raise ValueError(
            "Config values incorrect, no. of cells in spatial sim and GRN sim to be run together"
        )


if __name__ == "__main__":
    run_sim()

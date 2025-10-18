# Simulator

The functioning of the simulator along with the references and mathematical models used.

## Calculation of concentration of genes

Uses a network of Master Regulators and Genes along with their parameters to simulate the concentrations of the genes.
Parameters involved

- $b$ - Basal production rate
- $K_{ij}$ - Regulatory influence of gene $j$ on the transcription of gene $i$
- $n_{ij}$ - Hill function for influence of regulator gene $j$ on gene $i$
- $(x_i)_t$ - Concentration of gene $i$ at time $t$
- $P_i$ - Production rate of gene $i$, modeled as the influence of regulator genes and basal production rate and shown in [Production rate of gene](#production-rate-of-gene).
- $h_ij$ - Half response

### Equations

#### Estimating steady state concentration

Equation proposed in [SERGIO](https://doi.org/10.1016/j.cels.2020.08.003)[1].

The simulator starts off from the steady state concentration of each gene to speed up the simulator. The equation for the calculation is

$E[x_i] = \frac{b_i}{\lambda_i}$ - If gene $i$ is a Master Regulator

$E[x_i] = \frac{\sum p_{ij}(E[x_j])}{\lambda_i}$ - if gene $i$ is not a Master Regulator

#### Production rate of gene

Equation proposed in [SERGIO](https://doi.org/10.1016/j.cels.2020.08.003).

$P_{i} = \sum{p_{ij}} + b_i$

$p_{ij} = K_{ij} \frac{x_j^{n_{ij}}}{h_{ij}^{n_{ij}} + x_{j}^{n_{ij}}}$ - If regulator $j$ is activator

$p_{ij} = K_{ij} (1-\frac{x_j^{n_{ij}}}{h_{ij}^{n_{ij}} + x_{j}^{n_{ij}}})$ - If regulator $j$ is repressor

#### Concentration of gene

Current simulator does not factor in noise and uses the following equation to compute the concentration of gene $i$ at time $t$. Equation proposed in [SERGIO](https://doi.org/10.1016/j.cels.2020.08.003).

$(x_i)_{t+1} = (x_i)_t + (P_i(t)- \lambda_ix_t(t))\Delta t$

### Calculation of concentration of Protein

The protein calculation equations are taken from the paper [Kuchta et al](https://doi.org/10.1016/j.cels.2020.08.003).[2]

#### Parameters

- $Pr_i(t)$ - Concentration of protein $i$ at time $t$
- $k_{trans,i}$ - Protein translation rate
- $k_{d,i}$ - Protein degradation rate
- $G_i(t)$ - Concentraiton of gene/mRNA $i$ at time t

The protein concentration is calculated as a factor of the mRNA that encodes the protein. The concentration of the mRNA is taken from the genetic simulator and used to compute the protein concentration.

#### Steady state concentration

Similar to the gene concentration simulator, the proteins are also initialized with the steady state concentraiton. This is done by setting $\frac{d[Pr_i(t)]}{dt}$ to 0 and computing $E[Pr_i(t)]$.

$E[Pr_i(t)] = \frac{k_{trans,i} * E[G_i(t)]}{k_{d,i}}$

#### Concentration of protein

$\frac{d[Pr_i(t)]}{dt} = k_{trans,i} [G_i(t)] - k_{d,i}[Pr_i(t)]$

The Euler-Maruyama method is used in the current simulator to solve this differential and for computing the concentrations.

$(Pr_i)_{t+1} = (Pr_i)_t + (P_i(t)- \lambda_ix_t(t))\Delta t$

## References

[1] - [Dibaeinia, P., & Sinha, S. (2020). SERGIO: A Single-Cell Expression Simulator Guided by Gene Regulatory Networks.](https://doi.org/10.1016/j.cels.2020.08.003)

[2] - [Kuchta, K., Towpik, J., Biernacka, A., Kutner, J., Kudlicki, A., Ginalski, K., & Rowicka, M. (2018). Predicting proteome dynamics using gene expression data. Scientific Reports, 8(1), 13866](https://doi.org/10.1038/s41598-018-31752-4)

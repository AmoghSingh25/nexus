# Spatial Simulator - Cellular Level

# Diffusion

The area is divided into regions using a form of triangulation/neighbor computation and a region definition to generate regions using the neighborsA

## Ficks law

Considering the amount of substance for calculating concentration.


$$
\begin{align*}
C_1 = M_1 / V_1 \\ 
C_2 = M_2 / V_2 \\
J = -D \nabla C \approx \frac{|C_2 - C_1|}{\lVert pos_2- pos_1 \rVert} \\
D - \text{Diffusion constant}
\end{align*}
$$
Each particle or field contains $M_x$ amount of chemical used for calculating the concentration. This can later be used to impose a conservation of mass as

$$
\begin{align*}
\Delta M = J_{i->j}A\Delta t = -D\nabla C A\Delta t \\
M_{t+\Delta t}^i = M_{t}^i - \Delta M \\
M_{t+\Delta t}^j = M_t^j + \Delta M \\
M_t^j + M_t^i = M_{t+\Delta t}^j + M^i_{t+\Delta t} = M^j_0 + M^i_0
\end{align*}
$$

For Permeable boundaries, using P. Flux can now be calculated as

$$
J_{i->j, t} = - P_{ij} [C_{j,t} - C_{i,t}]
$$
$$
\begin{align*}
J_{i,t} = \sum_{j\subseteq neighbours}-P_{ij}[C_{j,t} - C_{i,t}] \\
\Delta C_{i,t} = -\nabla . J_{i,t} \Delta t \\
C_{i,t+\Delta t} = C_{i,t} + \Delta C_{i,t} \\
J_{i->j, t} = - J_{j->i, t} \\
P_{i,j} = P_{j,i}
\end{align*}
$$

### Choice of $D$

$$
\begin{align*}
D = \mu k_B T \\
\text{where } \mu=\frac{v_d}{F} \\
\text{or} \\
\text{For charged particles - }D = \frac{\mu k_B T}{q} \\
\text{or} \\
\text{Diffusion of spherical particles through liquid - }D=\frac{k_BT}{6\pi \eta r}
\end{align*}
$$

$\mu$ - Mobility 
$v_d$ - Terminal drift velocity under $F$ force
$k_B$ - Boltzmann constant
$q$ - Charge
$\eta$ - Dynamic viscosity
$r$ - Stokes radius
$T$ - Temperature

# Rate equations

## Zeroth Order

Produced spontaneously, produce substances randomly to the environment

### Deterministic equation

$$
[A]^{t+\Delta t} = [A]^{t} + k_0\Delta t
$$

### Stochastic equation

Adds in stochastic noise by sampling from a Poisson distribution

$$
\begin{align*}
[A]^{t+\Delta t} = [A]^{t} + Sample(Poisson(\lambda)) \\
\lambda = k_0\Delta t \\
Poisson(\lambda;n) = \frac{\lambda^n e^{-\lambda}}{n!}
\end{align*}
$$

## First order

$A→B$

Smoldyn uses a probability of reaction

$Prob(reaction) = 1-e^{-k_1 \Delta t}$

For multiple reactions,

$$
Prob(reaction_i) = \frac{K_{1,i}}{\sum_jK_{1,j}}[1-e^{-\Delta t \sum_j K_{1,j}}]
$$

### Deterministic reaction

$$
\begin{align*}
\ln[A]^{t+\Delta t} = \ln [A]^t - k_1\Delta t \\
\ln[B]^{t+\Delta t} = \ln [B]^t + k_1\Delta t \\
\end{align*}
$$

### Stochastic reaction

$$
\begin{align*}
\gamma = Sample(Poisson(\lambda)) \\
\ln[A]^{t+\Delta t} = \ln [A]^t - \gamma \\
\ln[B]^{t+\Delta t} = \ln [B]^t + \gamma \\
\end{align*}
$$

## Second order

$A+B → C$

Second order wrt to A

### Deterministic reaction

$$
\frac{1}{[A]^{t+\Delta t}} = \frac{1}{[A]^{t}} - k_2\Delta t
$$

### Stochastic reaction

$$
\begin{align*}
\gamma = Sample(Poisson(\lambda)) \\
\frac{1}{[A]^{t+\Delta t}} = \frac{1}{[A]^{t}} - \gamma \\
\end{align*}
$$

$A+B→C$

Second order wrt to A and B

### Deterministic reaction

$$
\frac{1}{[B]_0 - [A]_0}\ln \frac{[B][A]_0}{[A][B]_0} = kt
$$

For single reactant concentration

Assuming $1A + 1B → 1C$

$$
[A] = \frac{[A]_0([B]_0- [A]_0)}{[B]_0e^{kt([B]_0- [A]_0)} - [A]_0}
$$

Substituting $[A]=[A]_0-x$

Assuming $[C]_0$ is 0

$$[C] = \frac{[A]_0[B]_0(e^{kt([B]_0 - [A]_0)}-1)}{[B]_0e^{kt([B]_0-[A]_0)}-[A]_0}$$
### Stochastic reaction


$$
\frac{1}{[B]_0 - [A]_0}\ln \frac{[B][A]_0}{[A][B]_0} = \gamma
$$

For single reactant concentration

Assuming $1A + 1B → 1C$

$$
[A] = \frac{[A]_0([B]_0- [A]_0)}{[B]_0e^{\gamma([B]_0- [A]_0)} - [A]_0}
$$

Substituting $[A]=[A]_0-x$

Assuming $[C]_0$ is 0

$$[C] = \frac{[A]_0[B]_0(e^{\gamma([B]_0 - [A]_0)}-1)}{[B]_0e^{\gamma([B]_0-[A]_0)}-[A]_0}$$

# Movement and Forces
### Reference - Active inter-cellular forces in collective cell motility

## Inter cellular force
Compute a repulsion and attraction force. The repulsion force is used for correcting cell overlap while cell attraction is used for modelling tissue-like clumping of cells. Repulsion force is computed such that the cells do not overlap in a single simulation step while considering the scaling by _delta_. First, the force vectors are computed which denote the attractive force between the cells.

$$ \forall i \in neigh(x): \bar{f_{x,i}^a} = pos_i - pos_x $$
$$ \bar{f_{x,i}^r} = - \bar{f_{x,i}^a} $$

where $\bar{f_{x,i}^a}$ is the attraction force and $\bar{f_{x,i}^r}$ is the repulsion force.

The repulsion force is calculated as

$$f^{r}_{x,i} = \frac{2.0 * [Pos(x) - Pos(i)]}{\Delta} * [M(x) + M(i)] $$

where $M(x)$ is the mass of the cell _x_ and $Pos(x)$ returns the position of the cell.


The attraction force is calculated as 

$$
f^{a}_{x,i} = \frac{M(x) * M(i)}{\lVert Pos(x)-Pos(i) \rVert ^2}
$$

The force is modeled after the inverse square law.

Magnitudes of the repulsion and attraction force on the cell can be controlled using _attraction_coeff_ and _repulsion_coeff_.

## Drift velocity
Assigns a drift velocity to the cell that is a fraction of the previous step velocity of the cell. Magnitude is controlled using _drift_vel_coeff_.

## Random velocity
Add a velocity in a random direction to the cell. Magnitude of the velocity is controlled using _random_vel_coeff_
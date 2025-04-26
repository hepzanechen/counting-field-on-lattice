trivial vortices (with CdGM spectra in accordance with $\left.m= \pm \frac{1}{2}, \pm \frac{3}{2}, \ldots\right)$ has been reported ${ }^{[17]}$ A possible explanation is a high sensitivity of the surface topological superconducting phase to the exact stoichiometric composition and local chemical potential. ${ }^{[20|21| 29]}$ There are other concerns regarding the MVM interpretation of experimental results in the putative Fu-Kane materials. They include the possible trivial origin of a non-split zeroenergy vortex bound state, ${ }^{[30]}$ the sensitivity of the vortex subgap state's energy spacings to the pairing profile $\Delta(r)$ and to impurities, ${ }^{[31}$ or the lack of a robustly quantized conductance plateau in a strong-coupling transport experiment. ${ }^{[32]}$

In this work we propose a framework to identify the presence (or absence) of MVMs in the twodimensional platform using ideas of non-local transport first developed for one-dimensional superconducting heterostructures ${ }^{[6]}$ In particular, we propose to use a non-local transport measurement to spatially map the ratio $[q / n](\mathbf{r})$ of local charge-density $(q)$ and probabilitydensity $(n)$ of sub-gap wavefunctions at various energies. We discuss how the data reveals tell-tale signatures of either topological MVM or ordinary CdGM states. In contrast to a closely related pioneering experiment on a one-dimensional quantum wire, ${ }^{9}$ the application of the proposed technique to realistic vortex modes comes with a number of important modifications: In the onedimensional wire case, the spatial resolution is usually limited to the positions of the tunneling contacts at the two ends of the wire as STM is not applicable. In the twodimensional case at least one of the two required surface contacts can be realized as a movable STM tip (see "T" in Fig. 1), which is sufficient to achieve a spatially resolved $q / n$. The second contact ("C") can be another STM tip,,$\sqrt{33+\sqrt{35}}$ if available, or any other extended type of electrical contact like a patterned metallic overlayer or a graphene flake.

A second important difference pertains to the complexity of the electronic system: While an ideal onedimensional topological superconductor harbors two Majorana zero modes at its ends, the two-dimensional situation is characterized by the fact that vortices (and their putative MVMs) are located in a disordered lattice with local but essentially random hybridizations ${ }^{36[37}$ that modify the spectrum from the case of a uniform lattice ${ }^{[38 / 39}$ Although we start discussing the most simple case of a single vortex-pair analytically, we then take into account experimental reality with many vortices using extensive numerical simulations based on a tight-binding model of the Dirac Hamiltonian.

The rest of the paper is organized as follows: In Sec. II we present the low-energy two-dimensional FuKane model and its tight-binding approximation. We then review the description of non-local superconducting quantum transport in Sec. III The case of a single vortex pair is treated in Sec. IV which is suitable to present our protocol proposed for experiments. The applicability of our main ideas to a realistic disordered vortex lattice is
demonstrated in Sec. V and a conclusion is contained in Sec. VI.

## II. MODEL AND VORTEX MODES

We consider a single two-dimensional Dirac surface Hamiltonian $\mathcal{H}_{0}=-i \hbar v\left[\sigma_{x} \partial_{x}+\sigma_{y} \partial_{y}\right]-\mu$ with velocity $v$, chemical potential $\mu$ and the $\sigma$-Pauli matrices acting in spin-space ${ }^{40}$ The second-quantized s-wave pairing Hamiltonian reads ${ }^{11 / 36 \mid 37}$

$$
\begin{equation*}
H_{B C S}=\int_{\mathbf{r}} \psi_{\mathbf{r}}^{\dagger} \mathcal{H}_{0} \psi_{\mathbf{r}}+\Delta \psi_{\mathbf{r}, \uparrow}^{\dagger} \psi_{\mathbf{r}, \downarrow}^{\dagger}+\Delta^{*} \psi_{\mathbf{r}, \downarrow} \psi_{\mathbf{r}, \uparrow} \tag{1}
\end{equation*}
$$

where $\Delta$ is the pairing field and the spinor of electronic annihilation operators is given by $\psi_{\mathbf{r}}=\left(\psi_{\uparrow, \mathbf{r}}, \psi_{\downarrow, \mathbf{r}}\right)^{\mathrm{T}}$. The ansatz $\psi_{\mathbf{r}, \sigma} \equiv \sum_{n} u_{\sigma, n}(\mathbf{r}) \gamma_{n}+v_{\sigma, n}^{*}(\mathbf{r}) \gamma_{n}^{\dagger}$ leads to the following Bogoliubov-de Gennes (BdG) equations for eigenmodes $\gamma_{n}$ and -energies $E_{n}$,

$$
\begin{align*}
\mathcal{H}_{B d G} \Phi(\mathbf{r}) & =E_{n} \Phi(\mathbf{r})  \tag{2}\\
\mathcal{H}_{B d G} & =\tau_{z}\left(v\left[\sigma_{x} p_{x}+\sigma_{y} p_{y}\right]-\mu\right)+\tau_{x} \operatorname{Re} \Delta-\tau_{y} \operatorname{Im} \Delta \tag{3}
\end{align*}
$$

with $\Phi^{\mathrm{T}}(\mathbf{r})=\left(u_{\uparrow}, u_{\downarrow}, v_{\downarrow},-v_{\uparrow}\right)$ and Pauli matrices $\tau_{\mu}$ acting in particle-hole space. The particle-hole symmetry is $\mathcal{P}=\sigma_{y} \tau_{y} \mathcal{K}$ with $\mathcal{P}^{2}=+1$ and $\mathcal{K}$ complex conjugation. In the homogeneous case, the energies for momentum $\mathbf{k}$ are given by $E_{\mathbf{k}}= \pm\left(\Delta^{2}+( \pm v k-\mu)^{2}\right)^{1 / 2}$.

A magnetic field $B_{z}$ applied orthogonal to the surface creates vortices in the pairing field, ${ }^{41}$

$$
\begin{equation*}
\Delta(\mathbf{r})=\Delta_{0} \prod_{j} f\left(\left|\mathbf{r}-\mathbf{R}_{j}\right|\right) \frac{\left(x-x_{j}\right)+i\left(y-y_{j}\right)}{\left|\mathbf{r}-\mathbf{R}_{j}\right|} \tag{4}
\end{equation*}
$$

with $\mathbf{R}_{j}=x_{j} \mathbf{e}_{\mathbf{x}}+y_{j} \mathbf{e}_{\mathbf{y}}$ the vortex positions and the function $f(r)=\tanh (r / \xi)$ modeling the decay of the pairing amplitude from its bulk value $\Delta_{0}$ towards the vortex core within lengthscale $\xi$. For a single vortex at the origin, Eq. (4) reduces to the simple polarcoordinate expression $\Delta(r, \phi)=\Delta_{0} f(r) e^{i \phi}$. The magnetic field can be found from the solution of the London equation which, for the single vortex case, reads $B_{z}(r)=\frac{\Phi_{0}}{2 \pi \lambda^{2}} K_{0}(r / \lambda)$ with corresponding vector potential $\mathbf{A}(\mathbf{r})=\mathbf{e}_{\phi} \frac{\Phi_{0}}{2 \pi r}\left[1-\frac{r}{\lambda} K_{1}(r / \lambda)\right]$ in the London gauge. Here, $\Phi_{0}=\pi \hbar / e$ is the magnetic flux quantum piercing the vortex while the radial decay of $B_{z}(r)$ is controlled by the London penetration depth $\lambda$. The modified Bessel function of the second kind is denoted by $K_{l}(x)$. The vector potential enters in the Hamiltonian via the replacement $\mathbf{p} \rightarrow \mathbf{p}-\tau_{z} e \mathbf{A}(\mathbf{r})$. The generalization to the vector potential for multiple vortices corresponding to Eq. (4) is straightforward, $\mathbf{A}(\mathbf{r}) \rightarrow \sum_{j} \mathbf{A}\left(\mathbf{r}-\mathbf{R}_{j}\right)$.

For numerical simulations, we regularize the continuum model on a two-dimensional square lattice. We set the lattice constant $a=1$, along with the choice $v=1$, $\hbar=1$. The straightforward regularization $\mathcal{H}_{0} \rightarrow \mathcal{H}_{0, L}=$ $\sum_{\mathbf{k}} \sigma_{x} \sin k_{x}+\sigma_{y} \sin k_{y}+\sigma_{z}\left(-2+\cos k_{x}+\cos k_{y}\right)-\mu \operatorname{can}$
be improved upon replacing $\sin (k) \rightarrow \frac{4}{3} \sin (k)-\frac{1}{6} \sin (2 k)$ and $\cos (k) \rightarrow \frac{4}{3} \cos (k)-\frac{1}{3} \cos (2 k)$ which more faithfully approximates the continuum model $\mathcal{H}_{0}$ around $\mathbf{k}=0$ by canceling series expansion coefficients of order $k_{x}^{3}$ and $k_{y}^{4}$ at the cost of involving hoppings along bonds $2 a \mathbf{e}_{x, y}$. This will ultimately allow us to choose a large chemical potential $(\mu=0.6)$ for the simulations in the lattice model while still approximating the dispersion of the continuum model at the Fermi level to a satisfactory degree. This in turn yields a small length scale for the Fermi wavelength $k_{F}^{-1}\left(\mu=\hbar v k_{F}\right)$ allowing for tractable overall system sizes. In real space, the lattice Hamiltonian reads

$$
\begin{align*}
H_{0, L} & =\sum_{\mathbf{r}} c_{\mathbf{r}}^{\dagger}\left[-2 \sigma_{z}-\mu\right] c_{\mathbf{r}}  \tag{5}\\
& +c_{\mathbf{r}+\mathbf{e}_{\mathbf{x}}}^{\dagger}\left[\frac{4}{3} \times \frac{\sigma_{z}+i \sigma_{x}}{2}\right] c_{\mathbf{r}} \\
& +c_{\mathbf{r}+\mathbf{e}_{\mathbf{y}}}^{\dagger}\left[\frac{4}{3} \times \frac{\sigma_{z}+i \sigma_{y}}{2}\right] c_{\mathbf{r}} \\
& +c_{\mathbf{r}+2 \mathbf{e}_{\mathbf{x}}}^{\dagger}\left[-\frac{1}{6} \times \frac{2 \sigma_{z}+i \sigma_{x}}{2}\right] c_{\mathbf{r}} \\
& +c_{\mathbf{r}+2 \mathbf{e}_{\mathbf{y}}}^{\dagger}\left[-\frac{1}{6} \times \frac{2 \sigma_{z}+i \sigma_{y}}{2}\right] c_{\mathbf{r}}+\text { h.c. }
\end{align*}
$$

and the BdG Hamiltonian becomes

$$
\mathcal{H}_{B d G, L}=\left(\begin{array}{cc}
\mathcal{H}_{0, L} & \Delta  \tag{6}\\
\Delta^{*} & -\sigma_{y} \mathcal{H}_{0, L}^{*} \sigma_{y}
\end{array}\right)
$$

The inclusion of magnetic field and vortices in the lattice model is achieved via a discretized version of Eq. (4) and the Peierls substitution for the hopping matrix element from $\mathbf{r}_{1}$ to $\mathbf{r}_{2}$ in $H_{0, L}$,

$$
\begin{equation*}
t_{\mathbf{r}_{2}, \mathbf{r}_{1}} \rightarrow t_{\mathbf{r}_{2}, \mathbf{r}_{1}} \exp \left(\frac{i e}{\hbar} \int_{\mathbf{r}_{1}}^{\mathbf{r}_{2}} d \mathbf{r} \cdot \mathbf{A}(\mathbf{r})\right) \tag{7}
\end{equation*}
$$

In the limit $\lambda \gg a$, the argument of the exponent can be approximated by $i \sum_{j} \frac{\theta_{j}\left(\mathbf{r}_{12}\right)}{2}\left[1-\frac{r_{j}\left(\mathbf{r}_{12}\right)}{\lambda} K_{1}\left(r_{j}\left(\mathbf{r}_{12}\right) / \lambda\right)\right]$ where $r_{j}\left(\mathbf{r}_{12}\right) \equiv\left|\mathbf{R}_{j}-\left(\mathbf{r}_{1}+\mathbf{r}_{2}\right) / 2\right|$ is the distance between the vortex $j$ and the midpoint of the bond from $\mathbf{r}_{1}$ to $\mathbf{r}_{2}$ and $\theta_{j}\left(\mathbf{r}_{1,2}\right)$ is the angle between the connection lines $\mathbf{r}_{1,2}-\mathbf{R}_{j}$ measured at the vortex position. ${ }^{[1]}$

The MVM wavefunction for a single vortex in the continuum model reads ${ }^{36 \mid 37}$
$\Psi(r, \phi) \propto \exp \left[-\zeta^{-1} \int_{0}^{r} \mathrm{~d} p f(p)\right]\left(\begin{array}{c}e^{-i \pi / 4} J_{0}\left(r k_{F}\right) \\ e^{+i \pi / 4+i \phi} J_{1}\left(r k_{F}\right) \\ e^{-i \pi / 4-i \phi} J_{1}\left(r k_{F}\right) \\ -e^{+i \pi / 4} J_{0}\left(r k_{F}\right)\end{array}\right)$
where $J_{l}(x)$ is the Bessel function of the first kind and the decay in radial direction is governed by the Majorana coherence length is $\zeta=\hbar v / \Delta_{0}$. Here, the effect of the vector potential $\mathbf{A}(\mathbf{r})$ has been neglected as justified for a single vortex if $\lambda \gg \zeta$.

|  | $\hbar v$ | $\mu$ | $\Delta_{0}$ | $k_{F}=\mu / \hbar v$ |
| :---: | :---: | :---: | :---: | :---: |
| lat. model | 1 | 0.6 | 0.2 | 1.66 |
| $\mathrm{FeTe}_{\mathrm{x}} \mathrm{Se}_{1-\mathrm{x}}$ | $25 m e \mathrm{~V} \cdot \mathrm{~nm}$ | $5 m e \mathrm{~V}$ | $1.8 m e \mathrm{~V}$ | $0.2 / \mathrm{nm}$ |
|  | $\xi$ | $\zeta=\hbar v / \Delta_{0}$ | $\lambda$ |  |
| lat. model | 2 | 5 | 30 |  |
| $\mathrm{FeTe}_{\mathrm{x}} \mathrm{Se}_{1-\mathrm{x}}$ | 4.6 nm | 13.9 nm | 500 nm |  |

Table I. Summary of parameters used for the lattice model simulations and for the experimentally realized material $\mathrm{FeTe}_{\mathrm{x}} \mathrm{Se}_{1-\mathrm{x}}, x \simeq 0.55$ as compiled in Ref. 41. Here, $v$ and $\mu$ are the velocity and chemical potential of the Dirac surface Hamiltonian, respectively. For the lattice model, we set $\hbar v=1$ and $a=1$ for the lattice constant. The surface state pairing amplitude without vortices is given by $\Delta_{0}$ while $\xi$ denotes the length-scale on which the pairing decays towards vortex cores. The superconducting coherence length is $\zeta$ and the London penetration length is denoted by $\lambda$.

We choose the lattice model parameters as $\mu=0.6$, $\Delta_{0}=0.2, \xi=2, \lambda=30$, the unit of energy is given by $\hbar v / a=1$ and the unit of length is $a=1$. As summarized in Tab. I this choice of parameters is motivated by comparison to the experimentally extracted values for $\mathrm{FeTe}_{\mathrm{x}} \mathrm{Se}_{1-\mathrm{x}}$, which are of similar relative size. Only the London penetration length $\lambda$ of the lattice model, while still being by far the largest length scale, is chosen smaller than what would be appropriate in $\mathrm{FeTe}_{\mathrm{x}} \mathrm{Se}_{1-\mathrm{x}}$ to keep the required lattice sizes tractable. The one-dimensional gapless Majorana mode localized at the open boundaries of the system does not affect the results below due to sufficient distance between vortices and boundary, so that the hybridization between vortex bound states and the edge modes is negligible compared to inter-vortex hybridizations. The LDOS $\rho(\omega)$ (see Eq. (9) below for a definition) of the finite-size lattice model without vortices and averaged in the center region is shown in Fig. 2 and agrees to the expectation from the continuum model. Further, we have checked that the wavefunction obtained numerically for a single vortex zero-mode agrees with the analytic prediction for the MVM in Eq. (8) and that the first excited CdGM-state appears at an energy of order $0.09 \sim \Delta^{2} / \mu$ as predicted by theory. ${ }^{14] 16}$

## III. NON-LOCAL TRANSPORT

We now consider a transport setup and attach an STM tip "T" as well as a ground contact, see Fig. 1 (contact "C" is to be added at a later stage, see below). For concreteness and to set the stage for the lattice model simulations using the KWANT software package, ${ }^{[42}$ we model the tip " T " as a one-dimensional chain of single sites with hopping $t=1$ diagonal in spin space. This choice will provide a density of states that does not vary appreciably over the small range of bias $|\omega| \ll 1$ applied in the following. The lead is locally coupled to the surface with hopping $\gamma_{T}$ which reflects the tip-sample tunneling matrix


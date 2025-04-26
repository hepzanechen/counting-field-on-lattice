the MVM. The primary platform we consider is the (001) surface with vortices without translation symmetries; in this regard, the Bogoliubov-de Gennes (BdG) Hamiltonian can be rewritten in real space

$$
\begin{align*}
\widehat{H}_{\mathrm{BdG}} & =\sum_{\boldsymbol{r}}\left(\begin{array}{ll}
C_{\boldsymbol{r}}^{\dagger} & C_{\boldsymbol{r}}
\end{array}\right)\left(\begin{array}{cc}
H_{0}(m) & -i \Delta_{0} \tau_{0} \sigma_{y} \\
i \Delta_{0} \tau_{0} \sigma_{y} & -H_{0}^{*}(m)
\end{array}\right)\binom{C_{\boldsymbol{r}}}{C_{r}^{\dagger}} \\
& +\sum_{\boldsymbol{r}, \boldsymbol{\delta}}\left(\begin{array}{ll}
C_{\boldsymbol{r}}^{\dagger} & C_{\boldsymbol{r}}
\end{array}\right)\left(\begin{array}{cc}
H_{n n}(\boldsymbol{\delta}) & 0 \\
0 & -H_{n n}^{*}(\boldsymbol{\delta})
\end{array}\right)\binom{C_{r+\delta}}{C_{\boldsymbol{r}+\boldsymbol{\delta}}^{\dagger}} \tag{A3}
\end{align*}
$$

where the on-site part $H_{0}(m)=v_{F} m \rho_{z} \sigma_{0}-\mu \rho_{o} \sigma_{0}, \boldsymbol{\delta}= \pm a \hat{x}, \pm a \hat{y}, \pm a \hat{z}$ indicates the three directions of the hopping, and the nearest neighbor hopping part $H_{n n}( \pm a \hat{n})=$ $-v_{F} \rho_{z} \sigma_{0} / 2 \pm i v_{F} \rho_{x} \sigma_{n} / 2$ for $n=x, y, z$. To reduce the number of the layers in the $z$ direction for the numerical simulation, we specifically choose $m=2$ so that the surface Dirac cones with small momentum can be almost located on the top and bottom layers. On these layers, the neutral point of the Dirac cone is located at $(0,0)$ in the (001) surface BZ , which is consistent with $\mathrm{FeTe}_{x} \mathrm{Se}_{1-x}$.

Since $\mathrm{FeTe}_{x} \mathrm{Se}_{1-x}$ is a type-II superconductor, multiple Abrikosov vortices can be present with magnetic flux through the superconductor. In the presence of the vortices, the BdG Hamiltonian is in the form of

$$
\begin{align*}
\widehat{H}_{\mathrm{BdG}} & =\sum_{\boldsymbol{r}}\left(\begin{array}{ll}
C_{\boldsymbol{r}}^{\dagger} & C_{\boldsymbol{r}}
\end{array}\right)\left(\begin{array}{cc}
H_{0}(m) & -i \Delta(\boldsymbol{r}) \tau_{0} \sigma_{y} \\
i \Delta^{*}(\boldsymbol{r}) \tau_{0} \sigma_{y} & -H_{0}^{*}(m)
\end{array}\right)\binom{C_{\boldsymbol{r}}}{C_{\boldsymbol{r}}^{\dagger}} \\
& +\sum_{\boldsymbol{r}, \delta}\left(\begin{array}{ll}
C_{\boldsymbol{r}}^{\dagger} & C_{\boldsymbol{r}}
\end{array}\right)\left(\begin{array}{cc}
H_{n n}(\boldsymbol{\delta}) e^{-\frac{i e}{\hbar} \int_{\boldsymbol{r}}^{r+\delta} \boldsymbol{A}(\boldsymbol{r}) \cdot d \boldsymbol{l}} & 0 \\
0 & -H_{n n}^{*}(\boldsymbol{\delta}) e^{\frac{i e}{\hbar} \int_{\boldsymbol{r}}^{r+\boldsymbol{\delta}} \boldsymbol{A}(\boldsymbol{r}) \cdot d \boldsymbol{l}}
\end{array}\right)\binom{C_{\boldsymbol{r}+\boldsymbol{\delta}}}{C_{\boldsymbol{r}+\boldsymbol{\delta}}^{\dagger}} \tag{A4}
\end{align*}
$$

where the superconductor gap $\Delta(\boldsymbol{r})$ has an additional phase stemming from the vortices and the nearest neighbor hopping is modified by the Peierls substitution integral with the vector potential $\boldsymbol{A}(\boldsymbol{r})$ describing the magnetic flux. In order to perform the numerical simulation, it is necessary to know the explicit expressions of $\Delta(\boldsymbol{r})$ and $\boldsymbol{A}(\boldsymbol{r})$.

First, we consider a single vortex located at $\boldsymbol{r}=(\mathrm{x}, \mathrm{y})=(0,0)$ along the $z$ direction and then the superconducting gap is given by

$$
\begin{equation*}
\Delta(r)=\Delta_{0} \tanh \left(r / \xi_{0}\right) e^{i \theta} \tag{A5}
\end{equation*}
$$

where $\xi_{0}$ is the superconducting coherence length. According to the experimental results (refer to Fig. 2(g,h) in (9), the superconducting gap appears roughly 5 nm away from the vortex cores, so $\xi_{0}=4.6 \mathrm{~nm}$ is chosen for the simulation. We note that the Majorana
coherence length $\xi=13.9 \mathrm{~nm}$, which is the length scale of the MVM, and the superconducting coherence length $\xi_{0}$ represent the different physical entities. To obtain the vector potential $\boldsymbol{A}(\boldsymbol{r})$, we start with the magnetic field in the $z$ direction, which concentrates in the vortex core and can be described by the London equation (44)

$$
\begin{equation*}
B_{z}(\boldsymbol{r})-\lambda^{2} \nabla^{2} B_{z}(\boldsymbol{r})=\Phi_{0} \delta(\boldsymbol{r}) \tag{A6}
\end{equation*}
$$

where $\Phi_{0}=h / 2 e$ indicates the $\pi$-flux of the total magnetic flux on the entire surface and $\lambda$ is the London penetration depth. For $\mathrm{FeTe}_{x} \mathrm{Se}_{1-x}$, the London penetration length $\lambda \approx 500 \mathrm{~nm}(27,28)$. Since the London penetration length is much greater than the two characteristic lengths of the $\operatorname{MVM}\left(\xi=13.9 \mathrm{~nm}\right.$ and $\left.\frac{1}{k_{F}}=5 \mathrm{~nm}\right)$, the magnetic field is diluted in the characteristic length scale. It is reasonable to neglect the magnetic flux effect for a single MVM in the continuum model in the manuscript. However, the magnetic flux plays a pivotal role in the physics of multiple MVMs; we still include the magnetic flux effect in the TB model with a single vortex.
![](https://cdn.mathpix.com/cropped/2025_01_04_33447ce3fbdc7a337f5dg-2.jpg?height=1094&width=1527&top_left_y=1101&top_left_x=299)

Fig. S1. The line profile of the Majorana wave function. The spin up (a) and down (b) distributions of the particle part show that the continuum model (3) and the tight binding model (A4) are in great agreement. In the continuum model, by eq. 4, $\left|u_{\uparrow}\right|^{2}=$ $e^{-2 r / \xi} J_{0}^{2}\left(k_{F} r\right)$ and $\left|u_{\downarrow}\right|^{2}=e^{-2 r / \xi} J_{1}^{2}\left(k_{F} r\right)$. For the tight binding model, we numerically find the MVM trapped in the vortex located at the center of the $120 \times 144 \times 10$ (nm) system along the $z$ direction. (c) The line profile in the wide energy range shows the presence of the superconductor gap and the CdGM modes away from zero energy.

By solving the London equation, the magnetic field in the z direction is given by

$$
\begin{equation*}
B_{z}(r)=\frac{\Phi_{0}}{2 \pi \lambda^{2}} K_{0}(r / \lambda) \tag{A7}
\end{equation*}
$$

where $K_{i}(r / \lambda)$ is the modified Bessel function of the second kind. Furthermore, the total magnetic flux through the disk with the center at $(0,0)$ and the radius $r$ has the expression of

$$
\begin{equation*}
\phi(r)=2 \pi \int_{0}^{r} B_{Z}\left(r^{\prime}\right) r^{\prime} d r^{\prime}=\left(1-\frac{r}{\lambda} K_{1}(r / \lambda)\right) \Phi_{0} \tag{A8}
\end{equation*}
$$

As $r \rightarrow \infty$, the asymptotic behavior of the modified Bessel function leads to $\phi \rightarrow \Phi_{0}$ indicating the $\pi$-flux forming the single Abrikosov vortex. By using the relation $\phi(r)=\oint \boldsymbol{A}(\boldsymbol{r}) \cdot d \boldsymbol{l}$ in the closed integral path with the fixed radius r , the vector potential can be explicitly written as

$$
\begin{equation*}
\boldsymbol{A}(\boldsymbol{r})=\Phi_{0}\left(\frac{y}{r^{2}} \hat{x}-\frac{x}{r^{2}} \hat{y}\right)\left(1-\frac{r}{\lambda} K_{1}\left(\frac{r}{\lambda}\right)\right) \tag{A9}
\end{equation*}
$$

Now all of the parameters and functions in the BdG Hamiltonian (A4) with a single vortex are given; the wavefunction of the MVM trapped in the vortex on the surface of $\mathrm{FeTe}_{x} \mathrm{Se}_{1-x}$ can be solved in this TB model. fig. $\mathrm{S} 1(\mathrm{a}, \mathrm{b})$ shows the TB model (A4) and the continuum model (3) are in great agreement although the superconducting gap in the continuum model, which does not vanish near the vortex core (no $\tan \left(r / \xi_{0}\right)$ dependent), is slightly different from the superconducting gap in the tight binding model. We note that the MZM wavefunctions are almost identical in the presence and absence of the magnetic flux since the spatial range $r$ is much less than the London penetration depth (45). Furthermore, the first three lowest energies of the CdGM modes trapped in the vortex in the TB are given by $\pm 0.96, \pm 1.30, \pm 1.43 \mathrm{meV}$ as shown in fig. S 1 (c). The energy levels of the CdGM modes are in order of $\Delta^{2} / \mu$, it is not necessary to have equal energy spacing, when the CdGM energy is close to the superconductor gap.

Since the surface of the type-II superconductor practically hosts more than one Abrikosov vortex, the physics of a single vortex is not enough to capture the mechanism of the topological superconductivity on the entire surface. Therefore, we consider multiple Abrikosov vortices located at $\boldsymbol{r}_{j}=\left(x_{j}, y_{j}\right)$ and then the superconducting gap is related to the locations of the vortices

$$
\begin{equation*}
\Delta(\boldsymbol{r})=\Delta_{0} \Pi_{j} \tanh \left(\frac{\left|\boldsymbol{r}-\boldsymbol{r}_{j}\right|}{\xi_{0}}\right) \frac{x-x_{j}+\left(y-y_{j}\right) i}{\left|\boldsymbol{r}-\boldsymbol{r}_{j}\right|} \tag{A10}
\end{equation*}
$$

The vector potential for the multiple vortices is given by

$$
\begin{equation*}
\boldsymbol{A}(\boldsymbol{r})=\Phi_{0} \sum_{j} \frac{\left(y-y_{j}\right) \widehat{x}+\left(x-x_{j}\right) \widehat{y}}{\left|\boldsymbol{r}-\boldsymbol{r}_{j}\right|}\left(\frac{1}{\left|\boldsymbol{r}-\boldsymbol{r}_{j}\right|}-\frac{1}{\lambda} K_{1}\left(\frac{\left|\boldsymbol{r}-\boldsymbol{r}_{j}\right|}{\lambda}\right)\right) \tag{A11}
\end{equation*}
$$

Thus, the complete BdG Hamiltonian of the TB model for multiple vortices, which are given, can be solved in principle.


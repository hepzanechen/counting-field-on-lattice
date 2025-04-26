![](https://cdn.mathpix.com/cropped/2025_02_22_37845a9e920617d3565cg-1.jpg?height=535&width=1451&top_left_y=166&top_left_x=337)

FIG. S6. The minimum direct gap and minimum IVC strength in the BZ for the toy model. For ETI ( $\hat{C}_{2 z} \hat{\mathcal{T}}_{\text {-symmetric }}$ ) and CTI (no $\hat{C}_{2 z} \hat{\mathcal{T}}^{\text {-symmetry), the minimum IVC strength in the BZ is zero. In this phase diagram, the state is ETI on the }}$ horizontal axis, CTI on the vertical axis, and trivial IVC everywhere else.

## S2.3. Continuum $k \cdot p$ model

In this subsection, we consider a continuum $k \cdot p$ version of the ETI toy model introduced in Sec. III B in the main text. While the $k \cdot p$ model can describe the physics within a local patch of momentum space, it does not capture the constraints imposed by the combination of a finite BZ and associated topological winding of inter-Chern hybridization.

We consider two bands in each valley that are described by $2 \times 2$ matrices $H_{+}(\boldsymbol{k})$ and $H_{-}(\boldsymbol{k})$ in sublattice space. We first take the limit of a continuum $\boldsymbol{k} \cdot \boldsymbol{p}$ expansion about $\boldsymbol{k}=0$, and study the following model

$$
H_{+}(\boldsymbol{k})=\left(\begin{array}{cc}
0 & \left(k-k_{1}\right)\left(k-k_{2}\right)  \tag{5}\\
\left(k-k_{1}\right)^{*}\left(k-k_{2}\right)^{*} & 0
\end{array}\right)
$$

with $k=k_{x}+i k_{y}$, and $H_{-}(\boldsymbol{k})=H_{+}^{*}(-\boldsymbol{k})$. The system satisfies TRS and two-fold rotation which act on valley and sublattice as $\hat{\mathcal{T}}=\tau^{x} \mathcal{K}$ and $\hat{C}_{2 z}=\tau^{x} \sigma^{x}$. As in the two-band LLL model discussed in App. S3, the sublattice basis should be thought of as having Chern number $C=\tau \sigma$. The Dirac points at charge neutrality are located at $\tau \boldsymbol{k}_{1}$ and $\tau \boldsymbol{k}_{2}$ and have opposite winding in the two valleys. Since we will focus on filling $\nu=-1$, we will project the problem onto the lower bands
where

$$
\begin{align*}
\theta(\boldsymbol{k}) & =\arg \left[\left(k-k_{1}\right)\left(k-k_{2}\right)\right]  \tag{8}\\
\varphi(\boldsymbol{k}) & =\arg \left[\left(k+k_{1}\right)\left(k+k_{2}\right)\right] \tag{9}
\end{align*}
$$

As the goal is to induce a $\hat{\mathcal{T}}$ - and $\hat{C}_{2 z}$-symmetric IVC state, we add a uniform mean-field $U(1)_{\mathrm{v}}$-breaking term

$$
\begin{equation*}
H_{\mathrm{TIVC}}=\Delta \tau^{x} \sigma^{x} \tag{10}
\end{equation*}
$$

which could arise from interaction effects. Within the lower band subspace of the two valleys, the Hamiltonian can be parameterized as

$$
\tilde{H}(\boldsymbol{k})=\left(\begin{array}{cc}
\epsilon(\boldsymbol{k})+\delta(\boldsymbol{k}) & \tilde{\Delta}(\boldsymbol{k})  \tag{11}\\
{[\tilde{\Delta}(\boldsymbol{k})]^{*}} & \epsilon(\boldsymbol{k})-\delta(\boldsymbol{k})
\end{array}\right)
$$

with the model-specific quantities

$$
\begin{align*}
\epsilon(\boldsymbol{k}) & =-\frac{\left|\left(k-k_{1}\right)\left(k-k_{2}\right)\right|+\left|\left(k+k_{1}\right)\left(k+k_{2}\right)\right|}{2}  \tag{12}\\
\delta(\boldsymbol{k}) & =\frac{\left|\left(k+k_{1}\right)\left(k+k_{2}\right)\right|-\left|\left(k-k_{1}\right)\left(k-k_{2}\right)\right|}{2} \tag{13}
\end{align*}
$$

![](https://cdn.mathpix.com/cropped/2025_02_22_37845a9e920617d3565cg-2.jpg?height=964&width=1072&top_left_y=163&top_left_x=529)

FIG. S7. Toy $\boldsymbol{k} \cdot \boldsymbol{p}$ model for ETI. The model (see Eq. 5 for definition in the absence of IVC) is unbounded for large momentum. Any boost $\boldsymbol{q}$ has been absorbed into the momentum origin such that TRS takes $\boldsymbol{k} \rightarrow-\boldsymbol{k}$. Dirac points in valley $\tau=+$ are at $\boldsymbol{k}_{1}=(1.25,0.25)$ [black triangle] and $\boldsymbol{k}_{2}=(1.00,0.00)$ [grey triangle]. TRS-related non-interacting Dirac points in valley $\tau=-$ are indicated with dots. a) Difference in lower band valley dispersion $\delta(\boldsymbol{k})$. b,c,d) Projected IVC matrix element $\tilde{\Delta}(\boldsymbol{k})$ for a TIVC term of strength $\Delta=1$ and a $\tau^{x}$ perturbation with momentum-dependent strength $A_{\tau^{x}} e^{-|k|^{2} / 2}$. b, c,d) correspond to $A_{\tau^{x}}=0,1,2$ respectively. Note that the lower limit of the color scale has been clamped. Red stars indicate residual Dirac points in the projected Hamiltonian. The residual Dirac points have annihilated in d), leading to the formation of an ETI.

For the uniform TIVC mass of, we have

$$
\begin{equation*}
\tilde{\Delta}(\boldsymbol{k})=-\frac{\Delta}{2}\left(e^{i \theta(\boldsymbol{k})}+e^{i \varphi(\boldsymbol{k})}\right) \tag{14}
\end{equation*}
$$

In order to open a direct gap at $\nu=-1$ corresponding to half-filling of $\tilde{H}(\boldsymbol{k}), \sqrt{|\tilde{\Delta}(\boldsymbol{k})|^{2}+\delta(\boldsymbol{k})^{2}}$ must be nonvanishing for all momenta $\boldsymbol{k}$. In Fig. S7a, we show a representative plot of the valley dispersion difference $\delta(\boldsymbol{k})$. This takes values of opposite signs near the non-interacting Dirac points of the two valleys, since the non-interacting Dirac points are associated with high-energy features of the lower bands. In between, there is a line of $\delta(\boldsymbol{k})=0$, which crosses the time-reversal invariant momentum $\boldsymbol{k}=0$ due to TRS. In Fig. S7b, we plot the off-diagonal term $|\tilde{\Delta}(\boldsymbol{k})|$, which exhibits two nodal lines that connect non-interacting Dirac points in opposite valleys at $\boldsymbol{k}_{1}$ and $-\boldsymbol{k}_{2}$, as well as $\boldsymbol{k}_{2}$ and $-\boldsymbol{k}_{1}$. There are two residual Dirac points (red stars) where $\delta(\boldsymbol{k})=\tilde{\Delta}(\boldsymbol{k})=0$ with opposite winding number. Generally for any $\boldsymbol{k}_{1}, \boldsymbol{k}_{2}$, there will be residual Dirac points since the nodal lines connect non-interacting Dirac points in opposite valleys, and hence intersect the locus of $\delta(\boldsymbol{k})=0$.

To bring the residual Dirac points together and annihilate them, we project an additional IVC term ${ }^{27} \sim \tau^{x}$ to $\tilde{H}(\boldsymbol{k})$, which preserves the $\hat{C}_{2 z}$ and $\hat{\mathcal{T}}$ symmetries. We consider a momentum-dependent perturbation localized near $\boldsymbol{k}=0$, which avoids issues with global obstructions arising from the fact that $\tau^{x}$ couples sublattice bands with opposite Chern number. As shown in Figs. S7c,d, a sufficiently strong $\tau^{x}$ perturbation 'rewires' the nodal lines of the projected IVC matrix element $|\tilde{\Delta}(\boldsymbol{k})|$, so that they connect the non-interacting Dirac points within each valley. As a result, the residual Dirac points annihilate and a direct gap opens at $\nu=-1$, leading to an ETI.

[^0]
## S3. TWO-BAND LLL MODEL

In this section, we present the two-band LLL model, which is an explicit $\hat{C}_{2 z} \hat{\mathcal{T}}$-symmetric interacting LLL-based model whose valley-resolved Hilbert space has a non-trival Euler class $\left|e_{2}\right|=1$. This generalizes the LLL model introduced in Sec. II, which only has one band per valley. In addition to the valley index $\tau= \pm$, the two-band LLL model also has a sublattice degree of freedom with $\sigma=+$ corresponding to sublattice $A$, and $\sigma=-$ corresponding to sublattice $B$. The sublattice and valley-dependent magnetic field is $\boldsymbol{B}=-\tau \sigma B \hat{z}$ leading to Chern numbers $C=\tau \sigma$. The action of two-fold rotation and time-reversal symmetry on the magnetic Bloch creation operators is

$$
\begin{equation*}
\hat{C}_{2 z} d_{\boldsymbol{k}, \tau \sigma}^{\dagger} \hat{C}_{2 z}^{-1}=d_{-\boldsymbol{k}, \bar{\tau} \bar{\sigma}}^{\dagger}, \quad \hat{\mathcal{T}} d_{\boldsymbol{k}, \tau \sigma}^{\dagger} \hat{\mathcal{T}}^{-1}=d_{-\boldsymbol{k}, \bar{\tau} \sigma}^{\dagger}, \tag{1}
\end{equation*}
$$

where $\bar{\sigma}=-\sigma . \hat{C}_{2 z} \hat{\mathcal{T}}$ maps the Chern bands onto each other within a given valley, demonstrating the non-trivial Euler class.

## S3.1. Interaction term

We take the interaction to be density-density in sublattice and valley space

$$
\begin{equation*}
\hat{H}_{\mathrm{int}}=\frac{1}{2 A} \sum_{\boldsymbol{q} \in \mathrm{all}, \tau \tau^{\prime} \sigma \sigma^{\prime}} \tilde{U}_{\tau \sigma ; \tau^{\prime} \sigma^{\prime}}(\boldsymbol{q}) \rho_{\tau \sigma}(\boldsymbol{q}) \rho_{\tau^{\prime} \sigma^{\prime}}(-\boldsymbol{q}) \tag{2}
\end{equation*}
$$

where we have allowed for interaction anisotropies to influence the competition between various phases. For example, having a weaker inter-sublattice versus intra-sublattice interaction would tend to disfavor sublattice-polarized phases. We follow a similar prescription as the LLL model in Sec. II B, and consider a dual gate screened interaction $u_{0}(\boldsymbol{q})=$ $2 \pi U \frac{\tanh q d_{\mathrm{sc}}}{q}$ with effective dielectric constants depending on the valley-sublattice flavor. In particular, we have

$$
\begin{equation*}
\tilde{U}_{\tau \sigma ; \tau^{\prime} \sigma^{\prime}}(\boldsymbol{q})=\left[1-\left(u_{\mathrm{v}}-1\right) \delta_{\tau, \tau^{\prime}}\right]\left[1-\left(u_{\mathrm{sub}}-1\right) \delta_{\sigma, \sigma^{\prime}}\right] u_{0}(\boldsymbol{q}) \tag{3}
\end{equation*}
$$

with anisotropy factors $u_{\mathrm{v}}, u_{\text {sub }}$, where $u_{\mathrm{v}}<1\left(u_{\mathrm{v}}>1\right)$ corresponds to intravalley interactions being stronger (weaker) than intervalley interactions. For a completely isotropic interaction potential $u_{\mathrm{v}}=u_{\text {sub }}=1, \hat{H}_{\text {int }}$ has a $U(2) \times U(2)$ symmetry corresponding to rotations within each pair of bands with identical Chern numbers $C=\tau \sigma$ (i.e. a Chern sector). In other words, the form factors themselves satisfy $U(2) \times U(2)$.

## S3.2. Kinetic term

We now turn to the valley-diagonal kinetic term. Since we have single-valley $\hat{C}_{2 z} \hat{\mathcal{T}}$ symmetry and opposite Chern numbers in the two sublattices, we expect that the resulting band structure in each valley will have Dirac nodes with a net $4 \pi$ winding, as is the case for TBG. This $4 \pi$ winding encodes the non-trivial Euler index $\left|e_{2}\right|=1$ carried by the two bands within a given valley. In fact, we have the tools at our disposal to explicitly construct a Hamiltonian that possesses these topological features - we can use the EVL order parameter $\Delta_{\mathrm{EVL}}(\boldsymbol{k})$ from Sec. II D, which motivates the following single-particle Hamiltonian

$$
\begin{align*}
\hat{H}^{\mathrm{SP}}=\sum_{\boldsymbol{k} \in \mathrm{BZ}} & {\left[\Delta_{\mathrm{EVL}}^{*}(\boldsymbol{k}) d_{\boldsymbol{k},+A}^{\dagger} d_{\boldsymbol{k},+B}+\Delta_{\mathrm{EVL}}(\boldsymbol{k}) d_{\boldsymbol{k},+B}^{\dagger} d_{\boldsymbol{k},+A}\right.}  \tag{4}\\
& \left.+\Delta_{\mathrm{EVL}}^{*}(\boldsymbol{k}) d_{\boldsymbol{k},-B}^{\dagger} d_{\boldsymbol{k},-A}+\Delta_{\mathrm{EVL}}(\boldsymbol{k}) d_{\boldsymbol{k},-A}^{\dagger} d_{\boldsymbol{k},-B}\right]
\end{align*}
$$

which preserves $\hat{C}_{2 z}$ and $\hat{\mathcal{T}}$. The dispersion is plotted in Fig. S9a with $\boldsymbol{q}=0$ and $s=-1$ in the EVL order parameter (Eq. 18). There are two Dirac points in each valley with identical winding, which are maximally separated in momentum space.

We note that the kinetic term $\hat{H}^{\mathrm{SP}}$ (Eq. 4) satisfies a $U(2)$ subgroup of the $U(2) \times U(2)$ form factor symmetry, corresponding to identical rotations in the two Chern sectors.
![](https://cdn.mathpix.com/cropped/2025_02_22_37845a9e920617d3565cg-4.jpg?height=814&width=987&top_left_y=187&top_left_x=558)

FIG. S8. HF phase diagram of the two-band LLL model at $\nu=-1 . u_{\mathrm{v}}$ is the valley interaction anisotropy factor and $U$ controls the scale of the interaction (see Eq. 3). Color indicates the magnitude of intervalley coherence (IVC). White lines indicate approximate phase boundaries. The valley boost is fixed to either $\boldsymbol{q}=(0,0)$ or $(Q / 2, Q / 2)$. The gate distance $d_{\mathrm{sc}}=6 a$, sublattice interaction anisotropy $u_{\text {sub }}=1$, and system size is $16 \times 16$.

## S3.3. Phase diagram at $\nu=-1$

Fig. S8 shows the phase diagram of the two-band LLL model at $\nu=-1$ as a function of the valley interaction anisotropy $u_{\mathrm{v}}$ and interaction scale $U$. The interactions have been chosen to be isotropic in sublattice space $u_{\text {sub }}=1$. For strong interactions, we find a $|C|=1$ Chern insulator phase. For large $u_{\mathrm{v}}$, this is simply a valley- and sublatticepolarized insulator. For small $u_{\mathrm{v}}$, the Chern insulator becomes intervalley coherent. This IVC is not obstructed, as the IVC is predominaintly between bands of the same Chern number $C=\tau \sigma$. For intermediate $U \sim 1$, we find a time-reversal symmetric IVC phase. However, we find that $\hat{C}_{2 z}$ is broken (see Fig. S9b for an example). Furthermore, inspection of the valley-filtered basis reveals that the intervalley coherence is not frustrated. This will be explained in more detail in the next subsection.

We comment briefly on the phase diagram at $\nu=0$. Because the non-interacting band structure consists of just Dirac points at $E_{F}$, there is no 'lobe principle' that would motivate kinetically-driven IVC. Indeed, we do not find any appreciable regions of such phases in the phase diagram.

## S3.4. IVC states at $\nu=-1$

In the non-interacting problem at $\nu=-1, E_{F}$ lies within the valence bands, which possess two high-energy peaks (i.e. the Dirac points) at $\boldsymbol{k}= \pm(Q / 4, Q / 4)$ and two low-energy troughs at $\boldsymbol{k}= \pm(Q / 4,3 Q / 4)$ in each valley [see Fig. S9a]. This suggests a 'lobe' construction that boosts valley $\tau=-$ by $\boldsymbol{q}=(Q / 2,0)$, and results in four pairs of coincident peak/trough features. Note that this is in contrast to the lobe construction for the IKS in TBG, where there is a single high-energy region and low-energy trough in each valley, leading to two pairs of coincident peak/trough features after boosting [18]. For weak interactions, the large non-interacting Fermi surfaces and lack of nesting will lead to a gapless state. For sufficiently strong interactions, the system is expected to simply polarize into flavor and sublattice space and form a 'strong-coupling' Chern insulator. For intermediate interaction strengths though, we anticipate that the system will exploit the above 'lobe' construction and form a gapped IVC spiral state.

As shown in Fig. S9b for $U=1$, we indeed find a gapped IVC phase at $\nu=-1$ with $\boldsymbol{q}=(Q / 2,0)$ that preserves TRS. The occupied HF band has dominant weight on the single-particle valence bands, and modulates its valley pseudospin to adapt to the kinetic energy. However, while there is no net sublattice polarization, the system breaks $\hat{C}_{2 z}$, as can be deduced by the momentum-resolved sublattice polarization which has small opposite values around
![](https://cdn.mathpix.com/cropped/2025_02_22_37845a9e920617d3565cg-5.jpg?height=383&width=1600&top_left_y=166&top_left_x=271)

FIG. S9. Two-band LLL model and intervalley-coherent states. a) Non-interacting valence band dispersion in valley $\tau=+$. The single-particle band structure is constructed from Eq. 4 using $\boldsymbol{q}=0$ and $s=-1$ in the EVL ansatz of Eq. 18. The conduction band dispersion can be obtained using particle-hole symmetry, and there are Dirac points at $\boldsymbol{k}= \pm(Q / 4, Q / 4)$. The dispersion in valley $\tau=-$ is identical. b) Properties of a $\hat{C}_{2 z}$-broken IVC state at $\nu=-1$ with $\boldsymbol{q}=(Q / 2,0)$. Left: Difference between the energy of the second-lowest HF band $E_{1}(\boldsymbol{k})$ and the lowest HF band $E_{0}(\boldsymbol{k})$. The latter is fully occupied. Right: Momentum-resolved sublattice polarization of the HF state in valley $\tau=+$. Parameters are $U=u_{\mathrm{v}}=u_{\mathrm{sub}}=1, d_{\mathrm{sc}}=6 a$, and system size is $24 \times 24$. c) Same as b) left, except where $\hat{C}_{2 z}$ has been enforced in the HF calculation. Red circles indicate residual Dirac points.
the positions of the non-interacting Dirac points. There is also no indication of any topological frustration, since the IVC does not vanish anywhere in the BZ, unlike for the $\mathrm{CTI}_{n}$.

Another way to see the lack of frustration is to consider 'unfolding' the single occupied intervalley-coherent HF band

$$
\begin{equation*}
|\mathrm{HF}, \boldsymbol{k}\rangle=\alpha(\boldsymbol{k})\left|\mathrm{HF}_{+}, \boldsymbol{k}\right\rangle+\beta(\boldsymbol{k})\left|\mathrm{HF}_{-}, \boldsymbol{k}\right\rangle \tag{5}
\end{equation*}
$$

which implicitly defines a pair of time-reversal-related valley-diagonal bands $\left\{\left|\mathrm{HF}_{+}, \boldsymbol{k}\right\rangle\right\}$ and $\left\{\left|\mathrm{HF}_{-}, \boldsymbol{k}\right\rangle\right\}$, dubbed the valley-filtered bands. In Fig. S9b, we find these valley-filtered bands $\left\{\left|\mathrm{HF}_{\tau}, \boldsymbol{k}\right\rangle\right\}$ are topologically trivial with $C=0$. Since there is no topological obstruction to hybridization between $C=0$ bands, the IVC is therefore of a 'trivial' nature.

As argued in the main text, if the system was gapped and preserved $\hat{C}_{2 z}$ and $\hat{\mathcal{T}}$, the resulting state would be an ETI. The interpretation of the 'trivial' IVC state in Fig. S9b is therefore that the system removes the topological obstruction by spontaneously breaking $\hat{C}_{2 z}$ and sublattice-polarizing locally in momentum space. Unlike in the oneband LLL model, the presence of multiple bands within each valley allows for such 'unfrustration' of the IVC order parameter. We note that an alternative $\hat{C}_{2 z}$-breaking scenario to Fig. S9b consists of inducing identical sublattice masses at the two Dirac points for each valley. The resulting IVC state would instead be a $\mathrm{CTI}_{1}$ since the unfolded bands have $|C|=1$. This is energetically disfavored because it does not resolve the topological obstruction to IVC.

In Fig. S9c, we attempt to generate an ETI by enforcing $\hat{C}_{2 z}$ in the HF calculation. However, the resulting state remains gapless with two residual Dirac points at $E_{F}$ that are not at any of the non-interacting Dirac point positions. This means that while the two-band LLL model has the same topological features as the strained TBG Hamiltonian, the non-topological details prevent stabilization of an ETI. One key difference between the two models is the positions of the (renormalized) Dirac points in the BZ before IVC is induced. In TBG, due to the combination of the intrinsic nematic instability and the effect of external uniaxial heterostrain, the Dirac points migrate towards $\Gamma_{M}$ and become close to each other within each valley. Hence, the sign-changing sublattice polarization involved in the trivial IVC of Fig. S9b is unlikely, since the rapidly changing sublattice order would lead to a large exchange penalty, and the result is a topologically frustrated IKS. On the other hand, the Dirac points in the two-band LLL model are spaced as far as possible in the BZ, and there is no analogous mechanism that brings them together. Therefore, the system is more susceptible to the sublattice texturing shown in Fig. S9 that alleviates the topological frustration to IVC. Furthermore, even when $\hat{C}_{2 z}$ is imposed, there are still residual Dirac points which remain far apart.

## S4. PERTURBATIVE DERIVATION OF VALLEY/GRAVITATIONAL CHERN-SIMONS TERM

Consider a single massive Dirac fermion coupled to the $\omega$ (valley) gauge field defined in the main text (Sec. IV). Its action is given by

$$
\begin{equation*}
\mathcal{L}=\psi^{\dagger}\left(i \partial_{t}-\omega_{0} \tau^{z}\right) \psi-\psi^{\dagger} e_{m}^{n} \tau^{m}\left(i \partial_{n}-\omega_{n} \tau^{z}\right) \psi-M \psi^{\dagger} \tau^{z} \psi \tag{1}
\end{equation*}
$$


[^0]:    ${ }^{27}$ More generally, the total set of possible intervalley terms that preserve $\hat{C}_{2 z}$ and $\hat{\mathcal{T}}$ are $\tau_{x} \sigma_{x}, \tau_{x}, \tau_{y} \sigma_{z}$ for momentum-even functions and $\tau_{x} \sigma_{y}$ for momentum-odd functions. Out of these, only $\tau_{x} \sigma_{x}$ and $\tau_{x} \sigma_{y}$ commute with $C=\tau_{z} \sigma_{z}$ and hence hybridize bands within the Chern sectors.


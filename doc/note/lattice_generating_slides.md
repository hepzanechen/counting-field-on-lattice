---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-size: 28px;
  }
  h1 {
    font-size: 42px;
    color: #234E70;
  }
  h2 {
    font-size: 36px;
    color: #234E70;
  }
  img {
    display: block;
    margin: auto;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .small-text {
    font-size: 24px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
  }
  .grid-2x2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(2, auto);
    gap: 0.5rem;
  }
  .full-img {
    width: 90%;
    max-height: 75vh;
  }
  .max-img {
    width: 90%;
    max-height: 80vh;
    object-fit: contain;
  }
---

<!-- _class: lead -->
# Counting Fields in Tight Binding Lattice
## Efficient Transport Calculations Beyond Green's Functions

**Zhen Chen**

---

## Motivation

- **Problem**: Transport calculations in mesoscopic systems
  - Green's functions → good for basic quantities (DOS, transmission)
  - BUT cumbersome for higher-order correlations

- **Our Solution**: Counting Field approach offers:
  - Unified framework for transport statistics of all orders
  - Comparable computational efficiency to standard methods
  - Works for both normal and superconducting systems

---

## Keldysh Path Integral Contour

![width:75%](figs/contour2.pdf)

- Forward (upper) branch: time evolution
- Backward (lower) branch: return to initial state
- Non-equilibrium quantum systems require this closed-time contour
- Different counting field phases on each branch

---

## Path Integral on Keldysh Contour

<div class="columns">
<div>

![width:90%](contour2-eps-converted-to.pdf)

- Forward and backward branches required for non-equilibrium problems
- Time evolution on the contour preserves quantum correlations
- Counting fields act differently on each branch

</div>
<div>

- The counting field tracks particle transfers
- Key idea: λ modifies hopping amplitudes
- Different phases on forward/backward branches
- λ serves as source field for generating function

</div>
</div>

---

## Core Idea: The Counting Field Method

- Insert counting fields into hopping terms of tight-binding Hamiltonian
- Modify the action on Keldysh contour:

$$S_{0}=\sum_{mn}\left\{ \int dt\bar{\phi}_{m}^{+}(t)\left[\delta_{mn}i\partial_{t}\phi_{n}^{+}(t)-{\color{teal}e^{i\frac{\lambda_{mn}(t)}{2}}}\,t_{mn}\phi_{n}^{+}(t)\right]-\int dt\bar{\phi}_{m}^{-}(t)\left[\delta_{mn}i\partial_{t}\phi_{n}^{-}(t)-{\color{teal}e^{-i\frac{\lambda_{mn}(t)}{2}}}t_{mn}\phi_{n}^{-}(t)\right]\right\}$$

- Here, $\phi^+$ and $\phi^-$ are fields on forward and backward branches
- $\lambda_{mn}(t)$ counts particles hopping from site $n$ to site $m$
- Antisymmetry: $\lambda_{mn}(t) = -\lambda_{nm}(t)$

---

## Transport Observables from Derivatives

- Current = 1st-order derivative
$$\left\langle I_{mn}(t)\right\rangle = \dfrac{\delta\ln Z}{\delta(i\lambda(t))}\bigg|_{\lambda=0}$$

- Noise = 2nd-order derivative
$$\left\langle I_{mn}(t)I_{mn}(t^{\prime})\right\rangle = \dfrac{\delta^{2}\ln Z}{\delta(i\lambda(t))\delta(i\lambda(t^{\prime}))}\bigg|_{\lambda=0}$$

- Higher-order cumulants = Higher-order derivatives
  - 3rd order: Skewness (asymmetry in fluctuations)
  - 4th order: Kurtosis (extent of the tails)
  - All calculated through same computational framework

---

## Matrix Formulation

- Generating functional: $Z(\boldsymbol{\lambda},\omega) = \mathcal{N} \det\mathcal{M}(\boldsymbol{\lambda},\omega)$
  
- The $\mathcal{M}$ matrix combines the central region and leads:

$$\mathcal{M}(\boldsymbol{\lambda})=\begin{pmatrix}
\boldsymbol{g_{C}^{-1}} & \boldsymbol{-t_{CL1}\Lambda_{1}(\lambda_{1})} & \cdots & \boldsymbol{-t_{CLN}\Lambda_{N}(\lambda_{N})}\\
\boldsymbol{-\Lambda_{1}^{\dagger}t_{CL1}^{\dagger}} & \boldsymbol{g_{Lead1}^{-1}} & & \\
\vdots & & \ddots & \\
\boldsymbol{-\Lambda_{N}^{\dagger}t_{CLN}^{\dagger}} & & & \boldsymbol{g_{LeadN}^{-1}}
\end{pmatrix}$$

- Counting field enters via $\Lambda_i(\lambda_i)=\begin{pmatrix} \cos(\lambda_i/2) & -i\sin(\lambda_i/2) \\ -i\sin(\lambda_i/2) & \cos(\lambda_i/2) \end{pmatrix}$

---

## Keldysh Formalism: RAK Space (1/2)

- Each matrix block in $\mathcal{M}$ has $2×2$ internal structure in RAK space:

$$\boldsymbol{g_{c0}^{RAK}}=\left(\begin{array}{cc}
\left(g_{C}^{r}\right)^{-1} & \left(g_{C}^{r}\right)^{-1}g_{C}^{k}\left(g_{C}^{a}\right)^{-1}\\
0 & \left(g_{C}^{a}\right)^{-1}
\end{array}\right)$$

- Hopping matrices structure in RAK space:

$$\boldsymbol{t_{CL}}=\left(\begin{array}{cc}
t_{CL} & 0\\
0 & t_{CL}
\end{array}\right)$$

---

## Keldysh Formalism: RAK Space (2/2)

- For a single lead, $\mathcal{M}(\lambda,\omega)$ expands to:

$$\mathcal{M}(\lambda)=\begin{pmatrix}
(g_{C}^{r})^{-1} & (g_{C}^{r})^{-1}g_{C}^{k}(g_{C}^{a})^{-1} & t_{LC}\cos(\frac{\lambda}{2}) & -it_{LC}\sin(\frac{\lambda}{2}) \\
0 & (g_{C}^{a})^{-1} & -it_{LC}\sin(\frac{\lambda}{2}) & t_{LC}\cos(\frac{\lambda}{2}) \\
t_{CL}^{*}\cos(\frac{\lambda}{2}) & it_{LC}^{*}\sin(\frac{\lambda}{2}) & (g_{Lead}^{r})^{-1} & (g_{Lead}^{r})^{-1}g_{Lead}^{k}(g_{Lead}^{a})^{-1} \\
it_{LC}^{*}\sin(\frac{\lambda}{2}) & t_{CL}^{*}\cos(\frac{\lambda}{2}) & 0 & (g_{Lead}^{a})^{-1}
\end{pmatrix}$$

- Simplified inverse Green's functions:
$$(g_{C}^{r,a})^{-1} = \omega - H_{C} \pm i\eta$$

---

## PyTorch Implementation

```
Counting Field Method with PyTorch Autograd:

1. Initialize λ with requires_grad_(True) to enable auto-differentiation

2. Build the matrix M(λ) incorporating counting fields:
   - Central region + leads block diagonal structure
   - Hopping terms modified by counting field matrices

3. Compute ln(det(M)) with PyTorch's logdet operation

4. Obtain currents, noise, and higher-order correlations:
   - First derivative: jacrev(Im[Z]) → Current
   - Second derivative: jacrev(jacrev(Im[Z])) → Noise
   - Higher derivatives: Recursively apply jacrev → Higher cumulants
```

- **Key advantage**: Derivatives computed automatically and efficiently
- Time complexity comparable to standard Green's function methods

---

## Analytical Derivation: Current Formula (1/2)

Starting with the logarithm of the generating functional:

$$\ln Z(\lambda_{1},\omega) = \ln\det\mathcal{M}(\lambda_{1},\omega)$$

Using block determinant identity and expanding:

$$\ln Z(\lambda_{1},\omega) = \ln\det(g_{Lead}^{-1}) + \ln\det(g_{C}^{-1} - \Sigma(\lambda_{1}))$$

Where $\Sigma(\lambda_{1}) = t_{CL}\Lambda(\lambda_{1})g_{Lead}^{-1}\Lambda^{\dagger}(\lambda_{1})t_{CL}^{\dagger}$

Taking derivative at $\lambda_{1}=0$:

$$\frac{\partial\ln Z}{\partial\lambda_{1}}\bigg|_{\lambda_{1}=0} = \frac{\partial}{\partial\lambda_{1}}\ln\det(G_{C}^{-1} - \Sigma(\lambda_{1}) + \Sigma(0) - \Sigma(0))\bigg|_{\lambda_{1}=0}$$

---

## Analytical Derivation: Current Formula (2/2)

Simplifying further:

$$\frac{\partial\ln Z}{\partial\lambda_{1}}\bigg|_{\lambda_{1}=0} = \frac{\partial}{\partial\lambda_{1}}\ln\det(\mathbb{I} + G_{C}(\Sigma(0) - \Sigma(\lambda_{1})))\bigg|_{\lambda_{1}=0}$$

$$= \frac{\partial}{\partial\lambda_{1}}\text{Tr}\ln(\mathbb{I} + G_{C}(\Sigma(0) - \Sigma(\lambda_{1})))\bigg|_{\lambda_{1}=0}$$

$$= \text{Tr}\left[G_{C}\frac{\partial}{\partial\lambda_{1}}(\Sigma(0) - \Sigma(\lambda_{1}))\right]\bigg|_{\lambda_{1}=0}$$

$$= \frac{i}{2}\text{Tr}\left[G_{C}(\Sigma(0)\sigma_{x}^{RAK} - \sigma_{x}^{RAK}\Sigma(0))\right]$$

This equals the standard Meir-Wingreen current formula:

$$I = \frac{e}{\hbar}\int\frac{d\omega}{2\pi}\text{Tr}[(G^r-G^a)\Sigma^< + G^<(\Sigma^a-\Sigma^r)]$$

---

## Transport Scattering Theory (1/2)

For systems with superconductivity, the current through lead $i$ is given by:

$$I_{i}=\frac{e}{h}\sum_{\alpha,j\in NS,\beta}\operatorname{sgn}(\alpha)\left[\delta_{ij}\delta_{\alpha\beta}-T_{ij}^{\alpha\beta}(E)\right]f_{j\beta}(E)$$

where:
- $\alpha, \beta \in \{e, h\}$ are electron/hole indices
- $T_{ij}^{\alpha\beta}(E)$ is the transmission probability from channel $\beta$ in lead $j$ to channel $\alpha$ in lead $i$
- $f_{je}(E)=\left[1+\exp\left(\frac{E-\left(\mu_{j}-\mu_{S}\right)}{kT}\right)\right]^{-1}$ is the electron Fermi distribution
- $f_{jh}(E)=\left[1+\exp\left(\frac{E+\left(\mu_{j}-\mu_{S}\right)}{kT}\right)\right]^{-1}$ is the hole Fermi distribution

---

## Transport Scattering Theory (2/2)

Current noise power is given by:

$$\begin{aligned}\mathcal{S}_{ij}=\frac{e^2}{h}\sum_{\alpha,\beta} & \{\delta_{ij}\delta_{\alpha\beta}f_{i\alpha}(E)\left(1-f_{i\alpha}(E)\right)\\
 & -\text{sgn}(\alpha)\text{sgn}(\beta)\left[T_{ji}^{\beta\alpha}f_{i\alpha}(E)\left(1-f_{i\alpha}(E)\right)+T_{ij}^{\alpha\beta}f_{j\beta}(E)\left(1-f_{j\beta}(E)\right)\right]\\
 & +\sum_{k,\gamma,l,\delta}\text{sgn}(\alpha)\text{sgn}(\beta)\left(s_{ik}^{\alpha\gamma*}s_{jk}^{\beta\gamma}f_{k\gamma}(E)\right)\left(s_{il}^{\alpha\delta}s_{jl}^{\beta\delta*}f_{l\delta}(E)\right)
\end{aligned}$$

where $s_{ij}^{\alpha\beta}$ are scattering amplitudes related to transmission probabilities by $T_{ij}^{\alpha\beta} = |s_{ij}^{\alpha\beta}|^2$.

These formulas provide our benchmark for validating the counting field approach.

---

## Results: Consolidated Comparisons

<div class="grid-2x2">
<div>
<p align="center"><b>SSH Chain: Current</b></p>
<img src="figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png" height="200px">
</div>

<div>
<p align="center"><b>Kitaev Chain: Current</b></p>
<img src="figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png" height="200px">
</div>

<div>
<p align="center"><b>SSH Chain: Noise</b></p>
<img src="figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_noise_comparison.png" height="200px">
</div>

<div>
<p align="center"><b>Kitaev Chain: Noise</b></p>
<img src="figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_noise_comparison.png" height="200px">
</div>
</div>

**Key findings**: Perfect agreement between counting field method and Green's function calculation in both systems, even with superconductivity.

---

## SSH Chain: System Details

- Su-Schrieffer-Heeger model:
  - Alternating hopping strengths
  - Hosts topological zero modes
  - Two coupled Jackiw-Rebbi modes

**Parameters**:
- 8 sites, intra-cell hopping $t_u=-10.0$
- Inter-cell hopping $t_v=-5.0$
- Lead coupling $t_{lc}=1.0$

- Current peaks = coupled zero mode resonances
- Split peaks indicate hybridization
- Width reflects coupling to leads

---

## SSH Chain: Current Comparison

![width:75%](figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png)

---

## SSH Chain: Noise Features

- Second-order transport correlations:
  - Reveal quantum transport statistics
  - Show shot noise characteristics
  - Capture zero mode physics

**Key finding**:
- Our method calculates noise with the same accuracy as current
- Computational efficiency is maintained
- Perfect agreement validates approach for higher-order statistics

---

## SSH Chain: Noise Comparison

![width:75%](figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_noise_comparison.png)

---

## Kitaev Chain: System Details

- Kitaev model with superconductivity:
  - p-wave pairing terms
  - Hosts Majorana zero modes (MZM)
  - Includes Andreev reflection processes

**Parameters**:
- 8 sites, hopping $t_c=-50.0$
- Pairing $\Delta=50.0$
- Chemical potential $\mu_c=-50.0$

---

## Kitaev Chain: Current Comparison

![width:75%](figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png)

---

## Kitaev Chain: Noise Features

- Andreev reflection complicates transport:
  - Particle-hole mixing
  - Both electron and hole channels
  - Complex self-energy structure

- Counting field handles all quantum coherent processes:
  - Correctly captures superconducting correlations
  - Maintains computational efficiency
  - Works for all higher-order cumulants

---

## Kitaev Chain: Noise Comparison

![width:75%](figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_noise_comparison.png)

---

## Higher Order Correlations Overview: Onsite Energy Splitting

<div class="grid-2x2">
<div>
<p align="center"><b>1st Order (Current)</b></p>
<img src="figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order1_split_onsite.png" height="200px">
</div>

<div>
<p align="center"><b>2nd Order (Noise)</b></p>
<img src="figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order2_split_onsite.png" height="200px">
</div>

<div>
<p align="center"><b>3rd Order (Skewness)</b></p>
<img src="figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order3_split_onsite.png" height="200px">
</div>

<div>
<p align="center"><b>4th Order (Kurtosis)</b></p>
<img src="figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order4_split_onsite.png" height="200px">
</div>
</div>

**Physical interpretation**: Varying the onsite splitting directly controls zero mode hybridization, creating distinct signatures in higher-order transport statistics that reveal quantum coherent features.

---

## Onsite Splitting: Effects on Transport

- Vary coupling between zero modes:
  - Controls peak splitting in spectrum
  - Modifies transport resonances
  - Affects correlation strength

**Physical significance**:
- Direct control of zero mode hybridization
- Split resonance = finite coupling between edge modes
- Energy separation directly reflects coupling strength
- Higher orders reveal quantum signatures invisible in mean current

---

## Onsite Splitting: 1st Order (Current)

![width:75%](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order1_split_onsite.png)

---

## Onsite Splitting: 2nd Order (Noise)

![width:75%](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order2_split_onsite.png)

---

## Onsite Splitting: 3rd Order (Skewness)

![width:75%](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order3_split_onsite.png)

---

## Onsite Splitting: 4th Order (Kurtosis)

![width:75%](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order4_split_onsite.png)

---

## Higher Order Correlations Overview: Lead-System Coupling

<div class="grid-2x2">
<div>
<p align="center"><b>1st Order (Current)</b></p>
<img src="figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order1_t_lead_central.png" height="200px">
</div>

<div>
<p align="center"><b>2nd Order (Noise)</b></p>
<img src="figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order2_t_lead_central.png" height="200px">
</div>

<div>
<p align="center"><b>3rd Order (Skewness)</b></p>
<img src="figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order3_t_lead_central.png" height="200px">
</div>

<div>
<p align="center"><b>4th Order (Kurtosis)</b></p>
<img src="figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order4_t_lead_central.png" height="200px">
</div>
</div>

**Physical interpretation**: Lead coupling controls peak broadening and transport lifetimes. Higher-order correlations reveal asymmetric and non-Gaussian transport statistics that are inaccessible through conventional methods.

---

## Lead Coupling: Effects on Transport

- Vary hybridization with leads:
  - Controls resonance broadening
  - Modifies lifetime of quasiparticles
  - Changes transport characteristics

**Physical interpretation**:
- Stronger coupling = broader peaks
- Direct measure of lead-induced damping
- Transition from coherent to incoherent transport
- Higher orders show distinctive non-classical features

---

## Lead Coupling: 1st Order (Current)

![width:75%](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order1_t_lead_central.png)

---

## Lead Coupling: 2nd Order (Noise)

![width:75%](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order2_t_lead_central.png)

---

## Lead Coupling: 3rd Order (Skewness)

![width:75%](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order3_t_lead_central.png)

---

## Lead Coupling: 4th Order (Kurtosis)

![width:75%](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order4_t_lead_central.png)

---

## Current Formula Derivation: Key Steps

1. We start with the determinant representation of the generating function:
   $$Z(\boldsymbol{\lambda},\omega) = \mathcal{N} \det\mathcal{M}(\boldsymbol{\lambda},\omega)$$

2. Using the block determinant formula:
   $$\begin{aligned}\det\mathcal{M}(\boldsymbol{\lambda},\omega) & =\prod_{i=1}^{N_{L}}\det\left(g_{Leadi}^{-1}(\omega)\right)\det\left(g_{Central}^{-1}(\omega)-\sum_{i=1}^{N_{L}}t_{CLi}\Lambda_{i}(\lambda_{i},\omega)g_{Leadi}^{-1}(\omega)\Lambda_{i}^{\dagger}(\lambda_{i},\omega)t_{CLi}^{\dagger}\right)\\
 & =\prod_{i=1}^{N_{L}}\det\left(g_{Leadi}^{-1}(\omega)\right)\det\left(g_{Central}^{-1}(\omega)-\sum_{i=1}^{N_{L}}\Sigma_{i}\left(\lambda_{i},\omega\right)\right)
\end{aligned}$$

---

## Current Formula Derivation: Continued

3. For a single lead with counting field $\lambda_1$:
   $$\dfrac{\ln Z(\lambda_{1},\omega)-\ln Z(0,\omega)}{\lambda_{1}} = \dfrac{1}{\lambda_{1}}\ln\det\left(\mathbb{I}+G_{Central}(\omega)\left(\Sigma_{1}\left(0,\omega\right)-\Sigma_{1}\left(\lambda_{1},\omega\right)\right)\right)$$

4. Expanding $\Lambda(\lambda)$ around $\lambda=0$:
   $$\Lambda(\lambda) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} + \lambda \begin{pmatrix} 0 & -i/2 \\ -i/2 & 0 \end{pmatrix} + O(\lambda^2)$$

5. Taking the limit and expanding in Keldysh space:
   $$\lim_{\lambda_{1}\rightarrow 0}\dfrac{\ln Z(\lambda_{1},\omega)-\ln Z(0,\omega)}{\lambda_{1}} = \dfrac{i}{2}\text{Tr}\left[G_{Central}(\omega)\left(\Sigma_{1}\left(0,\omega\right)\sigma_{x}^{RAK}-\sigma_{x}^{RAK}\Sigma_{1}\left(0,\omega\right)\right)\right]$$

---

## Computational Advantages

<div class="columns">
<div>

**Versus Recursive Green's Function**
- Comparable time complexity O(MN³)
- Lower space complexity
- More natural for higher-order statistics
- Unified calculation of all orders

</div>
<div>

**Key Technical Innovations**
- Counting field matrix formulation
- PyTorch automatic differentiation
- Unified determinant calculation
- Higher-order statistics as easy as lower ones

</div>
</div>

**Performance**:
- Time scales with matrix size, not with correlation order
- Higher orders calculated at minimal additional cost
- Perfect for exploring non-Gaussian quantum transport physics

---

## Summary of Key Findings

1. **Unified Framework**: Single generating function determines all transport properties

2. **Method Validation**: Perfect agreement with established techniques for both normal and superconducting systems

3. **Higher-Order Statistics**: Demonstrated practical calculations of correlations up to 4th order

4. **Zero Mode Physics**: Clear signatures in transport correlations

5. **Parameter Control**: Demonstrated systematic tuning of transport properties

---

## Outlook: Future Applications

- **Interacting Systems**: Extending to models with electron-electron interactions

- **Time-Dependent Transport**: Apply to driven and non-equilibrium systems

- **Topological Materials**: Detailed characterization of exotic edge modes

- **Quantum Information Applications**: Studying correlation-based entanglement witnesses

- **Multi-Terminal Devices**: Analyzing complex transport in quantum networks

---

<!-- _class: lead -->
# Thank You!

**Questions?** 
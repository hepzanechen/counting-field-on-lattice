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
---

<!-- _class: lead -->
# Counting Fields in Tight Binding Lattice
## Efficient Transport Calculations Beyond Green's Functions

**Zhen Chen**

---

## Motivation

- **Problem**: Transport calculations in mesoscopic systems
  - Green's functions → good for basic quantities (DOS, transmission)
  - BUT cumbersome for multi-particle correlators

- **Our Solution**: Counting Field approach offers:
  - Unified framework for transport statistics of all orders
  - Comparable computational efficiency to standard methods
  - Works for both normal and superconducting systems

---

## Core Idea: The Counting Field Method

<div class="columns">
<div>

- Insert counting fields into hopping terms of tight-binding Hamiltonian
- Modify the action on Keldysh contour:

$$S_0=\sum_{mn}\int dt\bar{\phi}_{m}^{+}(t)[\delta_{mn}i\partial_t - {\color{red}e^{i\lambda_{mn}/2}}t_{mn}]\phi_{n}^{+}(t) - ...$$

- Derivatives of $\ln Z$ give transport properties

</div>
<div>

- Current = 1st-order derivative
$$\left\langle I_{mn}(t)\right\rangle = \dfrac{\delta\ln Z}{\delta(i\lambda(t))}\bigg|_{\lambda=0}$$

- Noise = 2nd-order derivative
$$\left\langle I_{mn}(t)I_{mn}(t^{\prime})\right\rangle = \dfrac{\delta^{2}\ln Z}{\delta(i\lambda(t))\delta(i\lambda(t^{\prime}))}\bigg|_{\lambda=0}$$

- Higher cumulants = Higher derivatives

</div>
</div>

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

## PyTorch Implementation

```
Counting Field Method with PyTorch Autograd:

1. Initialize λ with requires_grad_(True) to enable auto-differentiation

2. Build the matrix M(λ) incorporating counting fields:
   - Central region + leads block diagonal structure
   - Hopping terms modified by counting field matrices

3. Compute ln(det(M)) with PyTorch's logdet operation

4. Obtain currents, noise, and higher-order correlations:
   - First derivative: jacrev(Re/Im[Z]) → Current
   - Second derivative: jacrev(jacrev(Re/Im[Z])) → Noise
   - Higher derivatives: Recursively apply jacrev → Higher cumulants
```

- **Key advantage**: Derivatives computed automatically and efficiently

---

## Analytical Verification: Current Formula

- We show that our approach reproduces the standard current formula:

<div class="columns">
<div>

**Counting Field Approach**
- Take derivative of ln det[M(λ)]
- Apply block determinant formula
- Take λ → 0 limit

</div>
<div>

**Standard Current Formula**
$$I = \frac{e}{\hbar}\int\frac{d\omega}{2\pi}\text{Tr}[(G^r-G^a)\Sigma^< + G^<(\Sigma^a-\Sigma^r)]$$

- Meir-Wingreen formula
- Exact match with our result

</div>
</div>

---

## System 1: SSH Chain with Zero Modes

<div class="columns">
<div>

- Su-Schrieffer-Heeger model:
  - Alternating hopping strengths
  - Hosts topological zero modes
  - Two coupled Jackiw-Rebbi modes

**Parameters**:
- 8 sites, intra-cell hopping $t_u=-10.0$
- Inter-cell hopping $t_v=-5.0$
- Lead coupling $t_{lc}=1.0$

</div>
<div>

![height:280px](figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png)

- Perfect agreement between methods
- Two peaks = coupled zero modes

</div>
</div>

---

## System 1: SSH Chain - Noise Power

<div class="columns">
<div>

- Second-order transport correlations:
  - Reveal quantum transport statistics
  - Show shot noise characteristics
  - Capture zero mode physics

**Key finding**:
- Our method calculates noise with the same accuracy as current
- Computational efficiency is maintained

</div>
<div>

![height:280px](figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_noise_comparison.png)

- Perfect agreement for noise power
- Complex structure around resonances

</div>
</div>

---

## System 2: Kitaev Chain with Majorana Modes

<div class="columns">
<div>

- Kitaev model with superconductivity:
  - p-wave pairing terms
  - Hosts Majorana zero modes (MZM)
  - Includes Andreev reflection processes

**Parameters**:
- 8 sites, hopping $t_c=-50.0$
- Pairing $\Delta=50.0$
- Chemical potential $\mu_c=-50.0$

</div>
<div>

![height:280px](figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png)

- Agreement maintained with superconductivity
- Two peaks = coupled Majorana modes

</div>
</div>

---

## System 2: Kitaev Chain - Noise Power

<div class="columns">
<div>

- BdG formalism challenges:
  - Particle-hole mixing
  - Andreev reflection processes
  - Both particle and hole channels

**Key finding**:
- Our method handles superconductivity with ease
- All quantum coherent processes captured correctly

</div>
<div>

![height:280px](figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_noise_comparison.png)

- Excellent agreement for noise power
- Complex correlations in superconducting transport

</div>
</div>

---

## Parameter Study 1: Onsite Energy Splitting

<div class="columns">
<div>

- Vary coupling between zero modes:
  - Controls peak splitting in spectrum
  - Modifies transport resonances
  - Affects correlation strength

**Physical interpretation**:
- Direct control of zero mode hybridization
- Split resonance = finite coupling

</div>
<div>

![height:290px](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order1_split_onsite.png)

- Resonance peaks shift apart
- Amplitude changes with splitting
- Clear signature of zero mode physics

</div>
</div>

---

## Parameter Study 1: Higher-Order Correlations

<div class="columns">
<div>

![height:230px](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order2_split_onsite.png)

- Second order (noise)
- Peak structure evolution with splitting

</div>
<div>

![height:230px](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order3_split_onsite.png)

- Third order (skewness)
- Complex correlation pattern
- Unique "fingerprint" of zero mode physics

</div>
</div>

**Key insight**: Higher-order correlations reveal quantum transport features invisible in mean current

---

## Parameter Study 2: Lead-System Coupling

<div class="columns">
<div>

- Vary hybridization with leads:
  - Controls resonance broadening
  - Modifies lifetime of quasiparticles
  - Changes transport characteristics

**Physical interpretation**:
- Stronger coupling = broader peaks
- Direct measure of lead-induced damping

</div>
<div>

![height:290px](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order1_t_lead_central.png)

- Clear broadening of resonances
- Current magnitude changes with coupling
- Transition from sharp to broad features

</div>
</div>

---

## Parameter Study 2: Higher-Order Correlations

<div class="columns">
<div>

![height:230px](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order3_t_lead_central.png)

- Third order with lead coupling
- Asymmetric transport fluctuations
- Complex coupling dependence

</div>
<div>

![height:230px](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order4_t_lead_central.png)

- Fourth order (kurtosis)
- Extreme value statistics of transport
- Quantum coherent features at high orders

</div>
</div>

**Unique capability**: Only the counting field approach makes these calculations practical

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
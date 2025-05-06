---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-size: 30px;
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
---

<!-- _class: lead -->
# Counting Fields in Tight Binding Lattice

**Zhen Chen**

---

## Motivation

- Green's functions are powerful for basic quantities (DOS, transmission)
- But **cumbersome for multi-particle correlators**
- The Counting Field approach offers a significant advantage:
  - Efficient computation of higher-order correlations
  - Unified framework for transport statistics

---

## The Counting Field Method

- Introduce counting fields $\lambda_{mn}(t)$ into the hopping terms
- Calculate the action via determinant operations
- Obtain transport properties through derivatives:

$$\left\langle I_{mn}(t)\right\rangle = \dfrac{\delta\ln Z}{\delta(i\lambda(t))}\bigg|_{\lambda(t)=0}$$

$$\left\langle I_{mn}(t)I_{mn}(t^{\prime})\right\rangle = \dfrac{\delta^{2}\ln Z}{\delta(i\lambda(t))\delta(i\lambda(t^{\prime}))}\bigg|_{\lambda(t)=0}$$

---

## Mathematical Framework

- Generating functional: $Z(\boldsymbol{\lambda},\omega) = \mathcal{N} \det\mathcal{M}(\boldsymbol{\lambda},\omega)$

- Matrix $\mathcal{M}$ captures central region and leads:

$$\mathcal{M}(\boldsymbol{\lambda},\omega)=\left(\begin{array}{cccc}
\boldsymbol{g_{C}^{-1}(\omega)} & \boldsymbol{-t_{CL1}\Lambda_{1}(\lambda_{1})} & \boldsymbol{\cdots} & \boldsymbol{-t_{CLN_{L}}\Lambda_{N_{L}}}\\
\boldsymbol{-\Lambda_{1}^{\dagger}t_{CL1}^{\dagger}} & \boldsymbol{g_{Lead1}^{-1}(\omega)} & \boldsymbol{} & \boldsymbol{}\\
\boldsymbol{\vdots} & \boldsymbol{} & \boldsymbol{\ddots} & \boldsymbol{}\\
\boldsymbol{-\Lambda_{N_{L}}^{\dagger}t_{CLN_{L}}^{\dagger}} & \boldsymbol{} & \boldsymbol{} & \boldsymbol{g_{LeadN_{L}}^{-1}}
\end{array}\right)$$

---

## Numerical Implementation

```
PyTorch Autograd Implementation for Counting Field Calculations

1. Initialize counting field vector λ with requires_grad_(True)

2. Define generating function Z(λ):
   - Construct matrix M(λ) incorporating counting fields
   - Calculate ln(det(M(λ))) using PyTorch's logdet

3. For computing n-th order derivatives:
   - Use PyTorch's jacrev function recursively
   - Extract real/imaginary parts as needed

4. Obtain physical observables:
   - Current = 1st-order derivative
   - Noise = 2nd-order derivative
   - Higher-order cumulants = Higher-order derivatives
```

---

## Analytical Verification

- We proved that our method reproduces the standard current formula:

$$I = \frac{e}{\hbar}\int\frac{d\omega}{2\pi}\text{Tr}\left[\left(G^r(\omega) - G^a(\omega)\right)\Sigma^<(\omega) + G^<(\omega)\left(\Sigma^a(\omega) - \Sigma^r(\omega)\right)\right]$$

- This verifies the mathematical consistency of our approach
- Provides confidence for numerical implementations

---

## Numerical Results: SSH Chain

![height:350px](figs/ssh_chain/compare_generating_directGinv/results_20250506_1429_Nx8_tu-10.0+0.0j_tv-5.0+0.0j_muc0.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png)

- Perfect agreement between counting field (lines) and direct calculations (markers)
- Two peaks represent coupled Jackiw-Rebbi zero modes with finite splitting

---

## Numerical Results: Kitaev Chain

![height:350px](figs/kitaev_chain/compare_generating_directGinv/results_20250506_1413_Nx8_tc-50.0+0.0j_Delta50.0+0.0j_muc-50.0+0.0j_tlc1.0_E-1.00_1.00_pts1000/comparison_plots/all_terminal_currents_comparison.png)

- Agreement maintained even with superconducting correlations
- Captures Andreev reflection processes and Majorana zero modes (MZM)

---

## Parameter Studies: Onsite Energy Splitting

![height:400px](figs/ssh_chain/vary_split_onsite_values/results_20250506_1105/comparison_derivative_order1_split_onsite.png)

- Control of zero mode coupling by varying onsite energy splitting
- Resonance peaks shift apart with increasing splitting

---

## Parameter Studies: Lead-System Coupling

![height:400px](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order1_t_lead_central.png)

- Resonance peak broadening with increasing lead-system coupling
- Reflects enhanced hybridization between leads and zero modes

---

## Higher-Order Correlations

![height:400px](figs/ssh_chain/vary_t_lead_central/results_20250506_1159/comparison_derivative_order3_t_lead_central.png)

- The counting field method excels at computing higher-order statistics
- Reveals quantum coherent nature of transport beyond mean and variance

---

## Key Advantages

1. **Unified framework** for all transport statistics
2. **Computationally efficient** using automatic differentiation
3. **Natural extension** to higher-order correlations
4. Works in both **normal and superconducting** systems
5. **Comparable efficiency** to established recursive Green's function methods

---

## Conclusion and Outlook

- Developed and validated counting field method for tight-binding transport
- Demonstrated perfect agreement with established techniques
- Efficiently computes transport statistics of all orders
- Future work:
  - Interacting systems
  - Time-dependent transport
  - Topological superconducting systems

---

<!-- _class: lead -->
# Thank You!

**Questions?** 
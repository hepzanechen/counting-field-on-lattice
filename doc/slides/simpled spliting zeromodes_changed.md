---
marp: true
theme: default
paginate: true
backgroundColor: #fff
header: "Transport Signatures of Exotic Quantum States"
footer: "© Quantum Transport Research Group"
style: |
  section {
    font-size: 26px;
    text-align: left;
    padding: 40px;
  }
  h1 {
    font-size: 38px;
    color: #00356B;
    border-bottom: 1px solid #ccc;
    padding-bottom: 6px;
  }
  h2 {
    font-size: 34px;
    color: #00356B;
    margin-top: 10px;
  }
  h3 {
    font-size: 28px;
    color: #2a5885;
  }
  img {
    display: block;
    margin: auto;
    max-height: 450px;
    max-width: 90%;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  .columns-3 {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.5rem;
  }
  .img-sm {
    max-height: 250px;
  }
  .img-md {
    max-height: 350px;
  }
  .caption {
    font-size: 18px;
    color: #666;
    text-align: center;
    margin-top: 5px;
  }
  .reference {
    font-size: 18px;
    color: #666;
    border-top: 1px solid #eee;
    padding-top: 8px;
    margin-top: 20px;
  }
  .formula-block {
    margin: 1em 0;
  }
  .highlight {
    color: #d32f2f;
    font-weight: bold;
  }
  ul, ol {
    margin-top: 0.5em;
    margin-bottom: 0.5em;
  }
---

<!-- _class: lead -->
# Identification of Majorana Zero Modes and Dirac Fermionic Modes in Transport Experiments

<div class="reference">
Analysis of transport properties and their signatures in topological quantum systems
</div>

---

## Transport Signatures of Majorana Zero Modes

<div class="columns">
<div>

### Key Transport Features
- Resonant Andreev reflection
- Quantized zero-bias conductance
- Distinctive shot noise patterns
- Non-local transport correlations

</div>
<div>

![height:320px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-02.jpg?height=828&width=983&top_left_y=416&top_left_x=138)
<div class="caption">Schematic representation of Majorana transport processes</div>

</div>
</div>

---

## Majorana Fermion Induced Resonant Andreev Reflection (MFIRAR)

<div class="columns">
<div>

### Theoretical Foundation
- Perfect Andreev reflection at zero energy
- Quantized conductance $G = 2e^2/h$
- Low energy condition: $\mathrm{eV}, \Gamma_{\mathrm{i}} \ll E_{M}$
- Fano factor: $F_{ii}=1$

<div class="formula-block">

$$
\begin{gathered}
\mathrm{eV}, \Gamma_{\mathrm{i}} \ll E_{M} \\
F_{i i}=1
\end{gathered}
$$

</div>

</div>
<div>

![height:320px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-02.jpg?height=465&width=545&top_left_y=852&top_left_x=1752)
<div class="caption">Crossed Andreev reflection processes</div>

</div>
</div>

---

## Distinguishing MBS/QMBS and ABS in Transport Measurements

<div class="columns">
<div>

### Experimental Signatures
- MBS/QMBS: Zero-bias dip in shot noise
- ABS: Zero-bias peak in shot noise
- Transmission probabilities differ qualitatively
- Non-local correlation patterns serve as fingerprint

</div>
<div>

<div style="display: flex; flex-direction: column; gap: 10px;">
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-02.jpg?height=456&width=494&top_left_y=861&top_left_x=2292" class="img-sm" />
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-02.jpg?height=460&width=481&top_left_y=859&top_left_x=2779" class="img-sm" />
</div>

</div>
</div>

<div class="reference">
J.-S. Hong et al., Phys. Rev. B 105, 024503 (2022);<br>
Law et al., Phys. Rev. Lett. 103, 237001 (2009);<br>
Nilsson et al., Phys. Rev. Lett. 101, 120403 (2008)
</div>

---

## S-matrix Analysis of Jackiw-Rebbi Modes for Braiding Applications

![height:240px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-03.jpg?height=269&width=3058&top_left_y=384&top_left_x=138)

<div class="columns">
<div>

### SSH Chain Edge States
- Two Jackiw-Rebbi modes localized at edges
- Hamiltonian decomposition approach
- Tunability of coupling via parameters
- Braiding requirements: energetic isolation

</div>
<div>

### Model Hamiltonian
$H=H_{L 1}^{\prime}+H_{L 2}^{\prime}+H_{c}+H_{T 1}+H_{T 2}$

$t_{0} \equiv t^{2} / v_{f}$

$H_{L i}^{\prime}=-i v_{f} \int_{-\infty}^{\infty} d x^{\prime} \psi_{i}^{\dagger}\left(x^{\prime}\right) \partial_{x^{\prime}} \psi_{i}\left(x^{\prime}\right)$

$H_{c}=\left(\epsilon \varphi_{1}^{\dagger} \varphi_{2}+\right.$ h.c. $)+\Delta \varphi_{1}^{\dagger} \varphi_{1}-\Delta \varphi_{2}^{\dagger} \varphi_{2}$

</div>
</div>

---

## S-matrix Formulation for Jackiw-Rebbi Zero Modes

<div class="formula-block">

$$
S=\frac{1}{\left(\epsilon^{2}+\Delta^{2}+\frac{t_{0}^{2}}{4}-E^{2}\right)-i t_{0} E}\left(\begin{array}{cc}
\left(\epsilon^{2}+\Delta^{2}-\frac{t_{0}^{2}}{4}-E^{2}\right)+i t_{0} \Delta & i \epsilon t_{0} \\
i \epsilon t_{0} & \left(\epsilon^{2}+\Delta^{2}-\frac{t_{0}^{2}}{4}-E^{2}\right)-i t_{0} \Delta
\end{array}\right)
$$

</div>

<div class="columns">
<div>

### Transmission Probability

$$
T(E)=\left|S_{12}\right|^{2}=\frac{1}{\frac{1}{\epsilon^{2} t_{0}^{2}}\left(E^{2}+\frac{1}{4} t_{0}^{2}-\epsilon^{2}-\Delta^{2}\right)^{2}+\left(1+\frac{\Delta^{2}}{\epsilon^{2}}\right)}
$$

</div>
<div>

### Coupling Terms
$H_{T i}=-i t_{i} \varphi_{i}^{\dagger} \psi_{i}(x=0)+$ h.c.

### Fano Factors
$$
F_{12}=-\left.\frac{e}{\pi \hbar}(1-T)\right|_{E=E_{f}}
$$

$$
F_{11}=\left.\frac{e}{\pi \hbar}(1-T)\right|_{E=E_{f}}
$$

</div>
</div>

<div class="caption">
$2\Delta$ represents the energy splitting between two Jackiw-Rebbi zero modes
</div>

---

## Transport Coefficients Analysis

<div class="columns">
<div>

### Transmission coefficient $T(E)$
![height:320px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-04.jpg?height=898&width=1490&top_left_y=691&top_left_x=114)
<div class="caption">Transmission coefficient for fixed $\Delta/\varepsilon$ ratios</div>

</div>
<div>

### Noise derivative $\partial S_{11}/\partial E(E)$
![height:320px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-04.jpg?height=966&width=1485&top_left_y=640&top_left_x=1724)
<div class="caption">Noise characteristics showing distinctive features at zero energy</div>

</div>
</div>

---

## Majorana Representation of Jackiw-Rebbi Zero Modes

<div class="columns">
<div>

### Fermion to Majorana Mapping
$$
\begin{gathered}
\varphi_{1} \equiv \frac{\gamma_{1}^{a}+i \gamma_{1}^{b}}{2} \\
\varphi_{2} \equiv \frac{\gamma_{2}^{a}+i \gamma_{2}^{b}}{2}
\end{gathered}
$$

### Hamiltonian in Majorana Basis
$$
\begin{gathered}
H_{\mathrm{c}} = \frac{\varepsilon}{2}\left(i \gamma_{1}^{a} \gamma_{2}^{a}-i \gamma_{1}^{b} \gamma_{2}^{b}\right) \\
+ \frac{\Delta}{2}\left(i \gamma_{1}^{a} \gamma_{1}^{b}+i \gamma_{2}^{a} \gamma_{2}^{b}\right)
\end{gathered}
$$

</div>
<div>

### Lead and Tunnel Hamiltonians
$$
H_{L i}^{\prime}=-i v_{f} \int_{-\infty}^{\infty} d x^{\prime} \psi_{i}^{\dagger}\left(x^{\prime}\right) \partial_{x^{\prime}} \psi_{i}\left(x^{\prime}\right)
$$

$$
\begin{gathered}
H_{T 1}=-\frac{i}{2} \tau_{1}^{a}\left[\psi_{2} \gamma_{2}^{a}(x=0)+\psi_{1}^{\dagger}(x=0)\right] \\
-\frac{i}{2} t_{1} \gamma_{1}^{b}\left[\frac{\psi_{2}(x=0)-\psi_{2}^{\dagger}(x=0)}{i}\right]
\end{gathered}
$$

$$
H_{T 2}=-\frac{i}{2} t_{2} \gamma_{2}^{b}\left[\psi_{2}(x=0)+\psi_{2}^{\dagger}(x=0)\right]
$$

</div>
</div>

---

## Dirac to Majorana Decomposition

<div class="columns">
<div>

### Key Phenomenon
When $\Delta=0$, a pair of Dirac fermion modes decomposes into two separate Majorana modes

### Electron-Hole Representation
$$\left(\begin{array}{c}
\psi_{1, E}\left(0^{+}\right) \\
\psi_{1,-E}^{\dagger}\left(0^{+}\right) \\
\psi_{2, E}\left(0^{+}\right) \\
\psi_{2,-E}^{\dagger}\left(0^{+}\right)
\end{array}\right)=
S\left(\begin{array}{c}
\psi_{1, E}\left(0^{-}\right) \\
\psi_{1,-E}^{\dagger}\left(0^{-}\right) \\
\psi_{2, E}\left(0^{-}\right) \\
\psi_{2,-E}^{\dagger}\left(0^{-}\right)
\end{array}\right)$$

</div>
<div>

![height:340px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-05.jpg?height=396&width=1069&top_left_y=389&top_left_x=1847)
<div class="caption">Schematic representation of Dirac mode decomposition into Majorana modes</div>

</div>
</div>

---

## Paired Majorana Transport: $\gamma_{1}^{b}-\gamma_{2}^{b}$ Channel Analysis

<div class="formula-block" style="font-size: 0.9em">

$$
S=\frac{1}{\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)+i \tilde{E}}\left(\begin{array}{cccc}
\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E} & \frac{1}{4}(1+2 i \tilde{E}) & \frac{1}{2} i \tilde{\epsilon} & \frac{1}{2} i \tilde{\epsilon} \\
\frac{1}{4}(1+2 i \tilde{E}) & \tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E} & -\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{2} i \tilde{\epsilon} \\
\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{2} i \tilde{\epsilon} & \tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E} & -\frac{1}{4}(1+2 i \tilde{\epsilon}) \\
\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{4}(1+2 i \tilde{\epsilon}) & \tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E}
\end{array}\right)
$$

</div>

<div class="columns">
<div>

### Parameter Definitions
- $t_{0} \equiv \frac{t^{2}}{v_{f}}$: Effective coupling
- $\tilde{E} \equiv E / t_{0}$: Normalized energy
- $\tilde{\epsilon} \equiv \epsilon / t_{0}$: Normalized coupling

</div>
<div>

### Transport Implications
- 4×4 S-matrix from electron-hole basis
- Off-diagonal elements enable crossed processes
- Distinctive interference patterns
- Signature of Majorana entanglement

</div>
</div>

---

## Shot Noise and Fano Factor Formalism

<div class="columns">
<div>

### Current Operator
$$
\left\langle I_{i}\right\rangle=\frac{e}{h} \sum_{j=1,2} \sum_{a, b=e, h} \operatorname{sgn}(a)\left[\delta_{i j} \delta_{a b}-\left|S_{i j}^{a b}(E)\right|^{2}\right] f_{j}^{b}(E)
$$

### Noise Correlator
$$
\begin{aligned}
S_{i j}=\frac{2 e^{2}}{h} \sum_{k, l=1,2} \sum_{a, b, g, d=e, h} & \operatorname{sgn}(a) \operatorname{sgn}(b) \\
& \times A_{k g ; l d}(i, a, E) A_{l d ; k g}(j, b, E) \\
& \times f_{k}^{g}(E)\left[1-f_{l}^{d}(E)\right]
\end{aligned}
$$

### A-matrix Definition
$$
A_{k g ; l d}(i, a, E)=\delta_{i k} \delta_{i l} \delta_{a g} \delta_{a d}-S_{i k}^{a \dagger}(E) S_{i l}^{S d}(E)
$$

</div>
<div>

![height:300px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-07.jpg?height=656&width=1026&top_left_y=272&top_left_x=2158)
<div class="caption">Numerical results showing zero-energy suppression of noise, a key Majorana signature</div>

</div>
</div>

---

## Noise Suppression at Zero Energy

<div class="formula-block" style="font-size: 0.85em">

$$
\begin{aligned}
d S_{11} / d E & =\left[f_{1}^{e}\left(1-f_{1}^{h}\right)+f_{1}^{h}\left(1-f_{1}^{e}\right)\right] \cdot \frac{\tilde{E}^{2}\left[\tilde{\epsilon}^{2}-\tilde{E}^{2}-\frac{1}{4}\right]^{2}}{\left[\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)^{2}+\tilde{E}^{2}\right]^{2}} \\
& +\left[f_{1}^{e}\left(1-f_{2}^{e}\right)+f_{1}^{h}\left(1-f_{2}^{h}\right)+\ldots+f_{2}^{h}\left(1-f_{1}^{e}\right)\right] \cdot \frac{\frac{1}{4} \tilde{\epsilon}^{2} \cdot\left[\tilde{\epsilon}^{2}-\tilde{E}^{2}-\frac{1}{4}\right]^{2}}{\left[\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)^{2}+\tilde{E}^{2}\right]^{2}}
\end{aligned}
$$

$$
d\left\langle I_{1}\right\rangle / d E=\frac{e}{h}\left(f_{1}^{e}-f_{1}^{h}\right) \frac{\frac{1}{2}\left(\tilde{\epsilon}^{2}+\tilde{E}^{2}+\frac{1}{4}\right)}{\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)^{2}+\tilde{E}^{2}}
$$

</div>

<div class="columns">
<div>

### Critical Features
- Noise is suppressed to zero at specific energies
- Form factor depends on coupling strengths
- Distinctive from trivial resonances
- Robust against moderate disorder

</div>
<div>

![height:200px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-07.jpg?height=634&width=1030&top_left_y=1112&top_left_x=2160)
<div class="caption">Differential noise characteristics around zero energy</div>

</div>
</div>

---

## NEGF Method for SSH-Chain Analysis

![height:180px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-08.jpg?height=239&width=1056&top_left_y=263&top_left_x=284)

<div class="columns">
<div>

### Model Parameters
- Topological criterion: $t_{1} < t_{2}$
- On-site energy difference: $\Delta=\left|\varepsilon_{A}-\varepsilon_{B}\right|$
- Transport calculated via NEGF formalism

### Key Formula
$$T(E)=\operatorname{Tr}\left[\Gamma_{L}(E) G^{r}(E) \Gamma_{R}(E) G^{a}(E)\right]$$

$$I=\frac{e}{h} \int d E T(E)\left[f_{L}(E)-f_{R}(E)\right]$$

$$S=\frac{2 e^{2}}{h} \int d E T(E)[1-T(E)]\left[f_{L}(E)-f_{R}(E)\right]^{2}$$

</div>
<div>

### Shot Noise vs. Lead-Chain Coupling
Parameters: $\Delta=0$, $t_{1}=0.6$, $t_{2}=1.2$

![height:250px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-08.jpg?height=1000&width=1489&top_left_y=648&top_left_x=1705)
<div class="caption">Shot noise as function of lead-chain coupling strength</div>

</div>
</div>

---

## Transmission vs. On-site Energy Difference

![height:160px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=252&width=1060&top_left_y=265&top_left_x=282)

### Parameter Relationships
- $t_{\text{lead-c}} \sim t$: Lead-chain coupling
- $t_{1}, t_{2} \sim \epsilon$: Intracell and intercell hoppings
- $\Delta \sim \Delta$: On-site energy difference

<div class="columns-3">
<div>
  <div class="caption">$\Delta=0.02$</div>
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=775&width=1047&top_left_y=850&top_left_x=34" class="img-sm" />
</div>
<div>
  <div class="caption">$\Delta=0.05$</div>
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=771&width=1043&top_left_y=856&top_left_x=1120" class="img-sm" />
</div>
<div>
  <div class="caption">$\Delta=0.1$</div>
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=745&width=1017&top_left_y=852&top_left_x=2209" class="img-sm" />
</div>
</div>

<div class="reference">
The NEGF numerical results are consistent with S-matrix analytical predictions, confirming the validity of both approaches.
</div>

---

## Shot Noise Dependence on $\Delta$

![height:180px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=231&width=1059&top_left_y=310&top_left_x=287)

<div class="columns-3">
<div>
  <div class="caption">$\Delta=0.02$</div>
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=801&width=1089&top_left_y=922&top_left_x=55" class="img-sm" />
</div>
<div>
  <div class="caption">$\Delta=0.05$</div>
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=813&width=1073&top_left_y=912&top_left_x=1135" class="img-sm" />
</div>
<div>
  <div class="caption">$\Delta=0.1$</div>
  <img src="https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=809&width=1106&top_left_y=918&top_left_x=2224" class="img-sm" />
</div>
</div>

### Parameter Mapping
- $t_{0} \equiv \frac{t^{2}}{v_{f}}$: Effective coupling strength
- $\tilde{E} \equiv E / t_{0}$: Normalized energy
- $\tilde{\epsilon} \equiv \epsilon / t_{0}$: Normalized intracell hopping

---

<!-- _class: lead -->
## Conclusions

<div class="columns">
<div>

### Majorana Zero Mode Signatures
- Quantized zero-bias conductance
- Shot noise suppression at zero energy
- Distinctive Fano factor behavior
- Novel non-local correlations

### Braiding Applications
- Requires robust topological protection
- Minimal energy splitting ($\Delta \to 0$)
- Control of coupling parameters
- Engineering of interference patterns

</div>
<div>

### Experimental Implications
- Transport measurements provide unambiguous signatures
- Shot noise is crucial for distinguishing trivial and topological states
- Parameter tuning enables functional quantum devices
- Combined theoretical approaches (S-matrix and NEGF) provide complementary insights

### Future Directions
- Higher-order correlations in multi-terminal systems
- Time-dependent transport signatures
- Effects of interactions and disorder
- Implementation in scalable quantum computing architectures

</div>
</div>


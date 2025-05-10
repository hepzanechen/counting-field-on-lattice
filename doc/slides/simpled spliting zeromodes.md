---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-size: 24px;
    text-align: left;
    padding: 30px;
  }
  h1 {
    font-size: 36px;
    color: #234;
  }
  h2 {
    font-size: 32px;
    color: #345;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
  }
  img {
    display: block;
    margin: auto;
    max-height: 350px;
    max-width: 85%;
  }
  .img-sm {
    max-height: 160px;
  }
  .img-md {
    max-height: 220px;
  }
  .img-lg {
    max-height: 300px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
  }
  .small {
    font-size: 0.8em;
  }
  .math {
    font-size: 0.9em;
  }
  .ref {
    font-size: 0.7em;
    color: #666;
  }
---

<!-- headingDivider: 2 -->

# How to identify Majorana zero modes and Dirac fermionic modes in the transport experiment

## Noise evidence for detecting Majorana zero modes

Transportation process of Majorana
![height:480px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-02.jpg?height=828&width=983&top_left_y=416&top_left_x=138)

Majorana fermion induced resonant Andreev reflection (MFIRAR).

$$
\begin{gathered}
\mathrm{eV}, \Gamma_{\mathrm{i}} \ll E_{M} \\
F_{i i}=1
\end{gathered}
$$
Crossed Andreev reflections dominate

## Noise evidence for detecting Majorana zero modes


<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start;">

  <div style="text-align: center;">
    <img src="./img/probe_two_gamma.png" style="height: 200px;" />
    <p><em>Probing scheme for two Majorana modes</em></p>
  </div>

  <div style="text-align: center;">
    <img src="./img/shot_noise_twogamma.png" style="width: 600px;" />
    <p><em>an MBS/QMBS (a zero-bias dip) and ABS (a zero-bias peak) in current shot noise</em></p>
  </div>

</div>




</div>
</div>

<div class="ref">
J.-S. Hong, T.-F. J. Poon, L. Zhang, and X.-J. Liu, Phys. Rev. B 105, 024503 (2022)

Law K T, Lee P A, Ng T K. Majorana fermion induced resonant Andreev reflection[J]. Physical review letters, 2009, $103(23): 237001$.

Nilsson J, Akhmerov A R, Beenakker C W J. Splitting of a Cooper pair by a pair of Majorana bound states[J]. Physical review letters, 2008 , $101(12)$ : 120403.
</div>

## Jackiw-Rebbi modes Hamiltonian

![height:120px](./img/ssh_chain_two_leads.png)

Two Jackiw-Rebbi modes exists at the edge of the SSH chain

$$
\begin{aligned}
& H=H_{L 1}^{\prime}+H_{L 2}^{\prime}+H_c+H_{T 1}+H_{T 2} \\
& H_{L i}^{\prime}=-i v_f \int_{-\infty}^{\infty} d x^{\prime} \psi_i^{\dagger}\left(x^{\prime}\right) \partial_{x^{\prime}} \psi_i\left(x^{\prime}\right) \\
& H_c=\left(\epsilon \varphi_1^{\dagger} \varphi_2+h . c .\right)+\Delta \varphi_1^{\dagger} \varphi_1-\Delta \varphi_2^{\dagger} \varphi_2 \\
& H_{T i}=-i t_i \varphi_i^{\dagger} \psi_i(x=0)+h . c .
\end{aligned}
$$
2∆ are the energy difference between two Jackiw-Rebbi zero-modes 


# Scattering and Noise Properties

$$
\begin{pmatrix}
\psi_{1, E}\left(0^{+}\right) \\
\psi_{2, E}\left(0^{+}\right)
\end{pmatrix}
=
S
\begin{pmatrix}
\psi_{1, E}\left(0^{-}\right) \\
\psi_{2, E}\left(0^{-}\right)
\end{pmatrix}
$$

$$
S = \frac{1}{
\left( \epsilon^{2} + \Delta^{2} + \frac{t_0^2}{4} - E^{2} \right) - i t_0 E
}
\begin{pmatrix}
\left( \epsilon^{2} + \Delta^{2} - \frac{t_0^2}{4} - E^{2} \right) + i t_0 \Delta & i \epsilon t_0 \\
i \epsilon t_0 & \left( \epsilon^{2} + \Delta^{2} - \frac{t_0^2}{4} - E^{2} \right) - i t_0 \Delta
\end{pmatrix}
$$

$$
\begin{aligned}
T(E) &= |S_{12}|^2 = 
\frac{1}{
\frac{1}{\epsilon^2 t_0^2} \left( E^2 + \frac{1}{4} t_0^2 - \epsilon^2 - \Delta^2 \right)^2 + \left(1 + \frac{\Delta^2}{\epsilon^2}\right)
} \\
\left. \frac{\partial \mathcal{S}_{11}}{\partial E} \right|_{E = E_f} &= 
\left. \frac{e^2}{\pi \hbar} T(1 - T) \right|_{E = E_f} \\
\mathcal{S}_{12} = \left. \frac{\partial \mathcal{S}_{12}}{\partial I} \right|_{E = E_f} &= 
- \left. \frac{e}{\pi \hbar}(1 - T) \right|_{E = E_f} \\
\mathcal{S}_{11} = \left. \frac{\partial \mathcal{S}_{11}}{\partial I} \right|_{E = E_f} &= 
\left. \frac{e}{\pi \hbar}(1 - T) \right|_{E = E_f}
\end{aligned}
$$

> $S$ is the scattering matrix; $\mathcal{S}$ denotes noise spectrum.



## S-matrix method for judging if Jackiw rebbi modes can be used for braiding

Transmission coefficient $T(E)$

Transmission Coefficient $T$ for Fixed $\Delta / \varepsilon$ Ratios

<div class="grid">
<div>

![height:220px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-04.jpg?height=898&width=1490&top_left_y=691&top_left_x=114)

</div>
<div>

Noise $\partial S_{11} / \partial E(E)$

![height:220px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-04.jpg?height=966&width=1485&top_left_y=640&top_left_x=1724)

</div>
</div>

## S-matrix method for judging if Jackiw rebbi modes can be used for braiding

In the Majorana representation

<div class="math">

$$
\begin{gathered}
\varphi_{1} \equiv \frac{\gamma_{1}^{a}+i \gamma_{1}^{b}}{2} \quad \varphi_{2} \equiv \frac{\gamma_{2}^{a}+i \gamma_{2}^{b}}{2} \\
H_{\mathrm{c}} \equiv \\
\equiv \frac{\varepsilon}{2}\left(i \gamma_{1}^{a} \gamma_{2}^{a}-i \gamma_{1}^{b} \gamma_{2}^{b}\right)+\frac{\Delta}{2}\left(i \gamma_{1}^{a} \gamma_{1}^{b}+i \gamma_{2}^{a} \gamma_{2}^{b}\right) \\
H_{L i}^{\prime}= \\
H_{T 1}=-i v_{f} \int_{-\infty}^{\infty} d x^{\prime} \psi_{i}^{\dagger}\left(x^{\prime}\right) \partial_{x^{\prime}} \psi_{i}\left(x^{\prime}\right): \\
H_{T 2}=-\frac{i}{2} \tau_{1}^{a}\left[\psi_{2} \gamma_{2}^{a}(x=0)+\psi_{1}^{\dagger}(x=0)\right]-\frac{i}{2} t_{1} \gamma_{1}^{b}\left[\frac{\psi_{2}(x=0)-\psi_{2}^{\dagger}(x=0)}{i}\right]-\frac{i}{2} t_{2} \gamma_{2}^{b}\left[\psi_{2}(x=0)+\psi_{2}^{\dagger}(x=0)\right]
\end{gathered}
$$

</div>

![height:200px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-05.jpg?height=396&width=1069&top_left_y=389&top_left_x=1847)

When $\Delta=0$, A pair of Dirac modes decomposes into two majorana modes

In the electron-hole representation

$$\left(\begin{array}{c}\psi_{1, E}\left(0^{+}\right) \\ \psi_{1,-E}^{\dagger}\left(0^{+}\right) \\ \psi_{2, E}\left(0^{+}\right) \\ \psi_{2,-E}^{\dagger}\left(0^{+}\right)\end{array}\right)=S\left(\begin{array}{c}\psi_{1, E}\left(0^{-}\right) \\ \psi_{1,-E}^{\dagger}\left(0^{-}\right) \\ \psi_{2, E}\left(0^{-}\right) \\ \psi_{2,-E}^{\dagger}\left(0^{-}\right)\end{array}\right)$$

## S-matrix method for judging if Jackiw rebbi modes can be used for braiding

A pair of Majorana transport process
$\gamma_{1}^{b}-\gamma_{2}^{b}$ channel

<div class="math">

$$
S=\frac{1}{\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)+i \tilde{E}}\left(\begin{array}{cccc}
\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E} & \frac{1}{4}(1+2 i \tilde{E}) & \frac{1}{2} i \tilde{\epsilon} & \frac{1}{2} i \tilde{\epsilon} \\
\frac{1}{4}(1+2 i \tilde{E}) & \tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E} & -\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{2} i \tilde{\epsilon} \\
\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{2} i \tilde{\epsilon} & \tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E} & -\frac{1}{4}(1+2 i \tilde{\epsilon}) \\
\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{2} i \tilde{\epsilon} & -\frac{1}{4}(1+2 i \tilde{\epsilon}) & \tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{2} i \tilde{E}
\end{array}\right)
$$

</div>

$t_{0} \equiv \frac{t^{2}}{v_{f}}, \tilde{E} \equiv E / t_{0} \quad \tilde{\epsilon} \equiv \epsilon / t_{0}$.

## S-matrix method for judging if Jackiw rebbi modes can be used for braiding

Shot noise and Fano factor for a single pair of Majorana

<div class="math">

$$
\begin{aligned}
& \left\langle I_{i}\right\rangle=\frac{e}{h} \sum_{j=1,2} \sum_{a, b=e, h} \operatorname{sgn}(a)\left[\delta_{i j} \delta_{a b}-\left|S_{i j}^{a b}(E)\right|^{2}\right] f_{j}^{b}(E) \\
& S_{i j}=\frac{2 e^{2}}{h} \sum_{k, l=1,2, a, b, g, d=e, h} \sum_{c} \operatorname{sgn}(a) \operatorname{sgn}(b) A_{k g ; l d}(i, a, E) A_{l d ; k g}(j, b, E) f_{k}^{g}(E)\left[1-f_{l}^{d}(E)\right] \\
& \quad A_{k g ; l d}(i, a, E)=\delta_{i k} \delta_{i l} \delta_{a g} \delta_{a d}-S_{i k}^{a \dagger}(E) S_{i l}^{S d}(E)
\end{aligned}
$$

</div>

<div class="grid">
<div>

![height:220px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-07.jpg?height=656&width=1026&top_left_y=272&top_left_x=2158)

</div>
<div>

The lowest point of the noise is suppressed to 0

<div class="small">

$$
\begin{aligned}
d S_{11} / d E & =\left[f_{1}^{e}\left(1-f_{1}^{h}\right)+f_{1}^{h}\left(1-f_{1}^{e}\right)\right] \cdot \frac{\tilde{E}^{2}\left[\tilde{\epsilon}^{2}-\tilde{E}^{2}-\frac{1}{4}\right]^{2}}{\left[\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)^{2}+\tilde{E}^{2}\right]^{2}} \\
& +\left[f_{1}^{e}\left(1-f_{2}^{e}\right)+f_{1}^{h}\left(1-f_{2}^{h}\right)+f_{1}^{h}\left(1-f_{2}^{e}\right)+f_{1}^{e}\left(1-f_{2}^{h}\right)+f_{2}^{e}\left(1-f_{1}^{e}\right)+f_{2}^{h}\left(1-f_{1}^{h}\right)\right. \\
& \left.+f_{2}^{e}\left(1-f_{1}^{h}\right)+f_{2}^{h}\left(1-f_{1}^{e}\right)\right] \cdot \frac{\frac{1}{4} \tilde{\epsilon}^{2} \cdot\left[\tilde{\epsilon}^{2}-\tilde{E}^{2}-\frac{1}{4}\right]^{2}}{\left[\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)^{2}+\tilde{E}^{2}\right]^{2}}
\end{aligned}
$$

</div>

$$
d\left\langle I_{1}\right\rangle / d E=\frac{e}{h}\left(f_{1}^{e}-f_{1}^{h}\right) \frac{\frac{1}{2}\left(\tilde{\epsilon}^{2}+\tilde{E}^{2}+\frac{1}{4}\right)}{\left(\tilde{\epsilon}^{2}-\tilde{E}^{2}+\frac{1}{4}\right)^{2}+\tilde{E}^{2}}
$$

</div>
</div>

![height:180px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-07.jpg?height=634&width=1030&top_left_y=1112&top_left_x=2160)

<div class="ref">
Anantram M P, Datta S. Current fluctuations in mesoscopic systems with
</div>

## Numercial method for solving SSH-chain

![height:160px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-08.jpg?height=239&width=1056&top_left_y=263&top_left_x=284)

<div class="grid">
<div>

$t_{1}<t_{2}$ topological nontrival

on-site energy difference $\Delta=\left|\varepsilon_{A}-\varepsilon_{B}\right|$

NEGF method

$T(E)=\operatorname{Tr}\left[\Gamma_{L}(E) G^{r}(E) \Gamma_{R}(E) G^{a}(E)\right]$

$I=\frac{e}{h} \int d E T(E)\left[f_{L}(E)-f_{R}(E)\right]$

$S=\frac{2 e^{2}}{h} \int d E T(E)[1-T(E)]\left[f_{L}(E)-f_{R}(E)\right]^{2}$

</div>
<div>

Shot noise vs. $t_{\text {lead-c }} \quad \Delta=0$

$t_{1}=0.6 \quad t_{2}=1.2$

Shot Noise vs $t_{\text {lead_c }}$

![height:200px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-08.jpg?height=1000&width=1489&top_left_y=648&top_left_x=1705)

</div>
</div>

## Numercial method for solving SSH-chain

![height:160px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=252&width=1060&top_left_y=265&top_left_x=282)

$$
\mathrm{T}(\mathrm{E}) \text { vs. } \Delta \quad t_{1}=0.6 \quad t_{2}=1.2
$$

Physical quantities vs. S-matrix $\quad t_{0} \equiv \frac{t^{2}}{v_{f}}, \tilde{E} \equiv E / t_{0} \tilde{\epsilon} \equiv \epsilon / t_{0}$.

$$
t_{\text {lead }-c} \sim t \quad t_{1}, t_{2} \sim \epsilon \quad \Delta \sim \Delta
$$

<div class="grid-3">
<div>

$\Delta=0.02$
![height:150px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=775&width=1047&top_left_y=850&top_left_x=34)

</div>
<div>

$\Delta=0.05$
![height:150px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=771&width=1043&top_left_y=856&top_left_x=1120)

</div>
<div>

$\Delta=0.1$
![height:150px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-09.jpg?height=745&width=1017&top_left_y=852&top_left_x=2209)

</div>
</div>

The NEGF method is consistent with the results obtained by the S-matrix method

## ${ }_{t_{1}} \mathrm{NI}_{t_{2}} \mathrm{~m}_{2}$ orrinl mothnd for solving SSH-chain

![height:160px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=231&width=1059&top_left_y=310&top_left_x=287)

Shot noise vs. $\Delta$

$$
t_{1}=0.6 \quad t_{2}=1.2
$$

Physical quantities vs. S-matrix

<div class="grid-3">
<div>

$$
\Delta=0.02
$$

![height:150px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=801&width=1089&top_left_y=922&top_left_x=55)

</div>
<div>

$t_{0} \equiv \frac{t^{2}}{v_{f}}, \tilde{E} \equiv E / t_{0} \tilde{\epsilon} \equiv \epsilon / t_{0}$.

$\Delta=0.05$

![height:150px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=813&width=1073&top_left_y=912&top_left_x=1135)

</div>
<div>

$t_{\text {lead-c }} \sim t \quad t_{1}, t_{2} \sim \epsilon \quad \Delta \sim \Delta$

$$
\Delta=0.1
$$

![height:150px](https://cdn.mathpix.com/cropped/2025_05_09_77a2acaaea6dec30a1ecg-10.jpg?height=809&width=1106&top_left_y=918&top_left_x=2224)

</div>
</div>


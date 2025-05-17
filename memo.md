To make all things get done in GPU, E should be inited at GPU, the device of E it's the standard intermeidate tesor's device choose to follow.
Also lead's properties, central's properties should be checked.
the device is contrled by funcDevice, set to Ebatch device or lead.v1laph device in ginv_total

* lead_decimation calc hole and electron seperately and combine them in add_ginv_lead, and inv the whole, this may be improved by 1. pass in H_lead_BdG together, 2. use the hole and electron seperately to calc the hole and electron seperately, and combine them in add_ginv_lead use symmetric property.
* tLC can only be real number for now
* since $$ is current, the hermitial part automatically handles the minius sign needed becuae $\exp{(-i \lambda)}$ also reverse sign due to hermitian.
# sign and Real imag of genfunc/lambda
$$\left\langle\left\langle(\Delta n)^j\right\rangle\right\rangle=\left.\frac{\partial^j \ln Z(\lambda, t)}{\partial(i \lambda)^j}\right|_{\lambda=0}$$
$ \ln S \propto \frac{1}{2}(\psi_{1}^{\dagger}(E)\psi_{2}(E) + \psi_{2}^{\dagger}(E)\psi_{1}(E)) - \frac{1}{2}(\psi_{2}(-E)\psi_{1}^{\dagger}(-E) + \psi_{1}(-E)\psi_{2}^{\dagger}(-E)) $

$-E$ means $\omega\psi_{1}^{\dagger}\psi_{1} \rightarrow -\omega\psi_{1}\psi_{1}^{\dagger}$ corresponding hole energy is $-E$ and this is due to p-h symmetry redundancy, while $-\frac{1}{2}(\psi_{2}(-E)\psi_{1}^{\dagger}(-E))$ means $t\psi_{1}^{\dagger}\psi_{2} \rightarrow -t\psi_{2}\psi_{1}^{\dagger}$

It should be $tLch = -tLce^{\dagger}$

* eta can causing to lead current conservation fails, so we should let eta=0

* since electron and hole are $E<->-E$ symmetry, so there are issues if $\mu_1=-\mu_2$, then physically there should be curren, but the calculated current is not zero, so we should make hole part $-E$ in the gen_ginv_lead.

In direct calculation, the noise is calculated by the following formula:
$\begin{aligned}S_{ij}&=\frac{e^2}h\sum_{\alpha,\beta,k\gamma,l,\delta m\zeta,n,\eta}\operatorname{sgn}(\alpha)\operatorname{sgn}(\beta)\\&\times\int dE\:A_{k\gamma;l\delta}(i,E)A_{l\delta;k\gamma}(j,E)f_{k\gamma}(E)[1-f_{l\delta}(E)].\end{aligned}$
The coefficient before 'T' related dimensionless terms: $\dot{S_{LL}}=\frac{e^3|V|}{\pi\hbar}\sum_nT_n\left(1-T_n\right).$, $S_P=\frac{e^3|V|}{\pi\hbar}\sum_nT_n=2e\langle I\rangle.$.
current formula: $I_i=\frac eh\sum_{\alpha,j\in NS,\beta}\operatorname{sgn}(\alpha)[\delta_{ij}\delta_{\alpha\beta}-T_{ij}^{\alpha\beta}(E)]f_{j\beta}(E)$

## Unit
1. $e=1$
2. $\hbar=1$
3. $\Phi_0=2\pi$

Besides Nx,Ny is int, all other variables are initialized as torch.Tensor, and the device is set to funcDevice in the main function, for e.g. vortex_positions is a list of tuples of torch.Tensor.


$v_F=\frac{ta}\hbar,$

eta value should fit in E numpoints, for e.g. when eta is small E numpoints should be dense enough to trapz accurately.

* For e.g. " tCL[lead.position[idx], :] = lead.V1alpha[idx, :]" lead.position is a list of int, decided by central Nx,Ny.

# Electron and Hole
Note for both augograd and direct inv G method, for BdG case, we should only include electron part, this is make sure by 1. only add couting field the electron section
2. for direct inv G current only let current formula's $\alpha$ only loop e and noise $\alpha,\beta$ only loop 'e' 
image.png

# torch
Though both using complex64, for eigenvectors cpu seems more accurate than gpu.


#---
1. Thinking how to use block det to save mem.

2. add suplead and add_ginv_lead will be changed:
"note our formual will be :
$$
\hat{G}^{\mathrm{K}}=\hat{G}^{\mathrm{R}} \hat{F}-\hat{F} \hat{G}^{\mathrm{A}} .
$$
$$
\hat{F}=\left(\begin{array}{cc}
1-2 f_{e l} & 0 \\
0 & 1-2 f_h
\end{array}\right),
$$
next is in Lead.py add ginv lead and construct sns junstion, note current F has both longitudal(equlilibrum $tanh(E-\mu/T)$ term) and transverse term. which means the bias can be introduced.
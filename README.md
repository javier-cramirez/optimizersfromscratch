# optimizersfromscratch

My implementations of some cool optimizers. 

## 1. Sophia
Estimates diagonal Hessian entries using minibatch updates. Every $k$ steps, we update using 
$$\frac{\text{EMA}\nabla_{\theta}}{\text{EMA}H_{\theta}}$$
where $H_{\theta}$ is clipped by a scalar. The reasoning is that Newton's method usually fails to capture positive curvature, which can cause it to converge to local maxima instead of local minima. Instead, the authors propose the update:
$$\theta_{1}\leftarrow \theta_{1}-\nabla\text{clip}\left( \frac{\text{EMA}\nabla_{\theta}}{\text{EMA}H_{\theta}+\epsilon}, \rho\right)$$
Where $\nabla$ is the learning rate, $\rho$ is a positive threshold, and $\epsilon$ is a very small normalization constant (prevent $0$ in denominator).

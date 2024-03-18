# optimizersfromscratch

My implementations of some cool optimizers. 

## 1. Sophia
Estimates diagonal Hessian entries using minibatch updates. Every $k$ steps, we update using $\frac{\text{EMA}\nabla}{\text{EMA}H_{\theta}}$ where $H_{\theta}$ is clipped by a scalar.

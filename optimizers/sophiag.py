from typing import List, Optional, Tuple, Dict, Any
from torch import Tensor
import torch
from torch.optim.optimizer import Optimizer
import math

class SophiaTheG(Optimizer):
    def __init__(self,
                 lr: float = 1e-4,
                 betas: Tuple[float, float]= (0.965, 0.99),
                 eps: float = 1e-12,
                 rho: float = 0.04,
                 weight_decay: float = 1e-1,
                 defaults: Optional[Dict[str, Any]] = None):
        defaults = {} if defaults is None else defaults
        super(SophiaTheG, self).__init__()
        
    def init_state(self, state: Dict[str, Any], group: Dict[str, Any]):
        super().init_state(state)
        for group in self.param_groups:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(memory_format=torch.preserve_format)
            state['hessian'] = torch.zeros_like(memory_format=torch.preserve_format)
            
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for pa in group['params']:
                # Skip params w/o gradients
                if pa.grad is None:
                    continue
                state = self.state[pa]

                if len(state) == 0:
                    self.init_state(state, group, pa)
                
                # Update Hessian with:
                # H_t=B\cdot\del_{\theta}L(\theta)\circ\del_{\theta}L(\theta)
                # h_t = \beta_{2}h_{t-k}+(1-\beta_{2})H_t
                state['hessian'].mul(beta2).addcmul_(pa.grad, pa.grad, value=1-beta2)
                
    @torch.no_grad()
    def param_step(self, batch_size):
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']
            
            for pa in group['params']:
                if pa.grad is None:
                    continue
                params_with_grad.append(pa)
                
            grads.append(pa.grad)
            state = self.state[pa]
            
            if len(state) == 0:
                self.init_state(state, group, pa)
            exp_avgs.append(state['exp_avg'])
            state_steps.append(state['step'])
            hessian.append(state['hessian'])

def sophiag(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            hessian: List[Tensor],
            state_steps: List[Tensor],
            batchsize: int,
            beta1: float,
            beta2: float,
            rho: float, lr: float,
            weight_decay: float):
    
    
            
                
            
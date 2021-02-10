import torch
from torch.types import _TensorOrTensors
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

def custom_grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False
) -> Tuple[torch.Tensor, ...]:

    parmas_index = [i for i,j in enumerate(inputs) if j.requires_grad]
    params_requiring_grad = [i for i in inputs if i.requires_grad]
    
    grads = torch.autograd.grad(outputs, params_requiring_grad, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph, only_inputs = only_inputs)
    grads_ = [0]*len(inputs)
    j = 0
    for i in range(len(inputs)):
        if i in parmas_index:
            grads_[i] = grads[j]
            j+=1
    return grads_
    
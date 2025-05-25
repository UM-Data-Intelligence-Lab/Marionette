from typing import Union

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from datamodule import Batch

patch_typeguard()  # use before @typechecked


@typechecked
def generate_hpp(
    tmax: TensorType,
    n_sequences: int,
    x_n: Batch,
    time_segments : int,
    intensity: Union[TensorType, None] = None,
) -> Batch:
    """
    Generate a batch of sequences from a homogeneous Poisson process on [0,T].

    Returns
    -------
    Batch
        Batch of generated sequences
    """
    device = tmax.device
    if intensity is None:
        intensity = torch.ones(n_sequences, device=device)

    # Get number of samples
    n_samples = torch.poisson(tmax * intensity)
    max_samples = int(torch.max(n_samples).item()) + 1

    # Sample times
    times = torch.rand((n_sequences, max_samples), device=device) * tmax
    

    # Mask for padding events
    mask = (
        torch.arange(0, max_samples, device=device)[None, :]
        < n_samples[:, None]
    )
    times = times * mask

    condition1 = torch.ones_like(times)
    for index in range(1,time_segments+1):
        cond_window1 = times>= index-1
        cond_window2 = times< index 
        combined_condition = cond_window1 & cond_window2
        condition1 = torch.where(combined_condition ,x_n.condition1_indicator[:,[index-1]],condition1)
    condition1 = (condition1 * mask).to(torch.int64)

    condition2 = torch.ones_like(times)
    for index in range(1,time_segments+1):
        cond_window1 = times>= index-1
        cond_window2 = times< index 
        combined_condition = cond_window1 & cond_window2
        condition2 = torch.where(combined_condition ,x_n.condition2_indicator[:,[index-1]],condition2)
    condition2 = (condition2 * mask).to(torch.int64)

    condition3 = torch.ones_like(times)
    for index in range(1,time_segments+1):
        cond_window1 = times>= index-1
        cond_window2 = times< index 
        combined_condition = cond_window1 & cond_window2
        condition3 = torch.where(combined_condition ,x_n.condition3_indicator[:,[index-1]],condition3)
    condition3 = (condition3 * mask).to(torch.int64)

    condition4 = torch.ones_like(times)
    for index in range(1,time_segments+1):
        cond_window1 = times>= index-1
        cond_window2 = times< index 
        combined_condition = cond_window1 & cond_window2
        condition4 = torch.where(combined_condition ,x_n.condition4_indicator[:,[index-1]],condition4)
    condition4 = (condition4 * mask).to(torch.int64)

    condition5 = torch.ones_like(times)
    for index in range(1,time_segments+1):
        cond_window1 = times>= index-1
        cond_window2 = times< index 
        combined_condition = cond_window1 & cond_window2
        condition5 = torch.where(combined_condition ,x_n.condition5_indicator[:,[index-1]],condition5)
    condition5 = (condition5 * mask).to(torch.int64)

    condition6 = torch.ones_like(times)
    for index in range(1,time_segments+1):
        cond_window1 = times>= index-1
        cond_window2 = times< index 
        combined_condition = cond_window1 & cond_window2
        condition6 = torch.where(combined_condition ,x_n.condition6_indicator[:,[index-1]],condition6)
    condition6 = (condition6 * mask).to(torch.int64)



    assert (mask.sum(-1) == n_samples).all(), "wrong number of samples"
    return Batch.remove_unnescessary_padding(
        time=times,
        condition1=condition1,
        condition2=condition2,
        condition3=condition3,
        condition4=condition4,
        condition5=condition5,
        condition6=condition6,
        condition1_indicator=x_n.condition1_indicator,
        condition2_indicator=x_n.condition2_indicator,
        condition3_indicator=x_n.condition3_indicator,
        condition4_indicator=x_n.condition4_indicator,
        condition5_indicator=x_n.condition5_indicator,
        condition6_indicator=x_n.condition6_indicator,
        mask=mask, 
        tmax=tmax, 
        kept=None
    )

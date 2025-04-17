"""Quantization utilities"""
from typing import Tuple
import torch


def symmetric_quantize(
    tensor: torch.Tensor,
    quant_per_channel: bool = True,
    q_max: int = 127,
    q_min: int = -128,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize tensor with symmetric quantization

    Args:
        tensor (torch.Tensor): The tensor to be quantized
        quant_per_channel (bool): Quantize tensor per channel of as a whole
        q_max (int): max value of quantized tensor
        q_min (int): min value of quantized tenosr
        dtype (torch.dtype) quantized weight's dtype

    Returns:
        quantized_tensor (torch.Tensor): quantized tensor with type ```dtype```
        scale (float, torch.Tensor)
        zero_point (torch.Tensor)
    """
    _tensor = tensor.clone()
    zero_point = torch.tensor(0, dtype=dtype)

    if quant_per_channel:
        _tensor = _tensor.view(_tensor.size(0), -1)
        scale = _tensor.abs().max(-1)[0] / q_max
        zero_point = torch.zeros_like(scale, dtype=dtype)
        # broadcast scale to fit tensor size
        _scale = scale.view(scale.size(0), -1).expand_as(_tensor)

        quantized_tensor = (_tensor / _scale).view(tensor.shape).to(dtype).contiguous()
    else:
        scale = _tensor.abs().max() / q_max
        quantized_tensor = (_tensor / scale).to(dtype)

    # clamp value
    quantized_tensor = quantized_tensor.clamp(min=q_min, max=q_max)

    return quantized_tensor, scale, zero_point


def asymmetric_quantize(
    tensor: torch.Tensor,
    quant_per_channel: bool = True,
    q_max: int = 127,
    q_min: int = -128,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize tensor with asymmetric quantization

    Args:
        tensor (torch.Tensor): The tensor to be quantized
        quant_per_channel (bool): Quantize tensor per channel of as a whole
        q_max (int): max value of quantized tensor
        q_min (int): min value of quantized tenosr
        dtype (torch.dtype) quantized weight's dtype

    Returns:
        quantized_tensor (torch.Tensor): quantized tensor with type ```dtype```
        scale (float, torch.Tensor)
        zero_point (torch.Tensor)
    """
    _tensor = tensor.clone()

    if quant_per_channel:
        _tensor = _tensor.view(_tensor.size(0), -1)
        t_max = _tensor.max(-1)[0]
        t_min = _tensor.min(-1)[0]
        scale = (t_max - t_min) / (q_max - q_min)
        zero_point = q_max - (t_max / scale)
        # broadcast scale and zero_point to fit tensor size
        _scale = scale.view(scale.size(0), -1).expand_as(_tensor)
        _zero_point = zero_point.view(zero_point.size(0), -1).expand_as(_tensor)

        quantized_tensor = ((_tensor / _scale) + _zero_point).to(dtype)
        quantized_tensor = quantized_tensor.view(tensor.shape)
    else:
        t_max = _tensor.max()
        t_min = _tensor.min()
        scale = (t_max - t_min) / (q_max - q_min)
        zero_point = q_max - (t_max / scale)

        quantized_tensor = ((_tensor / scale) + zero_point).to(dtype)

    # clamp value
    quantized_tensor = quantized_tensor.clamp(min=q_min, max=q_max)

    return quantized_tensor, scale, zero_point.to(dtype)


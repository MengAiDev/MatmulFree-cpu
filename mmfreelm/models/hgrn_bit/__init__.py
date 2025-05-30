# -*- coding: utf-8 -*-

from .configuration_hgrn_bit import HGRNBitConfig
from .modeling_hgrn_bit import HGRNBitForCausalLM, HGRNBitModel

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("hgrn_bit", HGRNBitConfig)
AutoModelForCausalLM.register(HGRNBitConfig, HGRNBitForCausalLM)


__all__ = [
    "HGRNBitConfig",
    "HGRNBitForCausalLM",
    "HGRNBitModel"
]

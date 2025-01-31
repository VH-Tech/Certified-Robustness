"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch.nn as nn
from .adapter_utils import Activations
from .hypercomplex.layers import PHMLinear, PHMLayer
from .low_rank_layer import LowRankLinear


class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size) 
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim) 

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output 


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim #*
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLayer(in_features=self.input_dim, #
                                      out_features=self.down_sample_size, #
                                      bias=True, #
                                      c_init=config.phm_c_init, #
                                      phm_dim=config.hypercomplex_division, #
                                      learn_phm=config.learn_phm, #
                                      w_init=config.hypercomplex_nonlinearity,#
                                      shared_phm_rule=config.shared_phm_rule,#
                                      factorized_phm=config.factorized_phm,#
                                      shared_W_phm=config.shared_W_phm, #
                                      factorized_phm_rule=config.factorized_phm_rule, #
                                      phm_rank=config.phm_rank, #
                                      phm_init_range=config.phm_init_range,#
                                      kronecker_prod=config.kronecker_prod)#*
        self.up_sampler = PHMLayer(in_features=self.down_sample_size,
                                    out_features=self.input_dim, 
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)
    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)
    

class Compacter(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.config = lambda: None
        self.config.input_dim = input_dim #*
        self.config.reduction_factor = 32 #
        self.config.non_linearity = "gelu" #
        self.config.phm_c_init = "normal" #
        self.config.hypercomplex_division = 4 #
        self.config.learn_phm = True #
        self.config.hypercomplex_nonlinearity = "glorot-uniform" #
        self.config.shared_phm_rule = False #
        self.config.factorized_phm = True
        self.config.shared_W_phm = False #*
        self.config.factorized_phm_rule = False #
        self.config.phm_rank = 1 #
        self.config.phm_init_range = 0.0001 #
        self.config.kronecker_prod = False #
        self.adapter = HyperComplexAdapter(self.config)

    def forward(self, x):
        # print(x.shape)
        output = self.adapter(x)
        return output


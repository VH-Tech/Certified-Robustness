# The codes are from https://github.com/bayer-science-for-a-better-life/phc-gnn
import torch
import torch.nn as nn
from typing import Union, Optional
import torch.nn.functional as F
import functools
from .inits import glorot_uniform, glorot_normal  #phm_init
from .kronecker import kronecker_product, kronecker_product_einsum_batched
import threading

def matvec_product(W: torch.Tensor, x: torch.Tensor,
                       bias: Optional[torch.Tensor],
                       phm_rule: Union[torch.Tensor],
                       kronecker_prod=False) -> torch.Tensor:
    """
    Functional method to compute the generalized matrix-vector product based on the paper
    "Parameterization of Hypercomplex Multiplications (2020)"
    https://openreview.net/forum?id=rcQdycl0zyk
    y = Hx + b , where W is generated through the sum of kronecker products from the Parameterlist W, i.e.
    W is a an order-3 tensor of size (phm_dim, in_features, out_features)
    x has shape (batch_size, phm_dim*in_features)
    phm_rule is an order-3 tensor of shape (phm_dim, phm_dim, phm_dim)
    H = sum_{i=0}^{d} mul_rule \otimes W[i], where \otimes is the kronecker product
    """
    if kronecker_prod:
       H = kronecker_product(phm_rule, W).sum(0)
    else: 
       H = kronecker_product_einsum_batched(phm_rule, W).sum(0)

    y = torch.matmul(input=x, other=H)
    if bias is not None:
        y += bias
    return y

class ForwardContext:
    """
    Holds context information during a forward pass through a model. This class should be used via the
    ``ForwardContext.wrap()`` method.

    Note that the context is thread-local.
    """

    # thread-local storage that holds a stack of active contexts
    storage = threading.local()

    context_attributes = [
        "adapter_gating_scores",
        "adapter_fusion_attentions",
        "adapter_input_parallelized",
    ]
    # Additional used attributes not exposed to the user
    # - prompt_tokens_length: length of the prompt tokens

    def __init__(self, model, *args, **kwargs):
        # If the model has a method ``forward_context()``, use it to create the context.
        if hasattr(model, "forward_context"):
            model.forward_context(self, *args, **kwargs)

    def __enter__(self):
        ForwardContext.get_contexts().append(self)
        return self

    def __exit__(self, type, value, traceback):
        ForwardContext.get_contexts().pop()

    @classmethod
    def wrap(cls, f):
        """
        Decorator method that wraps a ``forward()`` function of a model class.
        """

        @functools.wraps(f)
        def wrapper_func(self, *args, **kwargs):
            if self.adapters_config is not None:
                with cls(self, *args, **kwargs) as ctx:
                    # whether to output the context attributes
                    output_context = kwargs.pop("output_context", False)
                    kwargs = {
                        k: v for k, v in kwargs.items() if k.replace("output_", "") not in cls.context_attributes
                    }
                    results = f(self, *args, **kwargs)

                    # append output attributes
                    if isinstance(results, tuple):
                        for attr in cls.context_attributes:
                            if getattr(ctx, "output_" + attr, False):
                                results = results + (dict(getattr(ctx, attr)),)
                    else:
                        for attr in cls.context_attributes:
                            if getattr(ctx, "output_" + attr, False):
                                results[attr] = dict(getattr(ctx, attr))

                    if output_context:
                        context_dict = ctx.__dict__

                if output_context:
                    return results, context_dict
                else:
                    return results
            else:
                return f(self, *args, **kwargs)

        return wrapper_func

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.storage, "contexts"):
            cls.storage.contexts = []
        return cls.storage.contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            return None

class PHMLayer(nn.Module):
    """
    This class is adapted from the compacter implementation at https://github.com/rabeehk/compacter
    """

    def __init__(
        self,
                 in_features: int,
                 out_features: int,
                 phm_dim: int,
                 phm_rule: Union[None, torch.Tensor] = None,
                 bias: bool = True,
                 w_init: str = "phm",
                 c_init: str = "random",
                 learn_phm: bool = True,
                 shared_phm_rule=False,
                 factorized_phm=False,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank = 1,
                 phm_init_range=0.0001,
                 kronecker_prod=False
    ) -> None:
        super(PHMLayer, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        assert c_init in ["normal", "uniform"]
        assert in_features % phm_dim == 0, f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert out_features % phm_dim == 0, f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod=kronecker_prod
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm_rule = factorized_phm_rule 
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1),
                       requires_grad=learn_phm)
                self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim),
                       requires_grad=learn_phm)
            else:
                self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim), 
                       requires_grad=learn_phm)
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.shared_W_phm = shared_W_phm 
        self.factorized_phm_W = factorized_phm
        if not self.shared_W_phm:
            if self.factorized_phm_W:
                self.W_left = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self.phm_rank)),
                              requires_grad=True)
                self.W_right = nn.Parameter(torch.Tensor(size=(phm_dim, self.phm_rank, self._out_feats_per_axis)),
                              requires_grad=True)
            else:
                self.W = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)),
                              requires_grad=True)
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def init_W(self, W_left=None, W_right=None, W=None):
        """
        Initialize the weights for the compacter module or the shared parameters
        """
        if self.factorized_phm_W:
            W_left = W_left
            W_right = W_right
        else:
            W = W
        if self.w_init == "glorot-normal":
            if self.factorized_phm_W:
                for i in range(self.phm_dim):
                    W_left.data[i] = nn.init.xavier_normal_(W_left.data[i])
                    W_right.data[i] = nn.init.xavier_normal_(W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    W.data[i] = nn.init.xavier_normal_(W.data[i])
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm_W:
                for i in range(self.phm_dim):
                    W_left.data[i] = nn.init.xavier_uniform_(W_left.data[i])
                    W_right.data[i] = nn.init.xavier_uniform_(W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    W.data[i] = nn.init.xavier_uniform_(W.data[i])
        elif self.w_init == "normal":
            if self.factorized_phm_W:
                for i in range(self.phm_dim):
                    W_left.data[i].normal_(mean=0, std=self.phm_init_range)
                    W_right.data[i].normal_(mean=0, std=self.phm_init_range)
            else:
                for i in range(self.phm_dim):
                    W.data[i].normal_(mean=0, std=self.phm_init_range)
        else:
            raise ValueError

    def _init_W(self, W_left=None, W_right=None, W=None):
        if self.factorized_phm_W:
            W_left = W_left if W_left is not None else self.W_left
            W_right = W_right if W_right is not None else self.W_right
            return self.init_W(W_left, W_right, W)
        else:
            W = W if W is not None else self.W
            return self.init_W(W_left, W_right, W)

    def reset_parameters(self):
        if not self.shared_W_phm:
            self._init_W()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)

        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                if self.c_init == "uniform":
                    self.phm_rule_left.data.uniform_(-0.01, 0.01)
                    self.phm_rule_right.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                    self.phm_rule_left.data.normal_(std=0.01)
                    self.phm_rule_right.data.normal_(std=0.01)
                else:
                    raise NotImplementedError
            else:
                if self.c_init == "uniform":
                    self.phm_rule.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                    self.phm_rule.data.normal_(mean=0, std=0.01)
                else:
                    raise NotImplementedError

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """
        If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right phm rules, and if this is not
        set, this is showing the phm_rule.
        """
        if self.factorized_phm_rule:
            self.phm_rule_left = phm_rule_left
            self.phm_rule_right = phm_rule_right
        else:
            self.phm_rule = phm_rule

    def set_W(self, W=None, W_left=None, W_right=None):
        if self.factorized_phm_W:
            self.W_left = W_left
            self.W_right = W_right
        else:
            self.W = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_W_phm:
            parameters = ForwardContext.get_context().shared_parameters[self.name]
            if self.factorized_phm_W:
                W = torch.bmm(parameters[f"W_{self.position}_left"], parameters[f"W_{self.position}_right"])
            else:
                W = parameters[f"W_{self.position}"]
        else:
            if self.factorized_phm_W:
                W = torch.bmm(self.W_left, self.W_right)
            else:
                W = self.W
        if self.shared_phm_rule:
            parameters = ForwardContext.get_context().shared_parameters[self.name]
            if self.factorized_phm_rule:
                phm_rule = torch.bmm(parameters["phm_rule_left"], parameters["phm_rule_right"])
            else:
                phm_rule = parameters["phm_rule"]
        else:
            if self.factorized_phm_rule:
                phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
            else:
                phm_rule = self.phm_rule

        H = kronecker_product(phm_rule, W).sum(0)

        y = torch.matmul(input=x, other=H)
        if self.b is not None:
            y += self.b
        return y


class PHMLinear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 phm_dim: int,
                 phm_rule: Union[None, torch.Tensor] = None,
                 bias: bool = True,
                 w_init: str = "phm",
                 c_init: str = "random",
                 learn_phm: bool = True,
                 shared_phm_rule=False,
                 factorized_phm=False,
                 shared_W_phm=False,
                 factorized_phm_rule=False,
                 phm_rank = 1,
                 phm_init_range=0.0001,
                 kronecker_prod=False) -> None:
        super(PHMLinear, self).__init__()
        assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
        assert c_init in ["normal", "uniform"]
        assert in_features % phm_dim == 0, f"Argument `in_features`={in_features} is not divisble be `phm_dim`{phm_dim}"
        assert out_features % phm_dim == 0, f"Argument `out_features`={out_features} is not divisble be `phm_dim`{phm_dim}"
        self.in_features = in_features
        self.out_features = out_features
        self.learn_phm = learn_phm
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = out_features // phm_dim
        self.phm_rank = phm_rank
        self.phm_init_range = phm_init_range
        self.kronecker_prod=kronecker_prod
        self.shared_phm_rule = shared_phm_rule
        self.factorized_phm_rule = factorized_phm_rule 
        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                self.phm_rule_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1),
                       requires_grad=learn_phm)
                self.phm_rule_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim),
                       requires_grad=learn_phm)
            else:
                self.phm_rule = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, phm_dim), 
                       requires_grad=learn_phm)
        self.bias_flag = bias
        self.w_init = w_init
        self.c_init = c_init
        self.shared_W_phm = shared_W_phm 
        self.factorized_phm = factorized_phm
        if not self.shared_W_phm:
            if self.factorized_phm:
                self.W_left = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self.phm_rank)),
                              requires_grad=True)
                self.W_right = nn.Parameter(torch.Tensor(size=(phm_dim, self.phm_rank, self._out_feats_per_axis)),
                              requires_grad=True)
            else:
                self.W = nn.Parameter(torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self._out_feats_per_axis)),
                              requires_grad=True)
        if self.bias_flag:
            self.b = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def init_W(self):
        if self.w_init == "glorot-normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_normal(self.W_left.data[i])
                    self.W_right.data[i] = glorot_normal(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_normal(self.W.data[i])
        elif self.w_init == "glorot-uniform":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i] = glorot_uniform(self.W_left.data[i])
                    self.W_right.data[i] = glorot_uniform(self.W_right.data[i])
            else:
                for i in range(self.phm_dim):
                    self.W.data[i] = glorot_uniform(self.W.data[i])
        elif self.w_init == "normal":
            if self.factorized_phm:
                for i in range(self.phm_dim):
                    self.W_left.data[i].normal_(mean=0, std=self.phm_init_range)
                    self.W_right.data[i].normal_(mean=0, std=self.phm_init_range)
            else:
                for i in range(self.phm_dim):
                    self.W.data[i].normal_(mean=0, std=self.phm_init_range)
        else:
            raise ValueError

    def reset_parameters(self):
        if not self.shared_W_phm:
           self.init_W()

        if self.bias_flag:
            self.b.data = torch.zeros_like(self.b.data)

        if not self.shared_phm_rule:
            if self.factorized_phm_rule:
                if self.c_init == "uniform":
                   self.phm_rule_left.data.uniform_(-0.01, 0.01)
                   self.phm_rule_right.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                   self.phm_rule_left.data.normal_(std=0.01)
                   self.phm_rule_right.data.normal_(std=0.01)
                else:
                   raise NotImplementedError
            else:
                if self.c_init == "uniform":
                   self.phm_rule.data.uniform_(-0.01, 0.01)
                elif self.c_init == "normal":
                   self.phm_rule.data.normal_(mean=0, std=0.01)
                else:
                   raise NotImplementedError

    def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
        """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
        phm rules, and if this is not set, this is showing  the phm_rule."""
        if self.factorized_phm_rule:
            self.phm_rule_left = phm_rule_left
            self.phm_rule_right = phm_rule_right 
        else:
            self.phm_rule = phm_rule 
 
    def set_W(self, W=None, W_left=None, W_right=None):
        if self.factorized_phm:
            self.W_left = W_left
            self.W_right = W_right
        else:
            self.W = W

    def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None) -> torch.Tensor:
        if self.factorized_phm:
           W = torch.bmm(self.W_left, self.W_right)
        if self.factorized_phm_rule:
           phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
        return matvec_product(
               W=W if self.factorized_phm else self.W,
               x=x,
               bias=self.b,
               phm_rule=phm_rule if self.factorized_phm_rule else self.phm_rule, 
               kronecker_prod=self.kronecker_prod)
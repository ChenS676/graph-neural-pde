import torch
from torch import nn
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
from utils import get_rw_adj
from torchdiffeq import odeint

class MixedODEblock(ODEblock):
  def __init__(self, 
               odefunc, 
               opt, 
               data, 
               device, 
               t=torch.tensor([0, 1]), 
               gamma=0.):
    super(MixedODEblock, self).__init__(odefunc, opt, data, device, t)
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], 
                           self.aug_dim * opt['hidden_dim'], 
                           opt, 
                           data, 
                           device)
    edge_index, edge_weight = get_rw_adj(data.edge_index, 
                                         edge_weight=data.edge_attr, 
                                         norm_dim=1,
                                         fill_value=opt['self_loop_weight'],
                                         num_nodes=data.num_nodes,
                                         dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.gamma = nn.Parameter(gamma * torch.ones(1))
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], 
                                                          opt['hidden_dim'], 
                                                          opt,
                                                          device).to(device)

  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def get_mixed_attention(self, x):
    gamma = torch.sigmoid(self.gamma)
    attention = self.get_attention_weights(x)
    # gamma balance between original edge weight and attention
    mixed_attention = attention.mean(dim=1) * (1 - gamma) + self.odefunc.edge_weight * gamma
    return mixed_attention

  def forward(self, x):
    t = self.t.type_as(x)
    self.odefunc.attention_weights = self.get_mixed_attention(x) #\bar A(X)
    integrator = self.train_integrator if self.training else self.test_integrator

    z = integrator(
      self.odefunc, x, t,
      method=self.opt['method'],
      options={'step_size': self.opt['step_size']},
      atol=self.atol,
      rtol=self.rtol)[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"

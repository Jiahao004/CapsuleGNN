import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.nn import GCNConv

epsilon = 1e-7


class ActivationFunc(nn.Module):
    def __init__(self):
        super(ActivationFunc, self).__init__()
        self.func = nn.Tanh()

    def forward(self, x):
        return self.func(x)


class Model(nn.Module):
    def __init__(self, n_class, node_dim, gnn_dim, n_gnn_layers,
                 n_prim_caps, n_digit_caps, caps_dim, n_caps_layers, gnn_channel, n_routing_iters,
                 is_precaps_share=True, dropout_p=0.1):
        super(Model, self).__init__()
        self.gnn_channel = gnn_channel
        self.gnn_dim = gnn_dim
        self.caps_dim = caps_dim
        self.n_class = n_class
        self.n_prim_caps = n_prim_caps
        assert gnn_dim // gnn_channel == caps_dim
        self.gnn_net = GNNNet(node_dim, gnn_dim, n_gnn_layers, dropout_p)
        self.capsule_net = CapsuleNet(n_prim_caps, n_digit_caps, caps_dim, n_caps_layers, gnn_channel * n_gnn_layers,
                                      n_routing_iters, is_precaps_share)
        self.cls_net = DigitCapsuleLayer(n_caps_layers * n_digit_caps * gnn_channel, n_class, caps_dim, 1,
                                         n_routing_iters,
                                         is_precaps_share)
        self.reconst_layer = nn.Sequential(nn.Linear(caps_dim, caps_dim),
                                           ActivationFunc(),
                                           nn.Linear(caps_dim, caps_dim))
        for p in self.reconst_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.criterion = ModelLoss(lambad_val=0.5)
        self.lb = LabelBinarizer()
        self.lb.fit(list(range(n_class)))

    def infer(self, x, edge_index, batch):
        gcn_output = self.gnn_net(x, edge_index)  # [n_nodes, n_layers, d_gcn]
        nodes = collect_nodes(gcn_output, batch)  # a list of nodes tensors: [n_nodes_in_graph, n_layers, d_gcn]

        nodes = batch_padding(nodes)  # [batch, n_nodes, n_layers, d_gcn]
        nodes = nodes.unflatten(dim=-1, sizes=(self.gnn_channel, self.caps_dim))
        # [batch, n_nodes, n_layers, channel, d_caps]
        nodes = nodes.flatten(2, 3)  # [batch, n_nodes, n_layers*channel, d_caps]

        caps_output = self.capsule_net(nodes)  # [batch, n_caps_layers, n_digit_caps, channel, d_model]
        caps_output = caps_output.flatten(1, 3).unsqueeze(2)  # [batch, n_caps_layers*n_digit_caps*channel, 1, d_model]
        assert isinstance(caps_output, torch.Tensor)
        predicted = self.cls_net(caps_output)  # [batch, n_class, 1, d_caps]
        return predicted.squeeze(2)

    def forward(self, x, edge_index, batch, y):
        predicted = self.infer(x, edge_index, batch)
        batch_size = predicted.size(0)
        assert isinstance(predicted, torch.Tensor)
        val_predicted = torch.norm(predicted, p=2, dim=-1)

        correct_predicted = []
        for i in range(batch_size):
            correct_predicted.append(predicted[i, y[i]])
        correct_predicted = torch.stack(correct_predicted, dim=0)  # [batch, d_caps]

        assert isinstance(predicted, torch.Tensor)
        pred_hist = self.reconst_layer(correct_predicted)

        class_nodes = collect_nodes(x, batch)
        # a list of nodes tensors: (n_graphs_in_batch, [n_nodes_in_graph, d_nodes])

        tgt_hist = batch_histogram(class_nodes)
        labels = torch.tensor(self.lb.transform(list(y.cpu()))).to(x.device)

        loss = self.criterion(tgt_hist, pred_hist, labels, val_predicted)
        return loss


class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=0.1):
        super(GNNLayer, self).__init__()
        self.gcn = GCNConv(input_dim, output_dim)
        self.activation_func = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index):
        return self.dropout(self.activation_func(self.gcn(x, edge_index)))


class GNNNet(nn.Module):
    def __init__(self, node_dim, gcn_dim, n_gcn_layers, dropout_p):
        super(GNNNet, self).__init__()

        self.d_model = gcn_dim
        self.dropout_p = dropout_p
        self.layers = nn.ModuleList([GNNLayer(node_dim, gcn_dim, dropout_p)] +
                                    [GNNLayer(gcn_dim, gcn_dim, dropout_p) for _ in range(n_gcn_layers - 1)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index):
        # shape of x: [n_nodes, d_model]
        # shape of edge_index: [2, n_edges]
        res = []
        output = x
        for layer in self.layers:
            output = layer(output, edge_index)  # [n_nodes, d_model]
            res.append(output)  # [n_nodes, d_model]
        res = torch.stack(res, dim=1)  # [n_nodes, n_layers, d_nodes]
        return res


class CapsuleNet(nn.Module):
    def __init__(self, n_prim_caps, n_digit_caps, d_model, n_layers, channel, n_routing_iters, is_precaps_share):
        super(CapsuleNet, self).__init__()

        self.prim_layer = CapsuleNetLayer(PrimaryCapsuleLayer(),
                                          d_model, channel)

        self.pre_layer = CapsuleNetLayer(DigitCapsuleLayer(n_prim_caps, n_digit_caps, d_model, channel, n_routing_iters,
                                                           is_precaps_share),
                                         d_model, channel)

        self.layers = nn.ModuleList([CapsuleNetLayer(DigitCapsuleLayer(n_digit_caps, n_digit_caps, d_model, channel,
                                                                       n_routing_iters, is_precaps_share),
                                                     d_model, channel)
                                     for _ in range(n_layers - 1)])

    def forward(self, x):
        # x: [batch, num_prim_cap, channel, d_nodes]
        res = []
        output = self.prim_layer(x)
        output = self.pre_layer(output)  # [batch, n_digit_caps, channel, d_model]
        res.append(output)
        for layer in self.layers:
            output = layer(output)
            res.append(output)
        res = torch.stack(res, dim=1)
        return res  # [batch, n_caps_layers, n_digit_caps, channel, d_model]


class CapsuleNetLayer(nn.Module):
    def __init__(self, capsule_layer, d_model, channel=1):
        super(CapsuleNetLayer, self).__init__()
        self.layer = capsule_layer
        self.attn_layer = AttentionModule(channel, d_model)

    def forward(self, x):
        return self.attn_layer(self.layer(x))


class PrimaryCapsuleLayer(nn.Module):
    def __init__(self):
        super(PrimaryCapsuleLayer, self).__init__()

    def forward(self, x):
        # x: [batch, n_nodes, channel, d_nodes]
        output = squash(x)
        return output


class DigitCapsuleLayer(nn.Module):
    def __init__(self, n_pre_caps, n_digit_caps, d_model, channel, n_routing_iters, is_precaps_share=True):
        super(DigitCapsuleLayer, self).__init__()
        assert n_routing_iters > 0

        self.n_digit_caps = n_digit_caps
        self.n_routing_iters = n_routing_iters
        self.is_precaps_share = is_precaps_share
        self.channel = channel

        if is_precaps_share:
            self.w = Parameter(torch.randn(1, 1, n_digit_caps, channel, d_model, d_model))
            # [batch, 1, n_digit_caps, channel, d_model, d_model]
        else:
            self.w = Parameter(torch.randn(1, n_pre_caps, n_digit_caps, channel, d_model, d_model))
            # [batch, n_pre_caps, n_digit_caps, channel, d_model, d_model]

    def forward(self, x):
        # x: [batch, n_pre_caps, channel, d_model]
        x = x[:, :, None, :, :, None]  # [batch, n_pre_caps, 1, channel, d_model, 1]
        u_hat = torch.matmul(self.w.to(x.device), x).squeeze(-1)  # [batch, n_pre_caps, n_digit_caps, channel, d_model]
        return self.dynamic_routing(u_hat)

    def dynamic_routing(self, u_hat):
        # u_hat: [batch, n_pre_caps, n_digit_caps, channel, d_model]
        # b: [batch, 1 or n_pre_caps, n_digit_caps, channel]
        tmp_u_hat = u_hat.detach()

        b = torch.zeros(1, u_hat.size(1), self.n_digit_caps, self.channel, requires_grad=False).to(u_hat.device)
        for i in range(self.n_routing_iters - 1):
            c = torch.softmax(b, dim=2).unsqueeze(-1)  # [batch, n_pre_caps, n_digit_caps, channel, 1]
            # tmp_u_hat: [batch, 1 or n_pre_caps, n_digit_caps, channel, d_model]
            s = torch.sum(c * tmp_u_hat, dim=1)  # [batch, n_digit_caps, channel, d_model]
            v = squash(s)
            a = torch.matmul(tmp_u_hat[:, :, :, :, None, :], v[:, None, :, :, :, None])
            # [batch, n_pre_caps, n_digit_caps, channel, 1, 1]
            b = b + a.squeeze(-1).squeeze(-1)

        c = torch.softmax(b, dim=2).unsqueeze(-1)
        s = torch.sum(c * u_hat, dim=1)
        v = squash(s)

        return v  # [batch, n_digit_caps, channel, d_model]


class AttentionModule(nn.Module):
    def __init__(self, channel, dim):
        super(AttentionModule, self).__init__()
        self.attn = nn.Sequential(nn.Linear(channel * dim, channel * dim),
                                  ActivationFunc(),
                                  nn.Linear(channel * dim, channel))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        ''' x: [batch, num_caps, channel, dim]'''
        assert isinstance(x, torch.Tensor)
        weights = self.attn(x.flatten(start_dim=-2))  # [b, n, c]
        weights = torch.softmax(weights, dim=2)[:, :, :, None]
        return weights * x


def squash(input_tensor, dim=-1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


def collect_nodes(x, batch):
    batch_size = int(batch.max() + 1)
    res = []
    # x: [n_nodes,...,?]
    for b in range(batch_size):
        res.append(x[batch == b])
    return res


def batch_padding(x, n_prim_caps=None):
    # x is a list of tensors, each tensors is the node features in a graph
    # x: (batch, [n_nodes, n_layers, d_nodes])
    node_featrue_size = x[0][0].size()
    max_n_nodes = max(len(xx) for xx in x)
    for i in range(len(x)):
        if n_prim_caps:
            if len(x[i]) <= n_prim_caps:
                pad_tensor = torch.zeros([n_prim_caps - len(x[i])] + list(node_featrue_size), device=x[0].device)
                x[i] = torch.cat([x[i], pad_tensor], dim=0)
            else:
                x[i] = x[i, :n_prim_caps]
        else:
            pad_tensor = torch.zeros([max_n_nodes - len(x[i])] + list(node_featrue_size), device=x[0].device)
            x[i] = torch.cat([x[i], pad_tensor], dim=0)
    res = torch.stack(x, dim=0).contiguous()
    return res


def batch_histogram(x):
    # x: a batch of graph tensors(batch, [n_nodes_in_graph, node_dim])
    res = []
    for xx in x:
        assert isinstance(xx, torch.Tensor)  # xx :[n_nodes_in_graph, node_dim]
        res.append((xx > 0).sum(dim=-2))
    return torch.stack(res, dim=0)


class ModelLoss(nn.Module):
    def __init__(self, lambad_val, m_plus=0.9, m_minus=0.1):
        super(ModelLoss, self).__init__()
        self.lambda_val = lambad_val
        self.m_plus = m_plus
        self.m_minus = m_minus

    def forward(self, tgt_hist, pred_hist, labels, val_c):
        """
        :param tgt_hist: [batch, d_nodes]
        :param pred_hist: [batch, d_nodes]
        :param labels: [batch, n_class]
        :param val_c: [batch, n_class]
        :return:
        """
        present_error = F.relu(self.m_plus - val_c, inplace=True) ** 2  # max(0, m_plus-||v_c||)^2
        absent_error = F.relu(val_c - self.m_minus, inplace=True) ** 2  # max(0, ||v_c||-m_minus)^2

        l_c = labels.float() * present_error + self.lambda_val * (1. - labels.float()) * absent_error
        margin_loss = l_c.sum(dim=1).mean()

        assert isinstance(tgt_hist, torch.Tensor)
        tgt_hist = tgt_hist / tgt_hist.max(dim=-1, keepdim=True)[0]
        mp = (tgt_hist > epsilon).int()  # [batch, d_nodes]
        diff = (pred_hist - tgt_hist) ** 2

        reconst_loss = torch.sum(mp * diff, dim=-1) / (epsilon + mp.sum(dim=-1)) + \
                       torch.sum((1 - mp) * diff, dim=-1) / (1 - mp + epsilon).sum(dim=-1)

        loss = margin_loss + reconst_loss.mean()
        return loss

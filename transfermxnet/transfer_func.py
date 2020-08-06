from mxnet import nd,init,gluon,autograd,np,npx
from mxnet.gluon import loss as gloss
from transfermxnet.d2l import mxnet as d2l
from mxnet.gluon import nn
import csv
npx.set_np()
import math

def transpose_qkv(X, num_heads):
    # Input `X` shape: (`batch_size`, `seq_len`, `num_hiddens`).
    # Output `X` shape:
    # (`batch_size`, `seq_len`, `num_heads`, `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # `X` shape:
    # (`batch_size`, `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
    X = X.transpose(0, 2, 1, 3)

    # `output` shape:
    # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output


def transpose_output(X, num_heads):
    # A reversed version of `transpose_qkv`
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, query, key, value, valid_len):
        # For self-attention, `query`, `key`, and `value` shape:
        # (`batch_size`, `seq_len`, `dim`), where `seq_len` is the length of
        # input sequence. `valid_len` shape is either (`batch_size`, ) or
        # (`batch_size`, `seq_len`).

        # Project and transpose `query`, `key`, and `value` from
        # (`batch_size`, `seq_len`, `num_hiddens`) to
        # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)

        # print('self.W_q(query)', self.W_q(query).shape)  ####(128, 1)
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_len is not None:
            # Copy `valid_len` by `num_heads` times
            if valid_len.ndim == 1:
                valid_len = np.tile(valid_len, self.num_heads)
            else:
                valid_len = np.tile(valid_len, (self.num_heads, 1))

        # For self-attention, `output` shape:
        # (`batch_size` * `num_heads`, `seq_len`, `num_hiddens` / `num_heads`)
        output = self.attention(query, key, value, valid_len)

        # `output_concat` shape: (`batch_size`, `seq_len`, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Block):
    def __init__(self, ffn_num_hiddens, pw_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(pw_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))


#@save
class AddNorm(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        # print(X.shape,self.P[:, :X.shape[1], :].as_in_ctx(X.ctx).shape)       ##(128, 10, 16) (1, 10, 16)
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)



def class_and_score_forward(x):
    class_part = nd.slice_axis(x,begin=0,end=3,axis=-1)
    concentration_part = nd.slice_axis(x,begin=3,end=5,axis=-1)

    class_part = nd.sigmoid(class_part)
    concentration_part = nd.sigmoid(concentration_part)
    return class_part,concentration_part

def concentration_transfer(class_pre_l,class_true_l,con_pre_l,con_true_l,data_utils):
    eth_co_me_limit = nd.array([[data_utils.scale_CO[1], data_utils.scale_CO[0], data_utils.scale_Me[0]]])
    concentration_mat_pre = nd.where(class_pre_l > 0.5,
                                     nd.repeat(eth_co_me_limit, repeats=class_pre_l.shape[0], axis=0), \
                                     nd.zeros_like(class_pre_l))
    concentration_mat_true = nd.where(class_true_l == 1,
                                      nd.repeat(eth_co_me_limit, repeats=class_true_l.shape[0], axis=0), \
                                      nd.zeros_like(class_true_l))

    eth_con_pre, eth_con_true = concentration_mat_pre[:, 0] * con_pre_l[:, 1], concentration_mat_true[:,
                                                                               0] * con_true_l[:, 1]
    co_con_pre, co_con_true = concentration_mat_pre[:, 1] * con_pre_l[:, 0], concentration_mat_true[:, 1] * con_true_l[
                                                                                                            :, 0]
    me_con_pre, me_con_true = concentration_mat_pre[:, 2] * con_pre_l[:, 0], concentration_mat_true[:, 2] * con_true_l[
                                                                                                            :, 0]

    eth_co_me_con_pre = nd.concat(nd.expand_dims(eth_con_pre, axis=0), nd.expand_dims(co_con_pre, axis=0), \
                                  nd.expand_dims(me_con_pre, axis=0), dim=0).transpose()
    eth_co_me_con_true = nd.concat(nd.expand_dims(eth_con_true, axis=0), nd.expand_dims(co_con_true, axis=0), \
                                   nd.expand_dims(me_con_true, axis=0), dim=0).transpose()
    return eth_co_me_con_pre,eth_co_me_con_true


def results_writer(fw,class_pres,class_trues,concentration_pres,concentration_trues):

    all_class_pres_list = class_pres.asnumpy().tolist()
    all_class_trues_list = class_trues.asnumpy().tolist()
    all_con_pres_list = concentration_pres.asnumpy().tolist()
    all_con_trues_list = concentration_trues.asnumpy().tolist()

    csv_writer = csv.writer(fw, dialect='excel')
    for i in range(class_pres.shape[0]):
        content = all_class_pres_list[i] + ["   "] + \
                  all_class_trues_list[i] + ["  | "] + \
                  all_con_pres_list[i] + ["  "] + \
                  all_con_trues_list[i]
        csv_writer.writerow(content)

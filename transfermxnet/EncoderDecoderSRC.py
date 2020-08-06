from transfermxnet.transfer_func import MultiHeadAttention,AddNorm,PositionalEncoding,PositionWiseFFN
from mxnet import autograd,np
from transfermxnet.d2l import mxnet as d2l
from mxnet.gluon import nn
import math



class EncoderBlock(nn.Block):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout,
                                            use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_len):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_len))
        return self.addnorm2(Y, self.ffn(Y))


class TFEncoder(d2l.Encoder):
    def __init__(self,num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TFEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout,
                             use_bias))

    def forward(self, X, valid_len, *args):
        # X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # print(X.shape,valid_len)        ##(128, 10, 16)
        # exit()
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X,valid_len)
        return X



def fully_connect(num_layers,num_hiddens_fully,outSize):        #全链接
    model = nn.Sequential()
    for _ in range(num_layers):
        model.add(nn.Dense(num_hiddens_fully, activation='tanh', use_bias=False,
                           flatten=False))
    model.add(nn.Dense(outSize, use_bias=False, flatten=False))
    return model

class TFDecoder(nn.Block):
    def __init__(self,num_layers,num_hiddens_fully,outSize, **kwargs):
        super(TFDecoder, self).__init__(**kwargs)
        self.fully_connect = fully_connect(num_layers,num_hiddens_fully,outSize)
        # self.out = nn.Dense(outSize, flatten=False)
    #
    # def init_state(self, enc_state, *args):
    #     # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
    #     return enc_state

    def forward(self,dec_states):
        output = self.fully_connect(dec_states)

        return output


class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        # print(enc_outputs.shape)         ##(128, 10, 16)(128, 10, 16)(30, 10, 16)
        # dec_state = self.decoder.init_state(enc_outputs, *args)
        enc_outputs=enc_outputs.reshape(enc_X.shape[0],-1)
        # print(enc_outputs.shape)         ##(128, 1600)
        # exit()
        return self.decoder(enc_outputs)






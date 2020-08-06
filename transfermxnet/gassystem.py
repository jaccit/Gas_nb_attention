from src_new.reader_new import Data_utils
from transfermxnet.EncoderDecoderSRC import TFEncoder,TFDecoder,EncoderDecoder
from transfermxnet.d2l import mxnet as d2l
from transfermxnet import TrainPredict,transfer_func
import argparse
from mxnet import np, npx
npx.set_np()



###basic configuration###

parser = argparse.ArgumentParser()
parser.add_argument("--datafile0",default="C:/MixtureGas326/data/ethylene_CO-1.txt")
parser.add_argument("--datafile1",default="C:/MixtureGas326/data/ethylene_methane-1.txt")

parser.add_argument("--num_step",default=100,type=int)           #截取的时间点time step
parser.add_argument("--time_scale",default=10,type=int)          #降采样
parser.add_argument("--train_scale",default=0.8,type=int)          #划分测试集大小
parser.add_argument("--datasize_share",default=1,type=int)          #取整个数据集部分的大小


parser.add_argument("--num_hiddens",default=16,type=int)              #number of sensors
parser.add_argument("--ffn_num_hiddens",default=32,type=int)           #PositionWiseFFN
parser.add_argument("--num_heads",default=4,type=int)                 #number of MultiHeads of Attention
parser.add_argument("--encnum_layers",default=1,type=int)
parser.add_argument("--decnum_layers",default=1,type=int)
parser.add_argument("--fully_unit",default=50,type=int)
parser.add_argument("--out_size",default=5,type=int)
parser.add_argument("--dropout",default=0.0,type=float)


args = parser.parse_args()

###Train Hyperparameter####
batch_size, num_epochs = 128, 100
lr, ctx = 0.005, d2l.try_gpu()


if __name__=="__main__":
    Gas_data = Data_utils(filename0=args.datafile0, filename1=args.datafile1, num_step=args.num_step,
                          time_scale=args.time_scale, train_scale=args.train_scale,
                          datasize_share=args.datasize_share)

    encoder = TFEncoder(num_hiddens=args.num_hiddens, ffn_num_hiddens=args.ffn_num_hiddens,
                 num_heads=args.num_heads, num_layers=args.encnum_layers, dropout=args.dropout)

    decoder = TFDecoder(num_layers=args.decnum_layers,num_hiddens_fully=args.fully_unit,outSize=args.out_size)

    model = EncoderDecoder(encoder, decoder)


    TrainPredict.train_GAS_ch9(model, Gas_data, batch_size, lr, num_epochs, ctx)

    TrainPredict.Test_writeresult(model,Gas_data,ctx)



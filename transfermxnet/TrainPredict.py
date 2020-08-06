from mxnet import nd,init,gluon,autograd,np,npx
from mxnet.gluon import loss as gloss
from transfermxnet.d2l import mxnet as d2l
from transfermxnet.transfer_func import class_and_score_forward,concentration_transfer,results_writer
npx.set_np()

def train_GAS_ch9(model, data_utils,batch_size, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam', {'learning_rate': lr})
    loss = d2l.MaskedSoftmaxCELoss()
    loss1 = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    loss2 = gloss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], ylim=[0, 0.25])
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        l_sum, l_class_sum, l_score_sum, n, acc_sum = 0.0, 0.0, 0.0, 0, 0.0
        rmse_sum = nd.array([0, 0, 0])
        data_iter = data_utils.get_batch_train(batch_size)
        # ti=0

        for i, (X, Y) in enumerate(data_iter):
            # print(X.shape,Y.shape)               ##X=(128, 10, 16) (128, 5)
            # exit()
            X, Y = nd.array(X).as_np_ndarray(), nd.array(Y).as_np_ndarray()
            # print("after reshape",X.shape, Y.shape)         ##after reshape (128, 10, 16) (128, 5)
            # exit()
            # vlinz=nd.random.randint(0,10,(X.shape[0],)).as_np_ndarray()
            # vlinz=nd.ones((X.shape[0],)).as_np_ndarray()
            valid_len = np.repeat(np.array([X.shape[1]]), X.shape[0])
            # print(valid_len.shape)                       ##(128,)
            # exit()
            # ti+=1
            # print('keepup',ti)
            with autograd.record():
                dec_output= model(X,valid_len)
                # print(dec_output.shape)             ##(128, 10, 5)
                # exit()
                #         ###################
        #         l = loss(dec_output, Y, vlinz)
        #     l.backward()
        #     d2l.grad_clipping(model, 1)
        #     num_tokens = vlinz.sum()
        #     trainer.step(num_tokens)
        #     metric.add(l.sum(), num_tokens)
        #     # exit()
        # if epoch % 10 == 0:
        #     animator.add(epoch, (metric[0] / metric[1],))
        # print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        #       f'tokens/sec on {str(ctx)}')
        #         ##########################
                output=dec_output.as_nd_ndarray()
                cl_res, score_res = class_and_score_forward(output)
                # print("shape of cl_res:",cl_res.shape,"shape of Y[0][:,:3]:",Y.shape)
                # print("shape of score_res:",score_res.shape,"shape of Y[0][:,3:]:",Y.shape)
                cl_weight, conc_weight = nd.ones_like(cl_res), nd.ones_like(score_res)
                l_class = loss1(cl_res.as_np_ndarray(), Y[:, :3], cl_weight.as_np_ndarray()).sum()
                l_conc  = loss2(score_res.as_np_ndarray(), Y[:, 3:], conc_weight.as_np_ndarray()).sum()
                n = Y.shape[0]
                l = (l_class/n)+(l_conc/n)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = n
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if epoch % 10 == 0:
            animator.add(epoch, (metric[0]/metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(ctx)}')
#####################################################




def Test_writeresult(model,data_utils,ctx):
    cl_pres, cl_trues = [], []
    con_pres, con_trues = [], []
    test_data = data_utils.get_batch_test(batch_size=100)
    for X, Y in test_data:
        X, Y = nd.array(X, ctx=ctx).as_np_ndarray(), nd.array(Y, ctx=ctx).as_np_ndarray()
        valid_len = np.repeat(np.array([X.shape[1]]), X.shape[0])
        predict_result= model(X,valid_len)
        cl_res, score_res = class_and_score_forward(predict_result.as_nd_ndarray())
        cl_pres.append(cl_res)
        cl_trues.append(Y[:, :3].as_nd_ndarray())
        con_pres.append(score_res)
        con_trues.append(Y[:, 3:].as_nd_ndarray())
    all_class_pres = nd.concat(*cl_pres,dim=0)
    all_class_trues = nd.concat(*cl_trues,dim=0)
    all_con_pres = nd.concat(*con_pres, dim=0)
    all_con_trues = nd.concat(*con_trues, dim=0)
    # acc = get_accuracy(all_class_pres, all_class_trues).asscalar()
    test_pre_and_true_concentration = concentration_transfer(all_class_pres, all_class_trues, all_con_pres,
                                                             all_con_trues, data_utils)
    f = open("forattrecord.csv", "w", newline='')
    results_writer(f, all_class_pres, all_class_trues, \
                   test_pre_and_true_concentration[0], test_pre_and_true_concentration[1])
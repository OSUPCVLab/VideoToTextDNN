import common
import data_engine
import os
import time
import theano
import numpy
import theano.tensor as tensor

from common import init_tparams, load_params, itemlist, unzip, grad_nan_report
from common import adadelta, adam, rmsprop, sgd
from model_lstmdd import Attention as LSTMDD
from model_mtle import Attention as MTLE
from model_attention import Attention

from model_attention import validate_options


class LiveDataEngine(data_engine.Movie2Caption):
    def __init__(self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,dec,proc, data_dir):
        super(LiveDataEngine, self).__init__(model_type, signature, video_feature,
                                             mb_size_train, mb_size_test, maxlen,
                                             n_words,dec,proc, n_frames=None, outof=None,
                                             data_dir=data_dir, feats_dir='')

    def load_data(self):
        self.worddict = common.load_pkl(os.path.join(self.data_dir, 'worddict.pkl'))
        self.word_idict = dict()
        # wordict start with index 2
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'

        if self.video_feature == 'googlenet':
            self.ctx_dim = 1024
        elif self.video_feature == 'resnet' or self.video_feature == 'resnet152':
            if self.proc == 'nostd':
                self.ctx_dim = 2048
            elif self.proc == 'pca':
                self.ctx_dim = 1024
        elif self.video_feature == 'nasnetalarge':
            self.ctx_dim = 4032
        elif self.video_feature == 'pnasnet5large':
            self.ctx_dim = 4320
        elif self.video_feature == 'polynet':
            self.ctx_dim = 2048
        elif self.video_feature == 'senet154':
            self.ctx_dim = 2048
        elif self.video_feature == 'densenet121':
            raise NotImplementedError()
        elif self.video_feature == 'c3d':
            if self.proc == 'nostd':
                self.ctx_dim = 4101
            elif self.proc == 'pca':
                self.ctx_dim = 1024
        elif self.video_feature == 'c3d_resnet':
            if self.proc == 'nostd':
                self.ctx_dim = 6149
            elif self.proc == 'pca':
                self.ctx_dim = 2048
            elif self.proc == 'pca512':
                self.ctx_dim = 1024
            elif self.proc == 'pca_c3d':
                self.ctx_dim = 3072
        else:
            raise NotImplementedError()

        print "ctx_dim: " + str(self.ctx_dim)
        self.kf_train = []
        self.kf_valid = []
        self.kf_test = []

    def prepare_live_data(self, feat):
        seqs = []
        z_seqs = []
        feat_list = []

        def clean_sequences(seqs,z_seqs,feat_list):
            if self.dec=="standard":
                lengths = [len(s) for s in seqs]
                if self.maxlen != None:
                    new_seqs = []
                    new_feat_list = []
                    new_lengths = []
                    new_caps = []
                    for l, s, y, c in zip(lengths, seqs, feat_list, [[] for _ in range(len(feat_list))]):
                        # sequences that have length >= maxlen will be thrown away
                        if l < self.maxlen:
                            new_seqs.append(s)
                            new_feat_list.append(y)
                            new_lengths.append(l)
                            new_caps.append(c)
                    lengths = new_lengths
                    feat_list = new_feat_list
                    seqs = new_seqs

                return seqs,None,feat_list,lengths

            else:
                lengths = [len(s) for s in seqs]
                z_lengths = [len(s) for s in z_seqs]
                if self.maxlen != None:
                    new_seqs = []
                    new_zseqs = []
                    new_feat_list = []
                    new_lengths = []
                    new_caps = []
                    new_zlengths = []
                    for l,z_l, s, y, c in zip(lengths,z_lengths, seqs, feat_list, [[] for _ in range(len(feat))]):
                        # sequences that have length >= maxlen will be thrown away
                        if l < self.maxlen and z_l < self.maxlen :
                            new_seqs.append(s)
                            new_zseqs.append(s)
                            new_feat_list.append(y)
                            new_lengths.append(l)
                            new_caps.append(c)
                    lengths = new_lengths
                    feat_list = new_feat_list
                    seqs = new_seqs
                    z_seqs = new_zseqs

                return seqs,z_seqs,feat_list,lengths

        feat_list.append(feat)
        words = []
        seqs.append([])

        if self.dec != "standard":
            z_seq = []
            z_seqs.append(z_seq)

        seqs, z_seqs, feat_list, lengths = clean_sequences(seqs, z_seqs, feat_list)

        if len(lengths) < 1:
            return None, None, None, None

        y = numpy.asarray(feat_list)
        # print len(y[1,1])
        y_mask = self.get_ctx_mask(y)

        n_samples = len(seqs)
        maxlen = numpy.max(lengths) + 1

        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx] + 1, idx] = 1.

        if self.dec == "standard":
            return x, x_mask, y, y_mask
        else:
            z = numpy.zeros((maxlen, n_samples)).astype('int64')  # This is the other label
            z_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
            for idx, s in enumerate(z_seqs):
                z[:lengths[idx], idx] = s
                z_mask[:lengths[idx] + 1, idx] = 1.

            return x, x_mask, y, y_mask, z, z_mask


class LiveCaptioner(MTLE):
    """
    Class to init some model and provide it as a captioner to an upper layer. Based on model_*.predict fns
    """
    def __init__(self, checkpoint_dir):
        super(LiveCaptioner, self).__init__()
        self._init_model(checkpoint_dir)

    def _init_model(self, from_dir):

        options_path = os.path.join(from_dir, 'model_options.pkl')
        assert os.path.exists(options_path), 'No model_options.pkl file was found in checkpoint directory.'
        model_options = common.load_pkl(options_path)

        video_feature = model_options['video_feature']
        decay_c = model_options['decay_c']
        alpha_c = model_options['alpha_c']
        alpha_entropy_r = model_options['alpha_entropy_r']
        clip_c = model_options['clip_c']
        optimizer = model_options['optimizer']
        dec = model_options['dec']
        maxlen = model_options['maxlen']
        n_words = model_options['n_words']
        proc = model_options['proc']

        self.rng_numpy, self.rng_theano = common.get_two_rngs()

        if 'self' in model_options:
            del model_options['self']
        model_options = validate_options(model_options)
        # with open(os.path.join(save_model_dir, 'model_options.pkl'), 'wb') as f:
        #     pkl.dump(model_options, f)

        print 'Loading data'
        self.engine = LiveDataEngine('mtle', 'live', video_feature, 1, 1, maxlen, n_words, dec, proc, from_dir)
        model_options['ctx_dim'] = self.engine.ctx_dim
        self.model_options = model_options

        # set test values, for debugging
        # idx = self.engine.kf_train[0]
        # [self.x_tv, self.mask_tv,
        #  self.ctx_tv, self.ctx_mask_tv, self.y_tv, self.y_mask_tv] = data_engine.prepare_data(
        #     self.engine, [self.engine.test[index] for index in idx])

        print 'init params'
        t0 = time.time()
        params = self.init_params(model_options)

        model_saved = os.path.join(from_dir, 'model_best_so_far.npz')
        assert os.path.isfile(model_saved)
        print "Reloading model params..."
        params = load_params(model_saved, params)

        tparams = init_tparams(params)
        self.current_params = tparams

        self.trng, self.use_noise, \
        self.x, self.x_mask, self.ctx, self.mask_ctx, self.alphas, \
        self.cost, self.extra, self.y, self.y_mask = self.build_model(tparams, model_options)

        print 'buliding sampler'
        self.f_init, self.f_next = self.build_sampler(tparams, model_options, self.use_noise, self.trng)
        # before any regularizer
        print 'building f_log_probs'
        f_log_probs = theano.function([self.x, self.x_mask, self.ctx, self.mask_ctx, self.y, self.y_mask], -self.cost,
                                      profile=False, on_unused_input='ignore')

        cost = self.cost.mean()
        if decay_c > 0.:
            decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        if alpha_c > 0.:
            alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
            alpha_reg = alpha_c * ((1. - self.alphas.sum(0)) ** 2).sum(0).mean()
            cost += alpha_reg

        if alpha_entropy_r > 0:
            alpha_entropy_r = theano.shared(numpy.float32(alpha_entropy_r),
                                            name='alpha_entropy_r')
            alpha_reg_2 = alpha_entropy_r * (-tensor.sum(self.alphas *
                                                         tensor.log(self.alphas + 1e-8), axis=-1)).sum(0).mean()
            cost += alpha_reg_2
        else:
            alpha_reg_2 = tensor.zeros_like(cost)
        print 'building f_alpha'
        self.f_alpha = theano.function([self.x, self.x_mask, self.ctx, self.mask_ctx, self.y, self.y_mask],
                                  [self.alphas, alpha_reg_2],
                                  name='f_alpha',
                                  on_unused_input='ignore')

        print 'compute grad'
        grads = tensor.grad(cost, wrt=itemlist(tparams))
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g ** 2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        lr = tensor.scalar(name='lr')
        print 'build train fns'
        self.f_grad_shared, self.f_update = eval(optimizer)(lr, tparams, grads,
                                                  [self.x, self.x_mask, self.ctx, self.mask_ctx, self.y, self.y_mask], cost,
                                                  self.extra + grads)

        print 'compilation took %.4f sec' % (time.time() - t0)

    def caption(self, feat):
        # processes = None
        # queue = None
        # rqueue = None
        # shared_params = None
        #
        # uidx = 0
        # alphas_ratio = []
        #
        # n_samples = 0
        # train_costs = []
        # grads_record = []
        #
        # # tags = [self.engine.test[index] for index in idx]
        # # n_samples += len(tags)
        # uidx += 1
        # self.use_noise.set_value(1.)
        #
        # pd_start = time.time()
        #
        # x, x_mask, ctx, ctx_mask, y, y_mask = self.engine.prepare_live_data(feat)
        #
        # pd_duration = time.time() - pd_start
        # # if x is None:
        # #     print 'Minibatch with zero sample under length ', maxlen
        # #     continue
        #
        # ud_start = time.time()
        # rvals = self.f_grad_shared(x, x_mask, ctx, ctx_mask, y, y_mask)
        # cost = rvals[0]
        # probs = rvals[1]
        # alphas = rvals[2]
        # grads = rvals[3:]
        # # grads, NaN_keys = grad_nan_report(grads, tparams)
        # # if len(grads_record) >= 5:
        # #     del grads_record[0]
        # # grads_record.append(grads)
        # # if NaN_keys != []:
        # #     print 'grads contain NaN'
        # #     import pdb;
        # #     pdb.set_trace()
        # # if numpy.isnan(cost) or numpy.isinf(cost):
        # #     print 'NaN detected in cost'
        # #     import pdb;
        # #     pdb.set_trace()
        # # update params
        # self.f_update(0.0001)
        # ud_duration = time.time() - ud_start
        #
        # # if eidx == 0:
        # #     train_error = cost
        # # else:
        # #     train_error = train_error * 0.95 + cost * 0.05
        # train_costs.append(cost)
        #
        # t0_valid = time.time()
        # alphas, _ = self.f_alpha(x, x_mask, ctx, ctx_mask, y, y_mask)
        # ratio = alphas.min(-1).mean() / (alphas.max(-1)).mean()
        # alphas_ratio.append(ratio)
        # # numpy.savetxt(os.path.join(save_model_dir, 'alpha_ratio.txt'), alphas_ratio)
        #
        # # current_params = unzip(tparams)
        # # numpy.savez(
        # #     os.path.join(save_model_dir, 'model_current.npz'),
        # #     history_errs=history_errs, **current_params)
        #
        # self.use_noise.set_value(0.)
        # train_err = -1
        # train_perp = -1
        # valid_err = -1
        # valid_perp = -1
        # test_err = -1
        # test_perp = -1
        #
        # mean_ranking = 0
        # blue_t0 = time.time()
        # # scores, processes, queue, rqueue, shared_params = \

        feat_mask = self.engine.get_ctx_mask(feat)

        print("Sampling")
        sample, score, _, _ = self.gen_sample(None, self.f_init, self.f_next, feat,
                                              feat_mask, self.model_options, maxlen=50)

        return sample

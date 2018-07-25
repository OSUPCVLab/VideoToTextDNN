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

    def _seqs2words(self, cap):
        """
        Create string caption using generated index-based sample
        :param cap:
        :return:
        """
        ww = []
        for w in cap:
            if w == 0:
                break
            ww.append(self.engine.word_idict[1]
                      if w > len(self.engine.word_idict) else self.engine.word_idict[w])

        return ' '.join(ww)

    def caption(self, feat):
        if len(feat[0]) != self.engine.ctx_dim:
            print("Bad feature given! Expected size {} but got {}.".format(self.engine.ctx_dim, len(feat[0])))
            return ""

        feat_mask = self.engine.get_ctx_mask(feat)

        print("Sampling")
        # Get index-based samples and associated scores from beam search
        sample, score, _, _ = self.gen_sample(None, self.f_init, self.f_next, feat,
                                              feat_mask, self.model_options, maxlen=50)
        # Get best sample
        sidx = numpy.argmin(score)
        sample = sample[sidx]
        # indices -> words
        caption = self._seqs2words(sample)

        return caption

import gzip
import os, socket, shutil
import sys, re
import time
from collections import OrderedDict
import numpy
# import tables
import theano
import theano.tensor as T
import common
import numpy as np

# sys.path.append('skip-thoughts')
# import skipthoughts
from scipy import spatial
from nltk.corpus import stopwords

from multiprocessing import Process, Queue, Manager

hostname = socket.gethostname()

                
class Movie2Caption(object):
            
    def __init__(self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,dec,proc,
                 n_frames=None, outof=None, data_dir='', feats_dir=''
                 ):
        self.signature = signature
        self.model_type = model_type
        self.video_feature = video_feature
        self.maxlen = maxlen
        self.n_words = n_words
        self.K = n_frames
        self.OutOf = outof
        self.dec = dec

        self.mb_size_train = mb_size_train
        self.mb_size_test = mb_size_test
        self.non_pickable = []
        self.proc = proc
        self.host = socket.gethostname()
        self.data_dir=data_dir
        self.feats_dir = feats_dir

        # self.test_mode = 0 #don't chage this when in production
        self.load_data()
        


        if dec=='multi-stdist':
            # self.st_model = skipthoughts.load_model()  #refactoring ...
            # vectors = skipthoughts.encode(engine.st_model, captions)

            self.cap_distances = {}

        
    def _filter_feature(self, vidID):
        feat = self.FEAT[vidID]
        # print vidID
        # print feat
        feat = self.get_sub_frames(feat)
        return feat

    def _filter_c3d_resnet(self, vidID):
        feat = self.FEAT[vidID]
        feat2 = self.FEAT2[vidID]
        # print vidID
        # print feat
        feat = self.get_sub_frames(feat)
        feat2 = self.get_sub_frames(feat2)

        cfeat =np.concatenate((feat,feat2),axis=1)
        return cfeat

    def _load_feat_file(self, vidID):

        # feats_dir =os.path.join(data_dir,'features_chal')
        feat = []
        feats_dir = self.feats_dir

        feat_filename = vidID#files.split('/')[-1].split('.avi')[0]
        feat_file_path = os.path.join(feats_dir,feat_filename)

        if os.path.exists(feat_file_path):
            feat = np.load(feat_file_path)

            if len(feat) > 0:
                feat = self.get_sub_frames(feat)
            else:
                print 'feature file is empty '+feat_file_path
                print feat
        else:
            print 'error feature file doesnt exist'+feat_file_path


        return feat

    def _load_c3d_feat_file(self,vidID):
        feats_dir = 'vid-desc/vtt/features_c3d'
        feat_filename = vidID
        feat_file_path = os.path.join(feats_dir,feat_filename)

        if os.path.exists(feat_file_path):
            files = os.listdir(feat_file_path)
            files.sort()
            allftrs = np.zeros((len(files), 4101),dtype=np.float32)

            for j in range(0, len(files)):

                feat = np.fromfile(os.path.join(feat_file_path, files[j]),dtype=np.float32)
                allftrs[j,:] = feat
            allftrs = self.get_sub_frames(allftrs)

            return allftrs
        else:
            print 'error feature file doesnt exist'+feat_file_path
            sys.exit(0)


    def get_video_features(self, vidID):
        # hack to be fixed
        available_features = ['googlenet', 'resnet', 'c3d', 'resnet152', 'nasnetalarge', 'pnasnet5large', 'densenet152', 'polynet', 'senet154']
        if self.video_feature in available_features:
            if self.signature == 'youtube2text' or self.signature == 'ysvd' or self.signature == 'vtt16' or self.signature == 'vtt17' or self.signature == 'trecvid':
                y = self._filter_feature(vidID)
            elif self.signature == 'lsmdc' or self.signature == 'lsmdc16' or self.signature == 'mpii' or self.signature == 'mvad' or self.signature == 'tacos':
                y = self._load_feat_file(vidID) #this is for large datasets, needs to be fixed with something better. Mpii might need this..
            # elif self.signature == 'vtt':
            #     y = self._load_c3d_feat_file(vidID)
            else:
                raise NotImplementedError()
        elif self.video_feature == 'c3d_resnet':
            y = self._filter_c3d_resnet(vidID)
        else:
            raise NotImplementedError()
        return y

    def pad_frames(self, frames, limit, jpegs):
        # pad frames with 0, compatible with both conv and fully connected layers
        last_frame = frames[-1]
        if jpegs:
            frames_padded = frames + [last_frame]*(limit-len(frames))
        else:
            padding = numpy.asarray([last_frame * 0.]*(limit-len(frames)))
            frames_padded = numpy.concatenate([frames, padding], axis=0)
        return frames_padded
    
    def extract_frames_equally_spaced(self, frames, how_many):
        # chunk frames into 'how_many' segments and use the first frame
        # from each segment
        n_frames = len(frames)
        splits = numpy.array_split(range(n_frames), self.K)
        idx_taken = [s[0] for s in splits]
        sub_frames = frames[idx_taken]
        return sub_frames
    
    def add_end_of_video_frame(self, frames):
        if len(frames.shape) == 4:
            # feat from conv layer
            _,a,b,c = frames.shape
            eos = numpy.zeros((1,a,b,c),dtype='float32') - 1.
        elif len(frames.shape) == 2:
            # feat from full connected layer
            _,b = frames.shape
            eos = numpy.zeros((1,b),dtype='float32') - 1.
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        frames = numpy.concatenate([frames, eos], axis=0)
        return frames
    
    def get_sub_frames(self, frames, jpegs=False):
        # from all frames, take K of them, then add end of video frame
        # jpegs: to be compatible with visualizations
        if self.OutOf:
            raise NotImplementedError('OutOf has to be None')
            frames_ = frames[:self.OutOf]
            if len(frames_) < self.OutOf:
                frames_ = self.pad_frames(frames_, self.OutOf, jpegs)
        else:
            if len(frames) < self.K:
                #frames_ = self.add_end_of_video_frame(frames)

                frames_ = self.pad_frames(frames, self.K, jpegs)

            else:

                frames_ = self.extract_frames_equally_spaced(frames, self.K)
                #frames_ = self.add_end_of_video_frame(frames_)
        if jpegs:
            frames_ = numpy.asarray(frames_)
        return frames_

    def prepare_data_for_blue(self, whichset):
        # assume one-to-one mapping between ids and features
        feats = []
        feats_mask = []
        if whichset == 'valid':
            ids = self.valid_ids
        elif whichset == 'test':
            ids = self.test_ids
        elif whichset == 'train':
            ids = self.train_ids
        elif whichset == 'blind':
            ids = self.btest_ids

        for i, vidID in enumerate(ids):
            feat = self.get_video_features(vidID)
            feats.append(feat)
            feat_mask = self.get_ctx_mask(feat)
            feats_mask.append(feat_mask)
            # print i, vidID
        return feats, feats_mask
    
    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        
        return rval
    
    def load_feats(self,dataset_path):
        if self.video_feature=='c3d':
            if self.proc=='pca':
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d_'+self.proc+'.pkl'))
            elif self.proc=='pca512':
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d_'+self.proc+'.pkl'))
            elif self.proc=='pca_c3d':
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d_pca.pkl'))
            else:
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d.pkl'))

        elif self.video_feature=='c3d_resnet':
            if self.proc=='pca':
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d_'+self.proc+'.pkl'))
                self.FEAT2 = common.load_pkl(os.path.join(dataset_path , 'FEATS_resnet_'+self.proc+'.pkl'))
            elif self.proc=='pca512':
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d_'+self.proc+'.pkl'))
                self.FEAT2 = common.load_pkl(os.path.join(dataset_path ,'FEATS_resnet_'+self.proc+'.pkl'))
            elif self.proc=='pca_c3d':
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d_pca.pkl'))
                self.FEAT2 = common.load_pkl(os.path.join(dataset_path ,'FEATS_resnet_nostd.pkl'))
            else:
                self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_c3d.pkl'))
                self.FEAT2 = common.load_pkl(os.path.join(dataset_path ,'FEATS_resnet.pkl'))

        elif self.video_feature == 'googlenet':
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_googlenet.pkl'))
        elif self.video_feature == 'resnet':
            if self.proc=='pca':
                self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_resnet_'+self.proc+'.pkl'))
            else:
                self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_resnet.pkl'))
        elif self.video_feature == 'nasnetalarge':
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_nasnetalarge.pkl'))
        elif self.video_feature == 'resnet152':
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_resnet152.pkl'))
        elif self.video_feature == 'pnasnet5large':
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_pnasnet5large.pkl'))
        elif self.video_feature == 'polynet':
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_polynet.pkl'))
        elif self.video_feature == 'senet154':
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEATS_senet154.pkl'))
        else:
            self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEATS_'+self.proc+'.pkl'))
        return self
        
    def load_data(self):


        if self.signature == 'youtube2text' or self.signature == 'trecvid':
            print 'loading {} {} features'.format(self.signature, self.video_feature)
            if self.data_dir=='':
                dataset_path = common.get_rab_dataset_base_path()+'youtube2text/'+self.video_feature
            else:
                dataset_path = self.data_dir

            # dataset_path = common.get_rab_dataset_base_path()
            self.train = common.load_pkl(os.path.join(dataset_path ,'train.pkl'))
            self.valid = common.load_pkl(os.path.join(dataset_path ,'valid.pkl'))
            self.test = common.load_pkl(os.path.join(dataset_path ,'test.pkl'))
            self.CAP = common.load_pkl(os.path.join(dataset_path , 'CAP.pkl'))


            # self.FEAT = common.load_pkl(os.path.join(dataset_path , 'FEAT_key_vidID_value_features_'+self.proc+'.pkl'))
            self.load_feats(dataset_path)

            self.train_ids = list(set(self.train[i].split('_')[0] for i in range(len(self.train))))
            self.valid_ids = list(set(self.valid[i].split('_')[0] for i in range(len(self.valid))))
            self.test_ids = list(set(self.test[i].split('_')[0] for i in range(len(self.test))))


        elif self.signature == 'lsmdc' or self.signature == 'lsmdc16' or self.signature == 'mvad' or self.signature == 'mpii' or self.signature == 'tacos':
            print 'loading {} {} features'.format(self.signature, self.video_feature)
            dataset_path = self.data_dir
            self.train = common.load_pkl(os.path.join(dataset_path, 'train.pkl'))
            self.valid = common.load_pkl(os.path.join(dataset_path, 'valid.pkl'))
            self.test = common.load_pkl(os.path.join(dataset_path, 'test.pkl'))
            self.CAP = common.load_pkl(os.path.join(dataset_path, 'CAP.pkl'))

            self.train_ids = self.train
            self.valid_ids = self.valid
            self.test_ids = self.test

            if self.signature == 'lsmdc16':
                self.btest = common.load_pkl(os.path.join(dataset_path, 'blindtest.pkl'))
                self.btest_ids = self.btest


        elif self.signature == 'ysvd':
            print 'loading ysvd %s features'%self.video_feature
            dataset_path = common.get_rab_dataset_base_path()+'ysvd/'

            self.all = common.load_pkl(os.path.join(dataset_path, 'all_vids.pkl'))
            self.CAP = common.load_pkl(os.path.join(dataset_path, 'CAP.pkl'))
            self.FEAT = common.load_pkl(os.path.join(dataset_path, 'FEAT_key_vidID_value_features.pkl'))

            self.train = self.all[0:500]
            self.valid = self.all[501:750]
            self.test = self.all[751:1000]

            self.train_ids = self.train
            self.valid_ids = self.valid
            self.test_ids = self.test

        elif self.signature == 'vtt16' or self.signature == 'vtt17':
            print 'loading {} {} features'.format(self.signature, self.video_feature)

            if self.data_dir=='':
                dataset_path = common.get_rab_dataset_base_path()+'vtt/'+self.video_feature
            else:
                dataset_path = self.data_dir

            self.train = common.load_pkl(os.path.join(dataset_path, 'train.pkl'))
            self.valid = common.load_pkl(os.path.join(dataset_path, 'valid.pkl'))
            self.test = common.load_pkl(os.path.join(dataset_path, 'test.pkl'))
            self.CAP = common.load_pkl(os.path.join(dataset_path, 'CAP.pkl'))


            self.load_feats(dataset_path)

            # Get list of just the videoID, instead of videoID_CapID. Use set to ignore duplicates, then recast to list
            self.train_ids = list(set(self.train[i].split('_')[0] for i in range(len(self.train))))
            self.valid_ids = list(set(self.valid[i].split('_')[0] for i in range(len(self.valid))))
            self.test_ids = list(set(self.test[i].split('_')[0] for i in range(len(self.test))))

            self.test_ids = self.test_ids #only for testing

        else:
            raise NotImplementedError()
                
        self.worddict = common.load_pkl(os.path.join(dataset_path ,'worddict.pkl'))
        self.word_idict = dict()
        # wordict start with index 2
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'

        if self.video_feature == 'googlenet':
            self.ctx_dim = 1024
        elif self.video_feature == 'resnet' or self.video_feature == 'resnet152':
            if self.proc=='nostd':
                self.ctx_dim = 2048
            elif self.proc=='pca':
                self.ctx_dim=1024
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
            if self.proc=='nostd':
                self.ctx_dim = 4101
            elif self.proc=='pca':
                self.ctx_dim=1024
        elif self.video_feature == 'c3d_resnet':
            if self.proc=='nostd':
                self.ctx_dim = 6149
            elif self.proc=='pca':
                self.ctx_dim=2048
            elif self.proc=='pca512':
                self.ctx_dim=1024
            elif self.proc=='pca_c3d':
                self.ctx_dim=3072
        else:
            raise NotImplementedError()

        print "ctx_dim: "+str(self.ctx_dim)
        self.kf_train = common.generate_minibatch_idx(
            len(self.train), self.mb_size_train)
        self.kf_valid = common.generate_minibatch_idx(
            len(self.valid), self.mb_size_test)
        self.kf_test = common.generate_minibatch_idx(
            len(self.test), self.mb_size_test)

        if self.dec == 'multi-stdist':
            self.skip_vectors = common.load_pkl(os.path.join(dataset_path,'skip_vectors.pkl'))

        
def prepare_data(engine, IDs):
    # print "Preparing engine "+engine.dec
    seqs = []
    z_seqs = []
    feat_list = []

    def get_words(vidID, capID):
        rval = None
        if engine.signature == 'youtube2text' or engine.signature == 'vtt16' or engine.signature == 'vtt17' or engine.signature == 'trecvid':
            caps = engine.CAP[vidID]
            for cap in caps:
                if cap['cap_id'] == capID:
                    rval = cap['tokenized'].split(' ')
                    break
        elif engine.signature == 'lsmdc' or engine.signature == 'lsmdc16':
            cap = engine.CAP[vidID][0]
            rval = cap['tokenized'].split()
        elif engine.signature == 'mvad' or engine.signature == 'tacos':
            cap = engine.CAP[vidID][0]
            rval = cap['tokenized'].split()
        elif engine.signature == 'mpii':
            cap = engine.CAP[vidID][0]
            rval = cap['tokenized'].split()
        elif engine.signature == 'ysvd':
            cap = engine.CAP[vidID][capID]
            rval = cap['tokenized'].split()

        assert rval is not None
        return rval

    def get_z_seq():
        caps = engine.CAP[vidID]
        num_caps = len(caps)
        #print vidID+" "+str(num_caps)

        if engine.dec == 'multi-stdist': #'stdist'

            # common.dump_pkl(caps,'/media/onina/SSD/projects/skip-thoughts/caps')

            if not engine.cap_distances.has_key(vidID):

                captions = [ caps[0]['caption'] for x in range(num_caps)] #initialized all with the firs caption
                for i in range(0,num_caps):
                    cap = caps[i]

                    if engine.signature != 'vtt16' or engine.signature != 'vtt17':
                        id = int(cap['cap_id'])
                        
                        caption = cap['caption']
                        # print str(id)+" "+caption
                        # print len(captions)
                        # print vidID
                        udata=caption.decode("utf-8")

                        # if id>=num_caps:
                        #     continue
                        captions[id] = udata.encode("ascii","ignore")

                        if captions[id].isspace():
                            captions[id] = captions[0]
                    else:
                        captions[i] = cap['tokenized']
                    # print captions[id]

                # common.dump_pkl(captions,'captions')
                # vectors = skipthoughts.encode(engine.st_model,captions)  #refactoring this line
                vectors = engine.skip_vectors[vidID]
                caps_dist = spatial.distance.cdist(vectors, vectors, 'cosine')
                engine.cap_distances[vidID] = caps_dist

            caps_dist = engine.cap_distances[vidID]
            query_id = int(capID)
            js =range(0, query_id) + range(query_id+1,num_caps)
            
            
            if len(js)>0 and engine.signature != 'mvad':
                # print js,query_id
                most_distant = np.argmax(caps_dist[query_id,js])
            else:
                most_distant = 0

            z_words = get_words(vidID, str(most_distant))
            z_seq = [engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in z_words]


        elif engine.dec == 'generative':
            z_words = get_words(vidID, str(1))
            z_words = [word for word in z_words if word not in stopwords.words('english')]
            z_seq = [engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in z_words]

        elif engine.dec == 'generative.2':
            
            z_words = get_words(vidID, str(1))
            z_words = [word for word in z_words if word not in stopwords.words('english')]
            # print z_words

            def get_hypernyms(z_words):

                from nltk.corpus import wordnet 
                new_z_words = []
                for word in z_words:
                    hypernyms = wordnet.synsets(word)
                    if len(hypernyms) > 1 :
                        h = hypernyms[0].hypernyms()
                        if len(h) >0:
                            nwords = h[0].lemma_names()
                            nword = str(nwords[0])
                            if '_' not in nword and '-' not in nword and engine.worddict.has_key(nword):
                                new_z_words.append(nword)
                                # print word+' replaced with '+ nword
                            else:
                                new_z_words.append(word)
                        else:
                            new_z_words.append(word)
                    else:
                        new_z_words.append(word)

                return new_z_words

            import random
            if random.randint(0,1):  #only change to hypernyms every .5 percent the time
                z_words = get_hypernyms(z_words) 
            # print z_words

            z_seq = [engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in z_words]


            # print new_z_words

        return z_seq

    def clean_sequences(seqs,z_seqs,feat_list):

        if  engine.dec=="standard":

            lengths = [len(s) for s in seqs]
            if engine.maxlen != None:
                new_seqs = []
                new_feat_list = []
                new_lengths = []
                new_caps = []
                for l, s, y, c in zip(lengths, seqs, feat_list, IDs):
                    # sequences that have length >= maxlen will be thrown away
                    if l < engine.maxlen:
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
            if engine.maxlen != None:
                new_seqs = []
                new_zseqs = []
                new_feat_list = []
                new_lengths = []
                new_caps = []
                new_zlengths = []
                for l,z_l, s, y, c in zip(lengths,z_lengths, seqs, feat_list, IDs):
                    # sequences that have length >= maxlen will be thrown away
                    if l < engine.maxlen and z_l < engine.maxlen :
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

    for i, ID in enumerate(IDs):
        #print 'processed %d/%d caps'%(i,len(IDs))
        # print ID
        if engine.signature == 'youtube2text' or engine.signature == 'vtt16' or engine.signature == 'vtt17' or engine.signature == 'trecvid':
            # load GNet feature
            vidID, capID = ID.split('_')
        elif engine.signature == 'tacos':
            vidID = ID
            capID = 0
        elif engine.signature == 'lsmdc' or engine.signature == 'lsmdc16':
            # t = ID.split('_')
            # vidID = '_'.join(t[:-1])
            # capID = t[-1]
            vidID = ID
            capID = 1
        elif engine.signature == 'mvad':
            # t = ID.split('_')
            # vidID = '_'.join(t[:-1])
            # capID = t[-1]
            vidID = ID
            capID = 1
        elif engine.signature == 'ysvd':
            # t = ID.split('_')
            # vidID = '_'.join(t[:-1])
            # capID = t[-1]
            vidID = ID
            capID = 0
        elif engine.signature == 'mpii':
            vidID = ID
            capID = 1
        else:
            raise NotImplementedError()

        feat = engine.get_video_features(vidID)

        # if len(feat[0])!= engine.ctx_dim:
        #     print 'dim error on '+vidID
        #     sys.exit(0)

        feat_list.append(feat)
        words = get_words(vidID, capID)
        # print words
        seqs.append([engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in words])

        # print engine.dec
        if engine.dec != "standard":
            z_seq = get_z_seq()
            z_seqs.append(z_seq)


    seqs,z_seqs,feat_list,lengths = clean_sequences(seqs,z_seqs,feat_list)

    if len(lengths) < 1:
        return None, None, None, None

    y = numpy.asarray(feat_list)
    # print len(y[1,1])
    y_mask = engine.get_ctx_mask(y)

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    if engine.dec=="standard":
        return x, x_mask, y, y_mask
    else:
        z = numpy.zeros((maxlen, n_samples)).astype('int64')  #This is the other label
        z_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
        for idx, s in enumerate(z_seqs):
            z[:lengths[idx],idx] = s
            z_mask[:lengths[idx]+1,idx] = 1.

        return x, x_mask, y, y_mask,z,z_mask


def test_data_engine():
    video_feature = 'googlenet' 
    out_of = None
    maxlen = 100
    mb_size_train = 64
    mb_size_test = 128
    maxlen = 50
    n_words = 30000 # 25770 
    signature = 'youtube2text' #'youtube2text'
    engine = Movie2Caption('attention', signature, video_feature,
                           mb_size_train, mb_size_test, maxlen,
                           n_words,'standard','nostd',
                           n_frames=26,
                           outof=out_of)
    i = 0
    t = time.time()
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train[index] for index in idx]
        x, mask, ctx, ctx_mask = prepare_data(engine, ids)
        print 'seen %d minibatches, used time %.2f '%(i,time.time()-t0)
        if i == 10:
            break
            
    print 'used time %.2f'%(time.time()-t)


if __name__ == '__main__':
    test_data_engine()



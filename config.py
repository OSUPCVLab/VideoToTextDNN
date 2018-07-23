from jobman import DD
import common

exp_path = common.get_rab_exp_path()

config = DD({
    'model': 'attention',
    'random_seed': 1234,
    # ERASE everything under save_model_path
    'erase_history': True,
    'attention': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'arctic-capgen-vid/test_non/',
        'from_dir': '',
        'dataset': 'youtube2text',#'youtube2text',#'lsmdc',mvad. 'ysvd'
        'video_feature': 'googlenet',
        'dim_word':468, # 474
        'ctx_dim':-1,# auto set 
        'dim':3518, # lstm dim # 536
        'n_layers_out':1, # for predicting next word        
        'n_layers_init':0, 
        'encoder_dim': 300,
        'prev2out':True, 
        'ctx2out':True, 
        'patience':20,
        'max_epochs':500, 
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.01,
        'selector':True,
        'n_words':20000, 
        'maxlen':30, # max length of the descprition
        'optimizer':'adadelta',
        'clip_c': 10.,
        'batch_size': 64, # for trees use 25
        # 'batch_size': 2, # for trees use 25
        'valid_batch_size':200,
        # 'valid_batch_size':2,
        # in the unit of minibatches
        'dispFreq':200,
        'validFreq':2000,
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        'use_dropout':True,
        'K':28, # 26 when compare
        'OutOf':None, # used to be 240, for motionfeature use 26
        'verbose': True,
        'debug': False,
        'dec':'standard',
        'encoder':None,
        'mode':'train',
        'proc':'nostd',
        'data_dir':'',
        'feats_dir':''
        }),
    'iLSTM': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'attention_mod/',
        'dec':'standard',
        'valid_batch_size':200,
        'dataset': 'youtube2text',
        'encoder': None,
        'max_epochs':500,
        'from_dir': '',
        }),
    'attention_mod': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'attention_mod/',
        'dec':'multi-stdist'
        }),
    'mtle': DD({
        'save_model_dir': exp_path + 'arctic-capgen-vid/test_non/',
        'reload_': False,
        'from_dir': '',
        'dec':'multi-stdist',
        'dim_word':468, # 474
        'encoder':None,
        'encoder_dim': 300,
        'batch_size': 64, #64 for trees use 25
        'valid_batch_size':200,
        'dataset': 'vtt',
        'dim':3518, # lstm dim # 536
        'video_feature': 'googlenet',
        'validFreq': 2000,
        'max_epochs': 500,
        'mode':'train',
        'proc':'nostd',
        'K':28, # 26 when compare
        'lrate':0.0001,
        'data_dir':'',
        'dispFreq':10,
        'feats_dir':'',
        'cost_type':'v1'
        }),
    'fcoupled': DD({
        'save_model_dir': exp_path + 'arctic-capgen-vid/test_non/',
        'reload_': False,
        'dec':'multi-random',
        'encoder':None,
        'encoder_dim': 300,
        'batch_size': 64, # for trees use 25
        'dataset': 'youtube2text',
        'dim':3518, # lstm dim # 536
        'from_dir': '',
        'valid_batch_size':200,
        'max_epochs':500,
        'video_feature': 'googlenet',
        }),
    'const': DD({
        'save_model_dir': exp_path + 'arctic-capgen-vid/test_non/',
        'reload_': False,
        'dec':'multi-random',
        'encoder':None,
        'encoder_dim': 300,
        'batch_size': 64, # for trees use 25
        'dataset': 'youtube2text',
        'dim':3518, # lstm dim # 536
        'from_dir': '',
        }),
    'const2': DD({
        'save_model_dir': exp_path + 'arctic-capgen-vid/test_non/',
        'reload_': False,
        'dec':'multi-random',
        'encoder':None,
        'encoder_dim': 300,
        'batch_size': 64, # for trees use 25
        'dataset': 'youtube2text'
        }),
    'LSTM': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'attention_mod/',
        'dec':'standard',
        'valid_batch_size':200,
        'dataset': 'youtube2text',
        'encoder': 'lstm_uni',
        'max_epochs':500,
        'from_dir': '',
        }),
    'lstmdd': DD({
        'save_model_dir': exp_path + 'arctic-capgen-vid/test_non/',
        'reload_': False,
        'from_dir': '',
        'dec':'multi-stdi',
        'dim_word':468, # 474
        'encoder':None,
        'encoder_dim': 300,
        'batch_size': 64, #64 for trees use 25
        'valid_batch_size':200,
        'dataset': 'vtt',
        'dim':3518, # lstm dim # 536
        'video_feature': 'googlenet',
        'validFreq': 2000,
        'max_epochs': 500,
        'mode':'train',
        'proc':'nostd',
        'K':28, # 26 when compare
        'lrate':0.0001,
        'data_dir':'',
        'dispFreq':10,
        'feats_dir':'',
        'cost_type':'v1'

    }),
    'gru': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'gru_model2/',
        'from_dir': '',
        'dataset': 'youtube2text',#'youtube2text',#'lsmdc',mvad. 'ysvd'
        'video_feature': 'googlenet',
        'dim_word':468, # 474
        'ctx_dim':-1,# auto set
        'dim':3518, # lstm dim # 536
        'n_layers_out':1, # for predicting next word
        'n_layers_init':0,
        'encoder_dim': 300,
        'prev2out':True,
        'ctx2out':True,
        'patience':20,
        'max_epochs':500,
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.01,
        'selector':True,
        'n_words':20000,
        'maxlen':30, # max length of the descprition
        'optimizer':'adadelta',
        'clip_c': 10.,
        'batch_size': 64, # for trees use 25
        # 'batch_size': 2, # for trees use 25
        'valid_batch_size':200,
        # 'valid_batch_size':2,
        # in the unit of minibatches
        'dispFreq':10,
        'validFreq':2000,
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        'use_dropout':True,
        'K':28, # 26 when compare
        'OutOf':None, # used to be 240, for motionfeature use 26
        'verbose': True,
        'debug': False,
        'dec':'standard',
        'encoder':None,
        'mode':'train',
        'proc':'nostd'
        }),
    'fc': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'attention_mod/',
        'dec':'standard',
        'dataset': 'youtube2text',
        'encoder': None,
        'from_dir': '',
        }),
    'ic': DD({
        'reload_': False,
        'save_model_dir': exp_path + 'attention_mod/',
        'dec':'standard',
        'dataset': 'youtube2text',
        'encoder': None,
        'from_dir': '',
        }),
    'const_w': DD({
        'save_model_dir': exp_path + 'const_w/',
        'reload_': False,
        'dec':'multi-stdist',
        'encoder':None,
        'encoder_dim': 300,
        'batch_size': 64, # for trees use 25
        'dataset': 'youtube2text',
        'video_feature': 'googlenet',
        }),


})

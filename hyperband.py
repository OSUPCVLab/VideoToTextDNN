import sys
sys.path.insert(1,'jobman')
sys.path.insert(1,'coco-caption')

import os
import random
import copy
import subprocess
import numpy as np

from math import *
from numpy import argsort
from multiprocessing import Pool


def args_as_typed(args):
    result = ""
    for key in args:
        result += key
        result += "="
        result += str(args[key])
        result += " "

    return result


def get_random_hyperparameter_configuration():
    hp_dict = {'dim_word': int(random.uniform(100, 1000)),
               'dim': int(random.uniform(100, 5000)),
               'encoder_dim': int(random.uniform(100, 900)),
               'cost_type': np.random.choice(['v1', 'v3', 'v4', 'v5', 'v6'])}

    return hp_dict


def run_then_return_val_loss(args, num_iters, hyperparameters, gpu_id):
    # -7: BLEU1
    # -6: BLEU2
    # -5: BLEU3
    # -4: BLEU4
    # -3: Meteor
    # -2: Rouge
    # -1: Cider
    colnum = -4

    # Parse through arguments and replace as necessary
    model = args['model'].replace('\'', '')

    # Do save_model_dir and logging for this run
    save_model_key = model + '.save_model_dir'
    save_model_dir = args[save_model_key].replace('\'', '')

    run_name = model + '_'
    run_name += 'HYPERBAND_{}-iters-{}'\
        .format('_'.join(['{}-{}'.format(k, hyperparameters[k]) for k in hyperparameters]), num_iters)

    logging_dir = os.path.join(save_model_dir, 'logs', run_name)
    if not os.path.isdir(logging_dir):
        os.makedirs(logging_dir)

    save_model_dir = os.path.join(save_model_dir, run_name)
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)

    args[save_model_key] = '\'' + save_model_dir + '\''

    # Do Epochs
    num_epochs_key = model + '.max_epochs'
    args[num_epochs_key] = num_iters

    # Set hyper-parameters
    for k in hyperparameters:
        args[model + '.' + k] = hyperparameters[k]

    theano_flag = "THEANO_FLAGS=\'device=gpu{}\'".format(gpu_id)
    # "/dev/null 2>&1"
    command = "{} {} {} > {} 2>&1".format(theano_flag, "python train_model.py", args_as_typed(args), os.path.join(logging_dir, 'record.txt'))
    print " ----- \n{}".format(command)

    os.system(command)

    print " %%%%% Job finished! \n{}".format(args_as_typed(args))
    train_loss_path = os.path.join(save_model_dir, 'train_valid_test.txt')
    if os.path.isfile(train_loss_path):
        train_loss_file = open(train_loss_path)
        lines = [i.replace('\n', '').split(' ') for i in train_loss_file]
        return float(lines[-1][colnum])
    else:
        print "Validation results were not found for this run! validFreq value must be lowered, or the training crashed."
        return 0.000


def HYPERBAND(args):
    """
    Adapted from:
    https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    Performs HYPERBAND across available GPUs using Theano flags.
    This version uses BLEU4 as the score.
    :param args:
    :return:
    """
    max_iter = 81  # maximum iterations/epochs per configuration
    eta = 3  # defines downsampling rate (default=3)
    logeta = lambda x: log(x) / log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max + 1) * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    # Modify this for your needs
    models_per_gpu = 2
    avail_gpus = [0, 1]
    #avail_gpus = range(num_gpu)

    num_gpu = len(avail_gpus)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max + 1)):
        n = int(ceil(B / max_iter / (s + 1) * eta ** s))  # initial number of configurations
        r = max_iter * eta ** (-s)  # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        T = [get_random_hyperparameter_configuration() for _ in range(n)]

        for i in range(s + 1):
            val_losses = []

            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n * eta ** (-i)
            r_i = int(floor(int(r * eta ** (i))))
            r_i += 3 # Add 3 iterations since only see results after 4-8 epochs
            if r_i > 60:
                continue

            print ' ---- \nAt s: {}, i: {}, r_i: {}, T is: {}'.format(s, i, r_i, T)
            #val_losses = [run_then_return_val_loss(args=copy.deepcopy(args), num_iters=r_i, hyperparameters=t) for t in T]
            # First figure out what runs must be done
            runs = [(copy.deepcopy(args), r_i, t) for t in T]

            # Now tag runs with a GPU id and add to pending jobs, until no more runs
            while len(runs) > 0:
                gpuPool = Pool(num_gpu * models_per_gpu)
                gpu_subprocess_params_list = []

                for gpu_id in avail_gpus:
                    # First build the params by tagging on correct gpu_id
                    model_params_per_gpu = [runs.pop() + (gpu_id,)
                                            for i in range(models_per_gpu) if len(runs) != 0]
                    # Use params to build list of async functions on new threads
                    model_params_per_gpu = [gpuPool.apply_async(run_then_return_val_loss, i)
                                            for i in model_params_per_gpu]

                    gpu_subprocess_params_list.extend(model_params_per_gpu)

                # Execute all pending jobs, getting results as jobs finish
                val_losses = map(lambda x: x.get(), gpu_subprocess_params_list)
                gpuPool.close()
                gpuPool.join()

            print 'val_losses was: {}'.format(val_losses)
            T = [T[i] for i in argsort(val_losses)[0:int(n_i / eta)]]


        #### End Finite Horizon Successive Halving with (n,r)

if __name__ == '__main__':
    args = {}
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            args[k] = v
    except:
        print 'args must be like a=X b.c=X'
        exit(1)

    HYPERBAND(args)

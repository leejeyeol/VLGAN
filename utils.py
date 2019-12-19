import time
import random
import torch
import numpy as np
from visdom import Visdom
vis = Visdom()

def tvt_divider(cur_file_paths, train_ratio = 4, val_ratio = 1, test_ratio = 1, seed = 1111):
    # original dataset => train, validation, test set
    random.seed(seed)
    random.shuffle(cur_file_paths)
    num_dataset = len(cur_file_paths)
    total_ratio = train_ratio + val_ratio + test_ratio

    train_idx = int(num_dataset / total_ratio * train_ratio)
    val_idx = train_idx + int(num_dataset / total_ratio * val_ratio)

    return cur_file_paths[: train_idx], cur_file_paths[train_idx: val_idx], cur_file_paths[val_idx:]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)

class Time_calculator():
    def __init__(self):
        self.total_time_start()
        self.mean_time = []

    def total_time_start(self):
        self.total_start_time = time.time()

    def total_time_end(self):
        self.time_print(time.time()-self.total_start_time, "total spended time")

    def simple_time_start(self, string=None):
        self.simple_start_time = time.time()
        if string is None :
            self.simple_time_name = 'spended time'
        else:
            self.simple_time_name = string

    def simple_time_end(self):
        used_time = time.time() - self.simple_start_time
        self.mean_time.append(used_time)
        #self.time_print(used_time, self.simple_time_name)
        return used_time

    def mean_reset(self):
        self.mean_time = []
    def mean_calc(self):
        mean = np.asarray(self.mean_time).mean()
        self.time_print(mean, 'mean time per epoch')
        self.mean_reset()

    def time_print(self, tm, string):
        h = tm/360
        h_ = tm%360
        m = h_/60
        m_ = h_%60
        s = m_
        print("%s is %d:%02d:%.4f" % (string, h, m, s))

def make_dirs(paths,allow_duplication =False):
    try :
        len(paths)
    except:
        print("error : paths don't have length")
        return
    for i in range(len(paths)):
        make_dirs(paths[i],allow_duplication)

def win_dict():
    win_dict = dict(
        exist=False)
    return win_dict

def draw_lines_to_windict(win_dict, value_list, legend_list, epoch, iteration, total_iter):
    # epoch * total_iter + iteration
    # no epoch 0,i,0

    num_of_values = len(value_list)
    for i in range(len(value_list)):
        if 'torch' in str(type(value_list[i])):
            if value_list[i].is_cuda:
                value_list[i] = value_list[i].cpu()
        value_list[i] =np.asarray(value_list[i])

    if type(win_dict) == dict:
        # first. line plots
        if legend_list is None :
            win_dict = vis.line(X=np.column_stack((0 for _ in range(num_of_values))),
                                Y=np.column_stack((value_list[i] for i in range(num_of_values))),
                                opts=dict(
                               title='loss-iteration',
                               xlabel='iteration',
                               ylabel='loss',
                               xtype='linear',
                               ytype='linear',
                               makers=False
                           ))
        else:
            win_dict = vis.line(X=np.column_stack((0 for _ in range(num_of_values))),
                                Y=np.column_stack((value_list[i] for i in range(num_of_values))),
                                opts=dict(
                                    legend=legend_list,
                                    title='loss-iteration',
                                    xlabel='iteration',
                                    ylabel='loss',
                                    xtype='linear',
                                    ytype='linear',
                                    makers=False
                                ))

    else:
        win_dict = vis.line(
            X=np.column_stack((epoch*total_iter + iteration for _ in range(num_of_values))),
            Y=np.column_stack((value_list[i] for i in range(num_of_values))),
            win=win_dict,
            update='append'
        )
    return win_dict

def torch_model_gradient(model_parameters):
    grad_list = []
    for f in model_parameters:
        num_of_weight = 1
        if type(f.grad) is not type(None):
            if torch.is_tensor(f.grad) is True:
                for i in range(len(f.grad.shape)):
                    num_of_weight *= f.grad.shape[i]
            else:
                for i in range(len(f.grad.data.shape)):
                    num_of_weight *= f.grad.shape[i]
            grad_list.append(float(torch.sum(torch.abs(f.grad))/num_of_weight))
    return np.asarray(grad_list).mean()
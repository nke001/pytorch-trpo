import argparse
from itertools import count
import pickle 

#import free_mjc
import gym
import scipy.optimize
import numpy as np
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from rl_zforcing_new import ZForcing
import cv2
import random
import scipy.misc
import os

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(550, 500))
display_.start()

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux-weight-start', type=float, default=0.,
                    help='start weight for auxiliary loss')
parser.add_argument('--aux-weight-end', type=float, default=0.,
                    help='end weight for auxiliary loss')
parser.add_argument('--bwd-weight', type=float, default=0.,
                    help='weight for bwd teacher forcing loss')
parser.add_argument('--bwd-l2-weight', type=float, default=1e-3,
                    help='weight for bwd l2 decoding loss')
parser.add_argument('--kld-weight-start', type=float, default=0.,
                    help='start weight for kl divergence between prior and posterior z loss')
parser.add_argument('--kld-step', type=float, default=1e-6,
                    help='step size to anneal kld_weight per iteration')
parser.add_argument('--aux-step', type=float, default=1e-6,
                    help='step size to anneal aux_weight per iteration')

parser.add_argument('--eval-interval', type=int, default=50, metavar='N',
                    help='evaluation interaval (default: 50)')

parser.add_argument('--val-batch-size', type=int, default=50, metavar='N',
                    help='random seed (default: 1)')

args = parser.parse_args()
lr = args.lr
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)


filename = args.env_name + '-model-based/zforce_reacher_decode_state_model_based_lr'+ str(args.lr) + '_bwd_weight_' + str(args.bwd_weight) + '_aux_w_' + str(args.aux_weight_start) + '_kld_w_' + str(args.kld_weight_start) + '_' + str(random.randint(1,500))
os.makedirs(filename, exist_ok=True)
train_folder = os.path.join(filename, 'train')
test_folder = os.path.join(filename, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
zforce_filename = os.path.join(filename, 'student.pkl')
log_file = os.path.join(filename, 'log.txt')

train_on_image = True

def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

def load_samples(filename):
    output = open(filename, "rb")
    all_data = pickle.load(output)
    return all_data


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def write_samples(all_samples_obs, all_samples_actions, filename):
    # write to pickle file
    all_data = list(zip(all_samples_obs, all_samples_actions))
    output = open(filename, "wb")
    pickle.dump(all_data, output)
    output.close()
    return True

def print_norm(rnn):
    param_norm = []
    
    for param_tuple in zf.named_parameters():
        name = param_tuple[0]
        param = param_tuple[1]
        if 'bwd' in name:
            norm = param.grad.norm(2).data[0]/np.sqrt(np.prod(param.size()))
            #if norm == 0.0:
            #    import ipdb; ipdb.set_trace()
            param_norm.append(norm)
    return param_norm
    #for param in rnn.parameters():
    #    import ipdb; ipdb.set_trace()
    #    norm = param.grad.norm(2).data[0]/ numpy.sqrt(numpy.prod(param.size()))
    #    #print param.size()
    #    param_norm.append(norm)
    #return param_norm 


def evaluate_(model):
    # evaluate how well model does 
    num_episodes = 0
    reward_batch = 0
    model.cuda()
    hidden = zf.init_hidden(1)
    all_action_diff = 0
    zf.eval()
    action = np.asarray([0.,0.])
    # Each iteration first collect #batch_size episodes
    while num_episodes < args.val_batch_size:
        #print(num_episodes)
        state = env.reset()
        reward_sum = 0
        for t in range(10000):
            image = env.render(mode="rgb_array") 
            if num_episodes % 5 == 0:
                image_file =  os.path.join(filename, 'test/episode_'+ str(num_episodes) +  '_t_' + str(t)+'.jpg')
                scipy.misc.imsave(image_file, image)
            action = torch.from_numpy(action).float().cuda()
            image = image_resize(image)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            mask = torch.ones([1,1])
            action_mu, action_var, hidden = zf.generate_onestep(image, mask, hidden, action = action.unsqueeze(0).unsqueeze(0)) 
            
            action_mu = action_mu.squeeze(0).squeeze(0)
            action_logvar = action_var.squeeze(0).squeeze(0)
            std = action_logvar.mul(0.5).exp_()
            
            eps = std.data.new(std.size()).normal_()
            
            action = eps.mul(std).add_(action_mu)
            
            action = action.cpu().data.numpy()

            expert_action = select_action(state).data.numpy()
            
            action_diff_norm =  (np.linalg.norm(expert_action - action))
            
            all_action_diff += action_diff_norm

            next_state, reward, done, _ = env.step(action)
            
            state = running_state(next_state)
            reward_sum += reward
            
            if done:
                break

        num_episodes += 1
        reward_batch += reward_sum

    print ('test reward is ', reward_batch/ num_episodes)
    print ('average action diff norm is ', all_action_diff / num_episodes / 50)
    log_line = 'test_reward is , ' + str(reward_batch/ num_episodes)
    with open(log_file, 'a') as f:
        f.write(log_line)
    return (reward_batch/num_episodes) 

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
running_state_test = ZFilter((num_inputs,), clip=5)

load_param(policy_net, "Reacher_policy.pkl")
load_param(value_net, "Reacher_value.pkl")

policy_net#.cuda()
value_net#.cuda()

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])

zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
              mlp_dim=256, out_dim=num_actions * 2, z_force=True, cond_ln=True)

#opt = torch.optim.Adam(zf.parameters(), lr=lr, eps=1e-5)
fwd_param = []
bwd_param = []

hist_return_mean = 0.0

for param_tuple in zf.named_parameters():
    name = param_tuple[0]
    param = param_tuple[1]
    if 'bwd' in name:
        bwd_param.append(param)
    else:
        fwd_param.append(param)

zf_fwd_param = (n for n in fwd_param)
zf_bwd_param = (n for n in bwd_param)
#opt = torch.optim.Adam(zf.parameters(), lr=lr, eps=1e-5)
fwd_opt = torch.optim.Adam(zf_fwd_param, lr = lr, eps=1e-5)
bwd_opt = torch.optim.Adam(zf_bwd_param, lr = lr, eps=1e-5)

kld_weight = args.kld_weight_start
aux_weight = args.aux_weight_start
bwd_weight = args.bwd_weight
bwd_l2_weight = args.bwd_l2_weight

#import ipdb.set_trace()
#zf = load_param(zf, 'zforce_reacher_64.pkl')
zf.float()
zf.cuda()
hist_test_reward = -30.0

#evaluate_(zf)
#import ipdb; ipdb.set_trace()

def image_resize(image):
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return image

#zf.eval()
#evaluate_(zf)

zf.train() 

data_file = 'data/Reacher-v2_num_samples_10000.pkl'
all_data = load_samples(data_file)
all_training_images, all_training_actions = [list(t) for t in zip(*all_data)]

num_samples = len(all_training_actions)

num_episodes = 50

for episode in range(num_episodes):
    
    for iteration in range(int(num_samples/ args.batch_size)):
        # After having #batch_size trajectories, make the python array into numpy array
        index = np.random.randint(num_samples - args.batch_size, size=1)[0]
        training_images = all_training_images[index : index + args.batch_size]
        training_actions = all_training_actions[index : index + args.batch_size]
        images_max_len = max_length(training_images)
        actions_max_len = max_length(training_actions)
        images_mask = [[1] * (len(array) - 1) + [0] * (images_max_len - len(array))
                   for array in training_images]
    
        # Here's something a little twisted, we want the trajectories in one batch to be the same
        # length. So we want to pad zero to the ends of short trajectories. However, the forward
        # and backward trajectories are shifted by one. So we need to create and pad the fwd/bwd
        # trajectories individually and pass them to the zforcing model.
        fwd_images = [pad(array[:-1], images_max_len - 1) for array in training_images]
        bwd_images = [pad(array[1:], images_max_len - 1) for array in training_images]
        training_actions = [pad(array, actions_max_len) for array in training_actions]
        fwd_images = np.array(list(zip(*fwd_images)), dtype=np.float32)
        bwd_images = np.array(list(zip(*bwd_images)), dtype=np.float32)
        images_mask = np.array(list(zip(*images_mask)), dtype=np.float32)
        training_actions = np.array(list(zip(*training_actions)), dtype=np.float32)
    
        x_fwd = torch.from_numpy(fwd_images).cuda()
        x_bwd = torch.from_numpy(bwd_images).cuda()
        y = torch.from_numpy(training_actions).cuda()
        x_mask = torch.from_numpy(images_mask).cuda()
    
        zf.float().cuda()
        hidden = zf.init_hidden(args.batch_size)


        fwd_opt.zero_grad()
        bwd_opt.zero_grad()
    
        fwd_nll, bwd_nll, aux_nll, kld, l2_loss = zf(x_fwd, x_bwd, y, x_mask, hidden)
        bwd_nll = (aux_weight > 0.) * (bwd_weight * bwd_nll)
        aux_nll = aux_weight * aux_nll
        all_loss = fwd_nll + bwd_nll + aux_nll + kld_weight * kld + args.bwd_l2_weight * l2_loss
        fwd_loss = (fwd_nll + aux_nll + kld_weight * kld)
        bwd_loss = bwd_nll + args.bwd_l2_weight * l2_loss
        # anneal kld cost
        kld_weight += args.kld_step
        kld_weight = min(kld_weight, 1.)
        if args.aux_weight_start < args.aux_weight_end:
            aux_weight += args.aux_step
            aux_weight = min(aux_weight, args.aux_weight_end)
        else:
            aux_weight -= args.aux_step
            aux_weight = max(aux_weight, args.aux_weight_end)
        log_line ='Episode: %d, Iteration: %d, All loss is %.3f , foward loss is %.3f, backward loss is %.3f, aux loss is %.3f, kld is %.3f, l2 loss is %.3f' % (
            episode,
            iteration,
            all_loss.item(),
            fwd_nll.item(),
            bwd_nll.item(),
            aux_nll.item(),
            kld.item(),
            args.bwd_l2_weight * l2_loss.item()
        ) + '\n'
        print(log_line)
        with open(log_file, 'a') as f:
            f.write(log_line)
        if np.isnan(all_loss.item()) or np.isinf(all_loss.item()):
            continue

        # backward propagation
        fwd_loss.backward()
        bwd_loss.backward()
    
        torch.nn.utils.clip_grad_norm_(zf.parameters(), 100.)
        print('norm is ', np.asarray(print_norm(zf)).mean())
        fwd_opt.step()
        bwd_opt.step()
        if (iteration + 1 ) % args.eval_interval == 0:
            test_reward = evaluate_(zf)
            if (-test_reward) < (-hist_test_reward):
                hist_test_reward = test_reward
                save_param(zf, zforce_filename) 
        zf.train()

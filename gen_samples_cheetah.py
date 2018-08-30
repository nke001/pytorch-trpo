
import readline
from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(150, 150))
display_.start()

import glfw

import argparse
from itertools import count

#import free_mjc
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from rl_zforcing_aux import ZForcing
import cv2
import random
import scipy.misc
import os
import pickle

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True


torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--index', type=int, default=0, 
                    help = 'index for data generated')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
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
parser.add_argument('--num-samples', type=int, default=1000,
                    help='number of samples ')

args = parser.parse_args()
lr = args.lr
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
filename = str(args.env_name) + 'num_samples_' + str(args.num_samples) 
env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)


train_on_image = True

def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def write_samples(all_samples_obs, all_samples_actions, filename, idx=0):
    # write to pickle file
    #all_data = list(zip(all_samples_obs, all_samples_actions))
    filename = filename + '_' + str(idx) + '.pkl'
    #outfile = open(filename, "wb")
    
    with open(filename, "wb+") as f:
        for obs, action in zip(all_samples_obs, all_samples_actions):
            pickle.dump((obs, action), f)
 
    print ('done writing to file ', filename) 
    #pickle.dump(all_samples_obs, outfile)
    #pickle.dump(all_samples_actions, outfile)
    #outfile.close()
    return True

def print_norm(rnn):
    param_norm = []
    
    for param_tuple in zf.named_parameters():
        name = param_tuple[0]
        param = param_tuple[1]
        if 'bwd' in name:
            norm = param.grad.norm(2).data[0]/np.sqrt(np.prod(param.size()))
            param_norm.append(norm)
    return param_norm



running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
running_state_test = ZFilter((num_inputs,), clip=5)

load_param(policy_net, "Cheetah_policy.pkl")
load_param(value_net, "Cheetah_value.pkl")

policy_net#.cuda()
value_net#.cuda()

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])


def image_resize(image):
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return image


def expert_sample():
    import ipdb; ipdb.set_trace()


idx = 0

for iteration in range(1):
    training_images, training_actions = [], []
    
    num_episodes, reward_batch = 0, 0
    
    # Each iteration first collect #batch_size episodes
    while num_episodes < args.num_samples:
        #print(num_episodes)
        episode_images, episode_actions, reward_sum = [], [], 0
        state = env.reset() 
        state = running_state_test(state)
        
        for t in range(10000):
            action = select_action(state).data[0].numpy()
            #action = action.data[0].numpy()
             
            if train_on_image:
                image = env.render(mode="rgb_array")
                image_filename = 'cheetah_images_expert/episode_'+ str(num_episodes) + '_t_' + str(t)+'.jpg'
                if num_episodes % 1000 == 0:
                    scipy.misc.imsave(image_filename, image)

            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = running_state_test(next_state)
            
            episode_images.append(np.transpose(img, (2,0,1))
            episode_actions.append(action)
            state = next_state
            
            if done:
                break
        
        num_episodes += 1
        reward_batch += reward_sum
        if (num_episodes > 0 ) and (num_episodes % 100 == 0):
            print('done ', num_episodes) 
        
        img = env.render(mode="rgb_array")
        episode_images.append(np.transpose(img,(2,0,1)))
        
        training_images.append(episode_images)
        training_actions.append(episode_actions)
        if num_episodes % 1000 == 0 and num_episodes > 10:
            write_samples(training_images, training_actions, filename, idx)
            training_images, training_actions = [], []
            idx += 1
    
    #write_samples(training_images, training_actions, filename, idx)
    print (reward_batch/ num_episodes)
    import ipdb; ipdb.set_trace()

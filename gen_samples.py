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

from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(150, 150))
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
filename = str(args.env_name) + 'valid_num_samples_' + str(args.num_samples) + '.pkl'
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
            param_norm.append(norm)
    return param_norm



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


def image_resize(image):
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return image


def expert_sample():
    import ipdb; ipdb.set_trace()


for iteration in range(1):
    training_images = []
    training_actions = []
    
    num_episodes = 0
    reward_batch = 0
    # Each iteration first collect #batch_size episodes
    while num_episodes < args.num_samples:
        #print(num_episodes)
        episode_images = []
        episode_actions = []
        state = env.reset() 
        state = running_state_test(state)
        reward_sum = 0
        
        for t in range(10000):
            action = select_action(state)
            action = action.data[0].numpy()
            
            if train_on_image:
                image = env.render(mode="rgb_array")
                crop_img = image[48:112, 30:110]
                image_filename = 'expert_images/episode_'+ str(num_episodes) + '_t_' + str(t)+'.jpg'
                #scipy.misc.imsave(image_filename, crop_img)
                #crop_img = cv2.resize(crop_img, dsize=(64,64), interpolation=cv2.INTER_CUBIC) 
                crop_img = np.transpose(crop_img, (2, 0, 1))
                #scipy.misc.imsave(image_filename, crop_img)

            next_state, reward, done, _ = env.step(action)
            
            reward_sum += reward
            next_state = running_state_test(next_state)
            
            episode_images.append(crop_img)
            episode_actions.append(action)
            
            state = next_state
            if done:
                break
        
        num_episodes += 1
        reward_batch += reward_sum
        if (num_episodes > 0 ) and (num_episodes % 100 == 0):
            print('done ', num_episodes) 
        image = env.render(mode="rgb_array")
        #image = image_resize(image)
        crop_img = image[48:112, 30:110]
        episode_images.append(np.transpose(crop_img,(2,0,1)))
        
        training_images.append(episode_images)
        training_actions.append(episode_actions)
    print (reward_batch/ num_episodes)
    write_samples(training_images, training_actions, filename)

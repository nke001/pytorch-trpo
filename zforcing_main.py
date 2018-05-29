import argparse
from itertools import count

import free_mjc
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from rl_zforcing import ZForcing
import cv2

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(1400, 900))
display_.start()

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--test', type=bool, default=False, help="no update params")
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state).cuda())
    action = torch.normal(action_mean, action_std)
    save_param(policy_net, 'Reacher.pkl')
    return action

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
load_param(policy_net, "./Reacher_policy.pkl")
load_param(value_net, "./Reacher_value.pkl")
policy_net.cuda()
value_net.cuda()

zf = ZForcing(emb_dim=256, rnn_dim=128, z_dim=128,
              mlp_dim=128, out_dim=2, z_force=True, cond_ln=True)
def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])

for i_episode in count(1):
    training_images = []
    training_actions = []
    num_episodes = 0
    while num_episodes < args.batch_size:
        print(num_episodes)
        episode_images = []
        episode_actions = []
        state = env.reset()
        state = running_state(state)
        for t in range(10000):
            action = select_action(state)
            action = action.data[0].cpu().numpy()
            image = env.render(mode="rgb_array")
            image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            image = np.transpose(image, (2, 0, 1))
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            episode_images.append(image)
            episode_actions.append(action)
            if done:
                break
        image = env.render(mode="rgb_array")
        image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image = np.transpose(image, (2, 0, 1))
        episode_images.append(image)
        training_images.append(episode_images)
        training_actions.append(episode_actions)
        num_episodes += 1
    images_max_len = max_length(training_images)
    actions_max_len = max_length(training_actions)
    images_mask = [[1] * (len(array) - 1) + [0] * (images_max_len - len(array))
                   for array in training_images]
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
    fwd_nll, bwd_nll, aux_nll, kld = zf(x_fwd, x_bwd, y, x_mask, hidden)

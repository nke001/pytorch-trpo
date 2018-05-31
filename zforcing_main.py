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
from rl_zforcing import ZForcing
import cv2

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

from pyvirtualdisplay import Display
display_ = Display(visible=0, size=(1400, 900))
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
parser.add_argument('--kld-weight-start', type=float, default=0.,
                    help='start weight for kl divergence between prior and posterior z loss')
parser.add_argument('--kld-step', type=float, default=5e-5,
                    help='step size to anneal kld_weight per iteration')
parser.add_argument('--aux-step', type=float, default=5e-5,
                    help='step size to anneal aux_weight per iteration')

args = parser.parse_args()

lr = args.lr
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

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])


zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
              mlp_dim=256, out_dim=num_actions, z_force=True, cond_ln=True)
opt = torch.optim.Adam(zf.parameters(), lr=lr, eps=1e-5)

kld_weight = args.kld_weight_start
aux_weight = args.aux_weight_start
bwd_weight = args.bwd_weight

for iteration in count(1):
    training_images = []
    training_actions = []
    num_episodes = 0
    # Each iteration first collect #batch_size episodes
    while num_episodes < args.batch_size:
        print(num_episodes)
        episode_images = []
        episode_actions = []
        state = env.reset()
        state = running_state(state)
        reward_sum = 0
        reward_batch = 0
        for t in range(10000):
            action = select_action(state)
            action = action.data[0].cpu().numpy()
            image = env.render(mode="rgb_array")
            image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            image = np.transpose(image, (2, 0, 1))
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = running_state(next_state)
            episode_images.append(image)
            episode_actions.append(action)
            if done:
                break
            reward_batch += reward_sum

        image = env.render(mode="rgb_array")
        image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image = np.transpose(image, (2, 0, 1))
        episode_images.append(image)
        training_images.append(episode_images)
        training_actions.append(episode_actions)
        print (reward_batch/ num_episodes)
        num_episodes += 1
    # After having #batch_size trajectories, make the python array into numpy array
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


    opt.zero_grad()
    # forward prop pass
    # fwd_nll: the forward rnn teacher forcing
    # bwd_nll: the backward rnn teacher forcing
    # aux_nll: the auxiliary task, z -> bwd prediction
    # kld: the kl between the posterior and prior of z
    fwd_nll, bwd_nll, aux_nll, kld = zf(x_fwd, x_bwd, y, x_mask, hidden)
    bwd_nll = (aux_weight > 0.) * (bwd_weight * bwd_nll)
    aux_nll = aux_weight * aux_nll
    all_loss = fwd_nll + bwd_nll + aux_nll + kld_weight * kld

    # anneal kld cost
    kld_weight += args.kld_step
    kld_weight = min(kld_weight, 1.)
    if args.aux_weight_start < args.aux_weight_end:
        aux_weight += args.aux_step
        aux_weight = min(aux_weight, args.aux_weight_end)
    else:
        aux_weight -= args.aux_step
        aux_weight = max(aux_weight, args.aux_weight_end)
    log_line =' All loss is %.3f , foward loss is %.3f, backward loss is %.3f, aux loss is %.3f' % (
            all_loss.item(),
            fwd_nll.item(),
            bwd_nll.item(),
            aux_nll.item()
        ) + '\n'
    print(log_line)
  
    if np.isnan(all_loss.item()) or np.isinf(all_loss.item()):
        continue

    # backward propagation
    all_loss.backward()
    torch.nn.utils.clip_grad_norm(zf.parameters(), 100.)


    opt.step()

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
parser.add_argument('--l2-weight', type=float, default=1.,
                    help='l2 weight for foward model')
parser.add_argument('--eval-interval', type=int, default=50, metavar='N',
                    help='evaluation interaval (default: 50)')

parser.add_argument('--val-batch-size', type=int, default=50, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5, 
                    help='k step ahead predication')
parser.add_argument('--zf-file', type=str, default='/private/home/nke001/pytorch-trpo/Reacher-v2hybrid-model/zforce_reacher_model_base_10k_0.0_lr0.0001_fwd_l2w_0.0_aux_w_0.0_kld_w_0.0_86/student.pkl', help='reloading zforcing file')
args = parser.parse_args()
lr = args.lr
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)


def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

def load_samples(filename):
    output = open(filename, "rb")
    all_data = pickle.load(output)
    return all_data


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

l2_loss = nn.MSELoss()

def evaluate_rollouts(true_traj, all_rollouts, k):
    losses = []
    rollouts = all_rollouts[0]
    for i in range( k + 1):
        loss = l2_loss(true_traj[-i].detach(), rollouts[-i].detach() )
        losses.append(loss)
    
    losses = torch.stack(losses).mean()
    return (losses, all_rollouts[1:])



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
            action = torch.from_numpy(action).float().cuda()
            image = image_resize(image)
            import ipdb; ipdb.set_trace()
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            mask = torch.ones([1,1])
            action_mu, action_var, hidden = zf.generate_onestep(image, mask, hidden, action = action.unsqueeze(0).unsqueeze(0)) 
            
            action_mu = action_mu.squeeze(0).squeeze(0)
            action_logvar = action_var.squeeze(0).squeeze(0)
            std = action_logvar.mul(0.5).exp_()
            
            eps = std.data.new(std.size()).normal_()
            
            action = eps.mul(std).add_(action_mu)
            
            action = action.cpu().data.numpy()

            next_state, reward, done, _ = env.step(action)
            
            state = running_state(next_state)
            reward_sum += reward
            
            if done:
                break

        num_episodes += 1
        reward_batch += reward_sum

    print ('test reward is ', reward_batch/ num_episodes)
    log_line = 'test_reward is , ' + str(reward_batch/ num_episodes)
    return (reward_batch/num_episodes) 

def sample_action(action_mu, action_var):
    action_mu = action_mu.squeeze(0).squeeze(0)                                                                                                                                                              
    action_logvar = action_var.squeeze(0).squeeze(0)                                                                                                                                                         
    std = action_logvar.mul(0.5).exp_()                                                                                                                                                                      
                                                                                                                                                                                                                     
    eps = std.data.new(std.size()).normal_()                                                                                                                                                                 

    action = eps.mul(std).add_(action_mu)  
    return action

def k_step_evaluate(model, k):
    # evaluate how well model does 
    num_episodes = 0
    reward_batch = 0
    model.cuda()
    hidden = zf.init_hidden(1)
    zf.eval()
    action = np.asarray([0.,0.])
    all_image_diff = [] 
    l2_loss = nn.MSELoss()
    # Each iteration first collect #batch_size episodes
    
    while num_episodes < args.val_batch_size:
        state = env.reset()
        reward_sum = 0
        k_step_loss, all_rollouts, true_traj, rollouts = [], [], [], []
        
        mask = torch.ones([1,1])
         
        for t in range(50):
            image = env.render(mode="rgb_array")
            action = torch.from_numpy(action).float().cuda()
            image = image_resize(image)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda() 
            image_tildt = image
            true_hidden = hidden
            true_traj.append(image)
            rollouts.append(image_tildt)
            # do k-step rollouts, every k-step, reset the image to the real image
            for i in range(k):
                # do k-step rollouts
                #image = image_resize(image)
                action_mu, action_var, hidden, image_tildt = zf.generate_onestep(image_tildt, mask, hidden, return_decode=True, action = action.unsqueeze(0).unsqueeze(0))
                image_file =  os.path.join('rollouts_new', 'episode_'+ str(num_episodes) +  '_t_' + str(t)+'.jpg')
                write_image = np.transpose(image_tildt.cpu().detach().squeeze(0).squeeze(0).numpy(), (1,2,0))
                scipy.misc.imsave(image_file, write_image)
          
                rollouts.append(image_tildt)
                action = sample_action(action_mu, action_var) 
            rollouts = [] 
            action_mu, action_var, hidden, _ = zf.generate_onestep(image, mask, true_hidden, return_decode=True, action = action.unsqueeze(0).unsqueeze(0))
            all_rollouts.append(rollouts) 
            
            if t >= (k-1):
                k_loss, all_rollouts = evaluate_rollouts(true_traj, all_rollouts, k)
                # start evaluating the k-step rollouts
                k_step_loss.append(k_loss)
            action = sample_action(action_mu, action_var)
            action = action.cpu().data.numpy()

            next_state, reward, done, _ = env.step(action)                                                                                                                                                           
            state = running_state(next_state)                                                                                                                                                                        
            reward_sum += reward                                                                                                                                                                                     
                                                                                                                                                                                                                     
            if done:                                                                                                                                                                                                 
                break                                                                                                                                                                                                
                                                                                                                                                                                                                     
        num_episodes += 1                                                                                                                                                                                            
        reward_batch += reward_sum                                                                                                                                                                                   
        if num_episodes % 5 ==0:
            print ('done with ', num_episodes)                                                                                                                                                                                                         
    print ('test reward is ', reward_batch/ num_episodes)                                                                                                                                                            
    #print ('average action diff norm is ', all_action_diff / num_episodes / 50)        
    print ('average k step predication loss is ', torch.stack(k_step_loss).mean().item())
    return (reward_batch/num_episodes)  

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
running_state_test = ZFilter((num_inputs,), clip=5)

def pad(array, length):
    return array + [np.zeros_like(array[-1])] * (length - len(array))
def max_length(arrays):
    return max([len(array) for array in arrays])

zf = ZForcing(emb_dim=512, rnn_dim=512, z_dim=256,
              mlp_dim=256, out_dim=num_actions * 2, z_force=True, cond_ln=True)

zf = load_param(zf, args.zf_file)

#import ipdb.set_trace()
#zf = load_param(zf, 'zforce_reacher_64.pkl')
zf.float()
zf.cuda()

#evaluate_(zf)

def image_resize(image):
    #image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image =  np.array(image[48:112, 30:110], dtype=float)
    image = np.transpose(image, (2, 0, 1))
    return image

#zf.eval()
#evaluate_(zf)

num_episodes = 1

for episode in range(num_episodes):
    
    for iteration in range(1):
    
        zf.float().cuda()
        hidden = zf.init_hidden(args.batch_size)

        k_step_loss = k_step_evaluate(zf, args.k)
        #test_reward = evaluate_(zf)

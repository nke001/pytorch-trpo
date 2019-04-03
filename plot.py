
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re










if __name__== '__main__':
    #filename = 'BabyAI-UnlockPickup-v0_model/zforce_room_8_lr0.0001_aux_w_0.0_kld_w_0.0_417.log'
    #filename = 'BabyAI-UnlockPickup-v0_model/zforce_2opt_room_10_lr0.0001_bwd_w_0.0_l2_w_1.0_aux_w_1e-05_kld_w_0.0_694.log'
    #filename = 'Reacher-v2_model/zforce_reacher_decode_state_lr0.0001_bwd_weight_0.0_aux_w_0.0_kld_w_0.0_379/log.txt'
    filename = '/private/home/nke001/pytorch-trpo/Reacher-v2_version1.4_12k/zforce_reacher_model_base_10k__lr0.0001_fwd_l2w_1.0_aux_w_0.0005_kld_w_0.2_340/log.txt'
    lines = [line.rstrip('\n') for line in open(filename)]
    
    zf_rewards = []
    lstm_rewards = []
    lstm_dec_rewards = []
    mean = 0.0
    for line in lines:
        if 'test_reward' in line:
            words = line.split()
            reward  = words[3][:-8]
            
            zf_rewards.append(float(reward))    
    
    filename ='/private/home/nke001/pytorch-trpo/Reacher-v2_version1.4_12k/zforce_reacher_model_base_10k__lr0.0001_fwd_l2w_1.0_aux_w_0.0_kld_w_0.0_193/log.txt'
    #filename = '/private/home/nke001/pytorch-trpo/Reacher-v2_version1.4_12k/zforce_reacher_model_base_10k__lr0.0001_fwd_l2w_1.0_aux_w_0.0_kld_w_0.0_193/log.txt'
    lines = [line.rstrip('\n') for line in open(filename)]

    
    for line in lines:
        if 'test_reward' in line:
            words = line.split()
            reward  = words[3][:-8]
            lstm_dec_rewards.append(float(reward))
    
    filename = '/private/home/nke001/pytorch-trpo/Reacher-v2_version1.4_12k/zforce_reacher_model_base_10k__lr0.0001_fwd_l2w_0.0_aux_w_0.0_kld_w_0.0_493/log.txt' 
    lines = [line.rstrip('\n') for line in open(filename)]

    lstm_rewards = []

    for line in lines:
        if 'test_reward' in line:
            words = line.split()
            reward  = words[3][:-8]
            lstm_rewards.append(float(reward))

    plt.plot(zf_rewards, label='zforcing')
    plt.plot(lstm_rewards, label='lstm')
    plt.plot(lstm_dec_rewards, label='lstm_dec')
    plt.legend(loc='upper right')
    plt.savefig('reacher_rewards.pdf')
    import ipdb; ipdb.set_trace()


# good set of hyperparameters

source activate py3.6

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/private/home/nke001/.mujoco/mjpro150/bin

print(torch.version.cuda)


vim /private/home/nke001/anaconda3/lib/python3.6/site-packages/gym/envs/mujoco/mujoco_env.py

vim /anaconda3/envs/py3.6_cuda9.0/lib/python3.6/site-packages/mujoco_py/generated

python zforcing_main_aux_new.py --batch-size 32 --aux-weight-start 0.0001  --aux-weight-end 0.0001 --kld-step 0.00005 --kld-weight-start 0.2 --bwd-weight 0.0   --lr 1e-4 --bwd-l2-
weight 1. --batch-size 32 --l2-weight 0.2

python zforcing_main.py --batch-size 10  --aux-weight-start 0.1 --kld-weight-start 0.1 --bwd-weight 1.0  --lr 1e-3 --aux-weight-end 0.001

from multiprocessing import Process, Queue, Pool, Manager
import numpy

def _generate_trajectory(env_name, train_on_image, queue_images, queue_actions, queue_reward):
  import torch
  from torch.autograd import Variable
  import numpy as np
  import free_mjc
  import gym
  from running_state import ZFilter
  from models import Policy, Value
  import cv2

  from pyvirtualdisplay import Display
  display_ = Display(visible=0, size=(550, 500))
  display_.start()

  def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))
    return model

  def image_resize(image):
    image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return image

  env = gym.make(env_name)
  num_inputs = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]
  policy_net = Policy(num_inputs, num_actions)
  value_net = Value(num_inputs)
  load_param(policy_net, "Reacher_policy.pkl")
  load_param(value_net, "Reacher_value.pkl")
  policy_net.float()
  value_net.float()
  running_state_test = ZFilter((num_inputs,), clip=5)

  def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0).float()
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

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
      image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
      image = np.transpose(image, (2, 0, 1))
    next_state, reward, done, _ = env.step(action)
    reward_sum += reward
    next_state = running_state_test(next_state)
    episode_images.append(image)
    episode_actions.append(action)
    state = next_state
    if done:
      break
  image = env.render(mode="rgb_array")
  image = image_resize(image)
  episode_images.append(image)
  queue_images.put(episode_images)
  queue_actions.put(episode_actions)
  queue_reward.put(reward_sum)

def generate_expert_trajectory(env_name, batch_size, num_processes):
  """ return training_images, training_actions just as
  what was done in the original zforcing_main file. But in parallel.
  """
  p = Pool(num_processes)
  m = Manager()
  queue_images = m.Queue()
  queue_actions = m.Queue()
  queue_reward = m.Queue()
  ret = p.starmap(
      _generate_trajectory,
      [(env_name, True, queue_images, queue_actions, queue_reward) for i in range(batch_size)])
  training_images = []
  training_actions = []
  rewards = []
  while queue_images.qsize() != 0:
    training_images.append(queue_images.get())
  while queue_actions.qsize() != 0:
    training_actions.append(queue_actions.get())
  while queue_reward.qsize() != 0:
    rewards.append(queue_reward.get())
  return training_images, training_actions, rewards

if __name__ == "__main__":
  training_images, training_actions, rewards = generate_expert_trajectory("Reacher-v2", 10, 10)

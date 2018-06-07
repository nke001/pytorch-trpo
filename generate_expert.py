from multiprocessing import Process, Queue, Pool, Manager
import numpy

num_processes = 10
def pool_initializer(env_name, output_queue):
  from running_state import ZFilter
  import torch
  from torch.autograd import Variable
  import numpy as np
  import free_mjc
  import gym
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
  running_state_train = ZFilter((num_inputs,), clip=5)

  def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0).float()
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

  def _generate_trajectory(train_on_image):
    episode_images = []
    episode_actions = []
    state = env.reset()
    state = running_state_train(state)
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
      next_state = running_state_train(next_state)
      episode_images.append(image)
      episode_actions.append(action)
      state = next_state
      if done:
        break
    image = env.render(mode="rgb_array")
    image = image_resize(image)
    episode_images.append(image)
    output_queue.put((episode_images, episode_actions, reward_sum))

  while True:
    _generate_trajectory(True)


processes = None
m = Manager()
output_queue = m.Queue(1000)

def initialize_pool(env_name):
  global processes
  processes = [Process(target=pool_initializer, args=(env_name, output_queue)) for i in range(num_processes)]
  for p in processes:
    p.daemon = True
    p.start()


def generate_expert_trajectory(batch_size):
  """ return training_images, training_actions just as
  what was done in the original zforcing_main file. But in parallel.
  """
  training_images = []
  training_actions = []
  rewards = []
  for i in range(batch_size):
    images, actions, reward = output_queue.get()
    training_images.append(images)
    training_actions.append(actions)
    rewards.append(reward)
  return training_images, training_actions, rewards

if __name__ == "__main__":
  initialize_pool("Reacher-v2")
  while True:
    training_images, training_actions, rewards = generate_expert_trajectory(10)
    print(sum(rewards) / 10)

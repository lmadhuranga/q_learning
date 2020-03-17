import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000


SHOW_EVERY = 10000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# print(discrete_os_win_size)

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decaying_value  = epsilon /  (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
  discrete_state = (state - env.observation_space.low) / discrete_os_win_size
  return tuple(discrete_state.astype(np.int))


ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg':[], 'min':[], 'max':[]}

for episode in range(EPISODES): 
  episode_reward = 0
  if episode % SHOW_EVERY == 0:
    render = True
  
  else:
    render = False

  discrete_state = get_discrete_state(env.reset())

  # print(discrete_state)

  # print(q_table[discrete_state])


  done = False
  while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, done, _ = env.step(action)
    episode_reward += reward
    new_discrete_state = get_discrete_state(new_state)
    
    if render:
      env.render()

    if not done:
      max_future_q = np.max(q_table[new_discrete_state]) 
      current_q =  q_table[discrete_state + (action, )]

      new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)
      q_table[discrete_state + (action,)] = new_q

    elif new_state[0] >= env.goal_position:
      print(f"We made it on episode {episode}") 
      q_table[discrete_state+ (action, )] = 0

    discrete_state = new_discrete_state

  if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
    epsilon -= epsilon_decaying_value

  ep_rewards.append(episode_reward)
  if not episode % SHOW_EVERY:
    average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
    aggr_ep_rewards['ep'].append(episode)
    aggr_ep_rewards['avg'].append(average_reward)
    aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
    aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:])) 

    print(f"Episode: {episode} avg: {average_reward} min: {aggr_ep_rewards['min']} max: {aggr_ep_rewards['max']}")
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
# plt.legend(loca=4)
plt.show()

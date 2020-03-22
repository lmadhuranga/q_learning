import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F

from Libs import CartPoleEnvManager, EpsilonGreedyStrategy, ReplayMemory, Agent, DQN, QValues

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

agent = Agent(strategy, em.num_actions_available(), device)
# 1. Initiate Replay memory capacity
memory = ReplayMemory(memory_size)

# 2. Initiate policy network with random weight
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
#  3. Clone the policy network as target network
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


# utility functions
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    if is_ipython: display.clear_output(wait=True)


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


episode_durations = []
for episode in range(num_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        # 1. Select the action via exploration or exploitation
        action = agent.select_action(state, policy_net)
        # Execute selected action in an emulator
        reward = em.take_action(action)
        # Observe reward and next state
        next_state = em.get_state()
        # Store experience in replay memory
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            # Sample random batch from replay memory
            experiences = memory.sample(batch_size)
            # Preprocessed states from batch.
            states, actions, rewards, next_states = extract_tensors(experiences)
            # Calculate loss between output Q-values and target Q-valeues.
            current_q_values = QValues.get_current(policy_net, states, actions)
            # Requires a pass to the target network for the next state
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            # Gradient descent updates weight in the policy network to minimize loss.
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
    # After X time steps, weights in the target network are updated to the weight in the policy network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()

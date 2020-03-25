import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F
# following tutorila url
# https://deeplizard.com/learn/video/ewRw996uevM

from Libs import CartPoleEnvManager, EpsilonGreedyStrategy, ReplayMemory, Agent, DQN, QValues

batch_size = 256
# gamma, which is the discount factor used in the Bellman equation, is being set to 0.999
gamma = 0.999
# eps_start is the starting value of epsilon. Remember, epsilon is the name we’ve given to the exploration rate.
# eps_end is the ending value of epsilon, and eps_decay is the decay rate we’ll use to decay epsilon over time.
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
# we set target_update to 10, and this is how frequently, in terms of episodes,
# we’ll update the target network weights with the policy network weights.
# So, with target_update set to 10, we’re choosing to update the target network every 10 episodes.
target_update = 10
# we set the memory_size, which is the capacity of the replay memory,
# to 100,000. We then set the learning rate lr that is used during training of the policy network to 0.001,
# and the number of episodes we want to play num_episodes to 1000.
memory_size = 100000
lr = 0.001
num_episodes = 1000

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

#  let’s set up our device for PyTorch. This tells PyTorch to use a GPU if it’s available, otherwise use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now, we set up our environment manager em using the CartPoleEnvManager class, and we pass in the required device.
# We then set our strategy to be an instance of the EpsilonGreedyStrategy class, and we pass in the required start,
# end, and decay values for epsilon.
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

# We then define an agent using our Agent class and pass in the required strategy, number of actions available, and device.
# We then initialize memory to be an instance of ReplayMemory and pass in the capacity using memory_size.
agent = Agent(strategy, em.num_actions_available(), device)
# 1. Initiate Replay memory capacity
memory = ReplayMemory(memory_size)

# Now, we define both our policy network and target network by creating two instances of our DQN class and
# passing in the height and width of the screen to set up the appropriate input shape of the networks.
# We put these networks on our defined device using PyTorch’s to() function.
# 2. Initiate policy network with random weight
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
#  3. Clone the policy network as target network
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

# We then set the weights and biases in the target_net to be the same as those in the policy_net using PyTorch’s state_dict()
# and load_state_dict() functions. We also put the target_net into eval mode, which tells PyTorch
# cthat this network is not in training mode. In other words, this network will only be used for inference.
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Lastly, we set optimizer equal to the Adam optimizer, which accepts our policy_net.
# parameters() as those for which we’ll be optimizing, and our defined learning rate lr.
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

# Training loop
# We’re now all set up to start training.
# We’re going to be storing our episode_durations during training in order to plot them using the plot()
# function we developed last time, so we create an empty list to store them in.

episode_durations = []
for episode in range(num_episodes):
    # For each episode, we first reset the environment, then get the initial state.cc
    em.reset()
    state = em.get_state()
    # For each episode, we first reset the environment, then get the initial state.
    #
    # Now, we'll step into the nested for loop that will iterate over each time step within each episode.
    for timestep in count():
        # For each time step, our agent selects an action based on the current state. Recall,
        # we also need to pass in the required policy_net since the agent will use this network to select
        # it’s action if it exploits the environment rather than explores it.
        action = agent.select_action(state, policy_net)  # 1. Select the action via exploration or exploitation
        reward = em.take_action(action)  # Execute selected action in an emulator

        # The agent then takes the chosen action and receives the associated reward, and we get the next_state.
        next_state = em.get_state()  # Observe reward and next state

        # We can create an Experience using the state, action, next_state,and reward and push this onto replay memory.
        # After which, we transition to the next state by setting our current state to next_state.
        memory.push(Experience(state, action, next_state, reward))  # Store experience in replay memory
        state = next_state

        # Now that our agent has had an experience and stored it in replay memory, we’ll check to see
        # if we can get a sample from replay memory to train our policy_net.
        # Remember, we covered in a previous episode that we can get a sample equal to the batch_size
        # from replay memory as long as the current size of memory is at least the batch_size.
        if memory.can_provide_sample(batch_size):
            # If we can get a sample from memory,
            # then we get a sample equal to batch_size and assign this sample to the variable experiences.
            # We’re then going to do some data manipulation to extract all the states, actions, rewards, and next_states
            # into their own tensors from the experiences list.
            experiences = memory.sample(batch_size)  # Sample random batch from replay memory

            # We do this using the extract_tensors() function.
            # We haven’t covered the inner workings of this function yet,
            # but stick around until the end, and we’ll circle back around to cover it in detail. For now,
            # let’s continue with our training loop so that we can stay in flow.
            states, actions, rewards, next_states = extract_tensors(experiences)  # Preprocessed states from batch.
            # Continuing with the code above, we now we get the q-values for the corresponding state-action pairs that
            # we’ve extracted from the experiences in our batch. We do this using QValues.get_current(),
            # to which we pass our policy_net, states, and actions.

            # just know that get_current() will return the q-values for any given state-action pairs, as predicted
            # from the policy network. The q-values will be returned as a PyTorch tensor.
            current_q_values = QValues.get_current(policy_net, states, actions)  # Calculate loss between output Q-values and target Q-valeues.

            # We also need to get the q-values for the next states in the batch as well.
            # We’re able to do this using QValues.get_next(), and passing in the target_net and next_states that
            # we extracted from the experiences.
            # This function will return the maximum q-values for the next states using using the best corresponding
            # next actions. It does this using the target network because, remember from our episode on fixed Q-targets,
            # the q-values for next states are calculated using the target network.
            next_q_values = QValues.get_next(target_net, next_states)  # Requires a pass to the target network for the next state

            # We multiply each of the next_q_values by our discount rate gamma and add this result to the corresponding
            # reward in the rewards tensor to create a new tensor of target_q_values.
            target_q_values = (next_q_values * gamma) + rewards
            # We now can calculate the loss between the current_q_values and the target_q_values using
            # mean squared error mse as loss function, and then we zero out the gradients using optimizer.zero_grad().

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))  # Gradient descent updates weight in the policy network to minimize loss.
            # This function sets the gradients of all the weights and baises in the policy_net to zero.
            # Since PyTorch accumulates the gradients when it does backprop, we need to call zero_grad()
            # before backprop occurs. Otherwise, if we didn’t zero out the gradients each time, then we’d be
            # accumulating gradients across all backprop runs.
            optimizer.zero_grad()

            # We then call loss.backward(), which computes the gradient of the loss with respect to all the weights and
            # biases in the policy_net.
            loss.backward()
            # We now call step() on our optimizer, which updates the weights and biases with the gradients that were
            # computed when we called backward() on our loss.
            optimizer.step()

        # We then check to see if the last action our agent took ended the episode by getting the value of done
        # from our environment manager em. If the episode ended, then we append the current timestep to the episode_
        # durations list to store how long this particular episode lasted.
        if em.done:
            episode_durations.append(timestep)
            # We then plot the duration and the 100-period moving average to the screen and break out of the inner
            # loop so that we can start a new episode.
            plot(episode_durations, 100)
            break
    # Before starting a new episode though, we have one final check to see if we should do an update to our target_net.
    if episode % target_update == 0:
        # Recall, our target_update variable is set to 10, so we check if our current episode is a multiple of 10,
        # and if it is, then we update the target_net weights with the policy_net weights.
        #
        # At this point, we can start a new episode. This whole process will end once we’ve reached the number of
        # episodes set in num_episodes. At that point, we'll close the enironment manager.
        target_net.load_state_dict(policy_net.state_dict())

em.close()

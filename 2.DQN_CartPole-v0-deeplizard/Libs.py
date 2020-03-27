
import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        #       initiate layers of networks
        self.fc1 = nn.Linear(in_features=img_height * img_width * 3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=24)
        self.out = nn.Linear(in_features=24, out_features=2)

    #     first forward and return the network
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t


# Epsilon Greedy Strategy - exploration versus exploitation. This has to do with the way our agent selects actions.
# Recall, our agent’s actions will either fall
# in the category of exploration, where the agent is just exploring the environment by taking a random action from a
# given state, or the category of exploitation, where the agent exploits what it’s learned about the environment to
# take the best action from a given state.
#
# To get a balance of exploration and exploitation, we use what we previously introduced as an epsilon greedy
# strategy. With this strategy, we define an exploration rate called epsilon that we initially set to 1 . This
# exploration rate is the probability that our agent will explore the environment rather than exploit it. With
# epsilon equal to 1 , it is 100 percent certain that the agent will start out by exploring the environment.
#
# As the agent learns more about the environment, though, epsilon will decay by some decay rate that we set so that
# the likelihood of exploration becomes less and less probable as the agent learns more and more about the
# environment. We’re now going to write an EpsilonGreedyStrategy class that puts this idea into code.
class EpsilonGreedyStrategy():
    # Our EpsilonGreedyStrategy accepts start, end, and decay, which correspond to the starting, ending,
    # and decay values of epsilon. These attributes all get initialized based on the values that are passed in during
    # object creation.
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

# So, later when we create an Agent object, we’ll need to already have an instance of EpsilonGreedyStrategy class
# created so that we can use that strategy to create our agent. num_actions corresponds to how many possible actions
# can the agent take from a given state. In our cart and pole example, this number will always be two since the agent
# can always choose to only move left or right.
class Agent():
    # We initialize the agent’s strategy and num_actions accordingly, and we also initialize the current_step
    # attribute to 0. This corresponds to the agent’s current step in the environment. The Agent class has a single
    # function called select_action(), which requires a state and a policy_net.
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    # Remember a policy network is the name we give to our deep Q-network that we train to learn the optimal policy.
    def select_action(self, state, policy_net):
        # Within this function, we first initialize rate to be equal to the exploration rate returned from the
        # epsilon greedy strategy that was passed in when we created our agent, and we increment the agent’s
        # current_step by 1.
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # We then check to see if the exploration rate is greater than a randomly generated number between 0 and 1.
        # If it is, then we explore the environment by randomly selecting an action, either 0 or 1, corresponding to
        # left or right moves.
        #
        # If the exploration rate is not greater than the random number, then we exploit the environment by selecting
        # the action that corresponds to the highest Q-value output from our policy network for the given state.
        #
        # We’re specifying with torch.no_grad() before we pass data to our policy_net to turn off gradient tracking
        # since we’re currently using the model for inference and not training.
        #
        # During training PyTorch keeps track of all the forward pass calculations that happen within the network. It
        # needs to do this so that it can know how to apply backpropagation later. Since we’re only using the model
        # for inference at the moment, we’re telling PyTorch not to keep track of any forward pass calculations.
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)


# Replay Memory Now that we have our Experience class, let’s define our ReplayMemory class, which is where these
# experiences will be stored.
#
# Recall that replay memory will have some set capacity. This capacity is the only parameter that needs to be
# specified when creating a ReplayMemory object.
class ReplayMemory():
    # We initialize ReplayMemory’s capacity to whatever was passed in, and we also define a memory attribute equal to
    # an empty list. memory will be the structure that actually holds the stored experiences. We also create a
    # push_count attribute, which we initialize to 0, and we’ll use this to keep track of how many experiences we’ve
    # added to memory.
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    # need a way to store experiences in replay memory as they occur, so we define this push() function to do just that.

    # ush() accepts experience, and when we want to push a new experience into replay memory, we have to check first
    # that the amount of experiences we already have in memory is indeed less than the memory capacity. If it is,
    # then we append the experience to memory.
    #
    # If, on the other hand, the amount of experiences we have in memory has reached capacity, then we begin to push
    # new experiences onto the front of memory, overwriting the oldest experiences first. We then update our
    # push_count by incrementing by 1.
    #
    # Aside from storing experiences in replay memory, we also want to be able to sample experiences from replay
    # memory. Remember, these sampled experiences will be what we use to train our DQN.
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # We define this sample() function, which returns a random sample of experiences. The number of randomly sampled
    # experiences returned will be equal to the batch_size parameter passed to the function.
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Finally, we have this can_provide_sample() function that returns a boolean to tell us whether or not we can
    # sample from memory. Recall that the size of a sample we’ll obtain from memory will be equal to the batch size
    # we use to train our network.
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# This class will manage our cart and pole environment. It will wrap several of gym’s environment capabilities,
# and it will also give us some added functionality, like image preprocessing, for the environment images that will
# be given to our network as input.
class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    # Using this action, we call step() on the environment, which will execute the given action taken by the agent in
    # the environment.
    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    # The point of this function is to return the current state of the environment in the form of a processed image
    # of the screen. Remember, a deep Q-network takes states of the environment as input, and we previously mentioned
    # that for our environment, states would be represented using screenshot-like images.
    #
    # Actually, note that we will represent a single state in the environment as the difference between the current
    # screen and the previous screen. This will allow the agent to take the velocity of the pole into account from
    # one single image. So, a single state will be represented as a processed image of the difference between two
    # consecutive screens. We’ll see in a moment what type of processing is being done.
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    # This function first renders the environment as an RGB array using the render() function and then transposes
    # this array into the order of channels by height by width, which is what our PyTorch DQN will expect.
    #
    # This result is then cropped by passing it to the crop_screen() function, which we’ll cover next. We then pass
    # the cropped screen to the function transform_screen_data(), again, which we’ll cover in a moment, which just
    # does some final data conversion and rescaling to the cropped image.
    #
    # This transposed, cropped, and transformed version of the original screen returned by gym is what is returned by
    # get_processed_screen().
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))  # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    # The crop_screen() function accepts a screen and will return a cropped version of it. We first get the height of
    # the screen that was passed in, and then we strip off the top and bottom of the screen.
    #
    # We’ll see an example of a screen both before and after it’s been processed in a moment, and there you’ll see
    # how there is a lot of plain white space at the top and bottom of the cart and pole environment,
    # so we’re removing this empty space here. We set top equal to the value that corresponds to 40% of the
    # screen_height. Similarly, we set bottom equal to the value that corresponds to 80% of the screen_height.
    #
    # With these top and bottom values, we then take a slice of the screen starting from the top value down to the
    # bottom value so that we’ve essentially stripped off the top 40% of the original screen and the bottom 20%.
    def crop_screen(self, screen):
        screen_height = screen.shape[1]


        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    # Convert and rescale screen image data We first pass this screen to the numpy ascontiguousarray() function,
    # which returns a contiguous array of the same shape and content as screen, meaning that all the values of this
    # array will be stored sequentially next to each other in memory.
    #
    # We’re also converting the individual pixel values into type float32 and rescaling all the values by dividing
    # them each by 255. This is a common rescaling process that occurs during image processing for neural network
    # input.
    #
    # We then convert this array to a PyTorch tensor.
    #
    # We then use torchvision’s Compose class to chain together several image transformations. We’ll call this
    # compose resize. So, when a tensor is passed to resize, it will first be converted to a PIL image, then it will
    # be resized to a 40 x 90 image. The PIL image is then transformed to a tensor.
    #
    # So, we pass our screen from above to resize and then add an extra batch dimension to the tensor by calling
    # unsqueeze(). This result is then what is returned by the transform_screen_data() function.
    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Resize((40, 90))
            , T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)  # add a batch dimension (BCHW)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The first static method is get_current(). This function accepts a policy_net, states, and actions.
    # When we call this function in our main program, recall that these states and actions are the state-action pairs
    # that were sampled from replay memory. So, the states and actions correspond with each other.
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))






    # This function accepts a target_net and next_states. Recall that for each next state, we want to obtain the
    # maximum q-value predicted by the target_net among all possible next actions.
    #
    # To do that, we first look in our next_states tensor and find the locations of all the final states. If an
    # episode is ended by a given action, then we’re calling the next_state that occurs after that action was taken
    # the final state.
    #
    # Remember, last time we discussed that final states are represented with an all black screen. Therefore,
    # all the values within the tensor that represent that final state would be zero.
    #
    # We want to know where the final states are, if we even have any at all in a given batch, because we’re not
    # going to want to pass these final states to our target_net to get a predicted q-value. We know that the q-value
    # for final states is zero because the agent is unable to receive any reward once an episode has ended.
    #
    # So, we’re finding the locations of these final states so that we know not to pass them to the target_net for
    # q-value predictions when we pass our non-final next states.
    #
    @staticmethod
    def get_next(target_net, next_states):
        # To find the locations of these potential final states, we flatten the next_states tensor along dimension 1,
        # and we check each individual next state tensor to find its maximum value. If its maximum value is equal to 0,
        # then we know that this particular next state is a final state, and we represent that as a True within this
        # final_state_locations tensor. next_states that are not final are represented by a False value in the tensor.
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)

        # We then create a second tensor non_final_state_locations, which is just an exact opposite of
        # final_state_locations. It contains True for each location in the next_states tensor that corresponds to a
        # non-final state and a False for each location that corresponds to a final state.
        non_final_state_locations = (final_state_locations == False)

        # Now that we know the locations of the non-final states, we can now get the values of these states by indexing
        # into the next_states tensor and getting all of the corresponding non_final_states.
        non_final_states = next_states[non_final_state_locations]

        # Next, we find out the batch_size by checking to see how many next states are in the next_states tensor. Using
        # this, we create a new tensor of zeros that has a length equal to the batch size. We also send this tensor to
        # the device defined at the start of this class.
        #
        # then index into this tensor of zeros with the non_final_state_locations, and we set the corresponding values
        # for all of these locations equal to the maximum predicted q-values from the target_net across each action.
        #
        # This leaves us with a tensor that contains zeros as the q-values associated with any final state and contains
        # the target_net's maximum predicted q-value across all actions for each non-final state. This result is what is
        # finally returned by get_next().
        #
        # The whole point of all this code in this function was to find out if we have any final states in our
        # next_states tensor. If we do, then we need to find out where they are so that we don’t pass them to the
        # target_net. We don’t want to pass them to the target_net for a predicted q-value since we know that their
        # associated q-values will be zero.
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values




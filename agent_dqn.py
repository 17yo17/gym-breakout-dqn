#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""
import random
from collections import namedtuple
from itertools import count
from torch.utils.tensorboard import SummaryWriter

# Set up TensorBoard writer
writer = SummaryWriter("runs/PrioriDoubleDQN")

torch.manual_seed(595)
MY_MODEL = 'Double_Priori_10000epochs_32batch__EpsDecay9997_10000memory.pth'

N_EPISODES = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9997
TAU = 0.01
LR = 1e-4
MEMORY_SIZE = 1000000
GRADIENT_CLIP = 5
TARGET_UPDATE_FREQ = 5000
CHECKPOINT_FREQ = 5000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Determines the level of prioritization
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, experience, error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize for stability
        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error.item() + 1e-5) ** self.alpha
            
    def __len__(self):
        return len(self.buffer)
        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *arg):
        """Save a transition"""
        self.memory.append(Transition(*arg))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Initialize parameters for epsilon and beta
        self.epsilon = EPS_START
        self.epsilon_end = EPS_END
        self.epsilon_decay = EPS_DECAY
        self.beta = 0.4  # Starting beta for importance sampling
        self.total_loss = 0
        self.steps_done = 0

        # Initialize environment, Q networks, and replay buffer
        self.env = env
        observation = self.env.reset()
        n_channels = len(np.array(observation).transpose(2,0,1))
        n_actions = self.env.action_space.n
        
        # Use GPU if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Replay Buffer
        #self.memory = ReplayMemory(MEMORY_SIZE)
        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        
        # Q-Network and Target-Network
        self.Q_net = DQN(n_channels, self.env.action_space.n).to(self.device)
        self.target_net = DQN(n_channels, self.env.action_space.n).to(self.device)
        self.target_net.eval()
        self.target_net.load_state_dict(self.Q_net.state_dict())
        

        # Optimizer and Loss function
        self.optimizer = optim.AdamW(self.Q_net.parameters(), lr=LR, amsgrad=True)
        self.loss_fn = torch.nn.MSELoss()

        # Load my model if exists
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            try:
                self.Q_net.load_state_dict(torch.load('/Users/ryotaono/Downloads/cs696-project3/Double_Priori_10000epochs_32batch__LinearEpsDecay_10000memory.pth', weights_only=True))
                self.Q_net.eval()
            except:
                print("Didn't Work!!")
                pass
            #if args.load_checkpoint != False:
                # load model checkpoint
                #self.Q_net.load_state_dict(torch.load(f'checkpoints/test{args.load_checkpoint}.pt', map_location=self.device))
                #self.optimizer = optim.AdamW(self.Q_net.parameters(), lr=LR, amsgrad=True)
                #self.target_net.load_state_dict(self.Q_net.state_dict())
                #self.epsilon = 0.01


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    def make_action(self, state, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Convert state to tensor with correct shape and device if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        sample = random.random()
        if sample > self.epsilon or test:
            with torch.no_grad():
                action = self.Q_net(state).max(1).indices.view(1,1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
            
        ###########################
        #return action.item() # FOr TEST
        return action # For TRAIN
    
    def push(self, state, action, nextstate, reward):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        #self.memory.push(state, action, nextstate, reward)
        #error = abs(reward.item())  # Use the reward as a proxy for initial error
        #self.memory.add(Transition(state, action, nextstate, reward), error)

        # Use TD error as initial error if available, otherwise default to abs(reward)
        with torch.no_grad():
            current_q = self.Q_net(state).gather(1, action)
            next_q = self.target_net(nextstate).max(1)[0] if nextstate is not None else 0
            td_error = abs(reward + (GAMMA * next_q) - current_q).item()
        
        error = td_error if nextstate is not None else abs(reward.item())
        self.memory.add(Transition(state, action, nextstate, reward), error)
        ###########################
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.memory) < BATCH_SIZE:
            return
        # Sample a random batch from the buffer 
        #transitions = self.memory.sample(BATCH_SIZE)
        transitions, indices, weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Mask to identify non-final states (to ignore terminal states)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        # Concatenate all non-final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Concatenate states, actions, and rewards from minibatch
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q-values for current state-action pairs
        state_action_values = self.Q_net(state_batch).gather(1, action_batch)
        state_action_values = state_action_values.float()
        
        # Double DQN target: use Q_net to select action, target_net to evaluate
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_actions = self.Q_net(non_final_next_states).max(1)[1].unsqueeze(1)
        with torch.no_grad():
            #USE next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values.float()
        errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)
        
        # Weighted loss calculation using importance sampling weights
        weights = torch.tensor(weights, device=self.device, dtype=torch.float)
        #loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = (weights * self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.Q_net.parameters(), GRADIENT_CLIP) 
        self.optimizer.step()

        # Log the loss to TensorBoard
        writer.add_scalar("Loss", loss.item(), self.steps_done)

        self.total_loss += loss.item()
        ###########################
         
    def update_target_network(self):

        # Update target network by copying the weights from the Q network
        target_net_state_dict = self.target_net.state_dict()
        Q_net_state_dict = self.Q_net.state_dict()
                
        for key in Q_net_state_dict:
            target_net_state_dict[key] = Q_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    
        self.target_net.load_state_dict(target_net_state_dict)

    def save_checkpoint(self, model, optimizer, episode, filename="checkpoint.pth"):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at episode {episode}")

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        total_reward = 0
        for i in range(N_EPISODES):
            self.total_loss = 0
            self.beta = min(1.0, self.beta + (1.0 - 0.4) / N_EPISODES)  # Linearly increase beta
            state = self.env.reset()
            # Flatten the state
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            for t in count():
                self.steps_done += 1
                action = self.make_action(state, False)

                observation, reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                nextstate = None if terminated else torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                self.push(state, action, nextstate, reward)
                state = nextstate
                self.replay_buffer()
                
                total_reward += reward
                if done:
                    break

                # Update target network
                self.update_target_network()        
            
            # Log total reward to TensorBoard
            writer.add_scalar("Total Reward", total_reward, i)
            # Print progress every 10 episodes
            if (i+1) % 10 == 0:
                print(f"Episode {i+1}/{N_EPISODES}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}, Beta: {self.beta:.2f}")
                total_reward = 0

            # Save checkpoint periodically
            if (i+1) % CHECKPOINT_FREQ == 0:
                self.save_checkpoint(self.Q_net, self.optimizer, i, filename=f"checkpoint_{i}.pth")

            # Update epsilon linearly (max - min) / n_episodes
            #self.epsilon = max(EPS_END, self.epsilon - (EPS_START - EPS_END) / N_EPISODES)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        torch.save(self.Q_net.state_dict(), MY_MODEL)
        writer.flush()
        self.env.close()
        writer.close()
        # TO SEE the log: tensorboard --logdir=runs 
        ###########################

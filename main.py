import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import numpy as np
from collections import deque
from dataclasses import dataclass
from operator import itemgetter
import copy
from tqdm import tqdm
from random import sample
import random
import wandb

@dataclass
class Sarsd:
    state: any
    action: int
    reward: float
    next_state: any
    done: int

class DQN(nn.Module):
    def __init__(self, input_size, no_of_actions, learning_rate = 0.0001):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, no_of_actions))
        self.opt = optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, input):
        return self.net(input)
    
    def train(self, tgt, states, actions, rewards, next_states, mask, gamma = 1):
       # with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

        self.opt.zero_grad()
        qvals = self(states)  # (N, num_actions)
        one_hot_actions = F.one_hot(torch.LongTensor(actions), 2)
        loss = ((rewards + (1-mask) *qvals_next - torch.sum(qvals*one_hot_actions, -1))**2).mean()
        loss.backward()
        self.opt.step()
        return loss


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def insert(self, step):
        self.buffer.append(step)

    def sample(self, no_of_samples):
        #no_of_samples = min(no_of_samples, len(self.buffer))
        samples = sample(self.buffer, no_of_samples)
        states_sample = torch.tensor([sample.state for sample in samples], dtype=torch.float32)
        actions_sample = torch.tensor([sample.action for sample in samples], dtype=torch.int64)
        rewards_sample = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
        next_states_sample = torch.tensor([sample.next_state for sample in samples], dtype=torch.float32)
        done_sample = torch.tensor([sample.done for sample in samples], dtype=torch.int)
        return states_sample, actions_sample, rewards_sample, next_states_sample, done_sample

    
def update_model(model, target_model):
    target_model.load_state_dict(model.state_dict())

def main(train, PATH):
    pbar = tqdm()
    wandb.init()

    BATCH_SIZE = 2500
    MIN_RB_SIZE = 10000
    RB_CAPACITY = 100000
    TRAIN_AFTER_STEPS = 100
    TARGET_MODEL_UPDATE_FREQ = 500
    eps = 1.0
    EPS_MIN = 0.01
    EPS_DECAY = 0.999999

    train_steps = 0
    global_step = 0

    rolling_reward = 0
    steps_since_train = 0
    episodes_reward = []

    env = gym.make('CartPole-v1')
    last_obs = env.reset()
    loss = 0


    rb =  ReplayBuffer(RB_CAPACITY)
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model = DQN(env.observation_space.shape[0], env.action_space.n)
    update_model(model, target_model)

    if not train:
        model.load_state_dict(torch.load(PATH))
        #model.eval()


    while True:
        pbar.update(1)
        if not train:
            eps = 0
            env.render()
            time.sleep(0.05)


        if random.uniform(0, 1) < eps:
            action = env.action_space.sample()
        else:
            action = model(torch.Tensor(last_obs)).argmax().item()
  
        eps = EPS_DECAY**(global_step)
        obs, reward, done, _ = env.step(action)
        rolling_reward += reward

        rb.insert(Sarsd(last_obs, action, reward, obs, int(done)))
        last_obs = obs
        
        if done:
            episodes_reward.append(rolling_reward)
            rolling_reward = 0
            last_obs = env.reset()
        if train:
            steps_since_train += 1
            global_step += 1
            if len(rb.buffer) > MIN_RB_SIZE and steps_since_train > TRAIN_AFTER_STEPS:
                loss = model.train(target_model, *rb.sample(BATCH_SIZE))
                wandb.log({"loss": loss, "avg_reward": np.mean(episodes_reward), "eps": eps}, step=global_step)
                train_steps += 1
                if train_steps > TARGET_MODEL_UPDATE_FREQ:
                    print('updating target model')
                    update_model(model, target_model)
                    if np.mean(episodes_reward) > 180.0:
                        print('saved')
                        torch.save(model.state_dict(), f"models/{global_step}.pth")

                    train_steps = 0
                    #print(global_step, 'Loss: ', loss.detach().item(), 'avg_reward: ', np.mean(episodes_reward))
                steps_since_train = 0
                episodes_reward = []

    env.close()
    wandb.finish()


if __name__ == '__main__':
    main(True, './models/fit.pth')

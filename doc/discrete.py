from copy import Error
import gym
import tianshou as ts
from tianshou import policy
import torch
from torch import nn
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

env = gym.make('CartPole-v0')
# train_envs = gym.make('CartPole-v0')
# test_envs = gym.make('CartPole-v0')

train_envs = ts.env.VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(8)])
test_envs = ts.env.VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(100)])

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h

class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s):
        logits, h = self.preprocess(s, None)
        logits = self.last(logits)
        return logits

class DQN(nn.Module):

    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_shape)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.reshape(x.size(0), -1))
        return self.head(x), state


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

def build_policy(_p='PG'):
    if _p == 'PG':
        net = Net(3, state_shape, action_shape)
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        dist = torch.distributions.Categorical
        policy = ts.policy.PGPolicy(net, optim, dist, discount_factor=0.99)
    elif _p == 'PPO':
        net = Net(3, state_shape)
        actor = Actor(net, action_shape)
        critic = Critic(net)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=1e-3)
        dist = torch.distributions.Categorical
        policy = ts.policy.PPOPolicy(
            actor, critic, optim, dist,
            discount_factor=0.99, action_range=None)
    elif _p == 'DQN':
        # model
        net = Net(3, state_shape, action_shape)
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        policy = ts.policy.DQNPolicy(
            net, optim, 0.9, 3,
            use_target_network=320 > 0,
            target_update_freq=320)
    elif _p == 'A2C':
        net = Net(3, state_shape)
        actor = Actor(net, action_shape)
        critic = Critic(net)
        optim = torch.optim.Adam(list(
            actor.parameters()) + list(critic.parameters()), lr=1e-3)
        dist = torch.distributions.Categorical
        policy = ts.policy.A2CPolicy(
            actor, critic, optim, dist, 0.9, vf_coef=0.5,
            ent_coef=0.01, max_grad_norm=None)
    else:
        raise ValueError('No such policy in this file!')

    return policy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PG')
    # parser.add_argument('--task', type=str, default='CartPole-v0')
    # parser.add_argument('--seed', type=int, default=1626)
    # parser.add_argument('--buffer-size', type=int, default=20000)
    # parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--training-num', type=int, default=32)
    parser.add_argument('--test-num', type=int, default=100)
    # parser.add_argument('--logdir', type=str, default='log')
    # parser.add_argument('--render', type=float, default=0.)
    # parser.add_argument(
    #     '--device', type=str,
    #     default='cuda' if torch.cuda.is_available() else 'cpu')
    # # a2c special
    # parser.add_argument('--vf-coef', type=float, default=0.5)
    # parser.add_argument('--ent-coef', type=float, default=0.001)
    # parser.add_argument('--max-grad-norm', type=float, default=None)
    args = parser.parse_known_args()[0]
    return args


def stop_fn(x):
    return x >= env.spec.reward_threshold


if __name__ == "__main__":
    args = get_args()
    policy = build_policy(args.policy)
    writer = SummaryWriter('log' + '/' + args.policy + '/' + time.strftime("%d_%Y_%H_%M_%S"))
    train_collector = ts.data.Collector(
            policy, train_envs, ts.data.ReplayBuffer(2000))
    test_collector = ts.data.Collector(policy, test_envs)
    result = ts.trainer.onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, writer=writer)
    assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()

# -*- coding: utf-8 -*-
"""
HW3-2: Enhanced DQN Variants for Player Mode [40%]
=====================================================
實作並比較:
1. Standard DQN (Experience Replay) — 基準
2. Double DQN — 解決 Q 值高估問題
3. Dueling DQN — 分離狀態價值與動作優勢

環境模式: player (Player 隨機, 其他固定)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import copy
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Gridworld import Gridworld

# ============================================================
# 超參數
# ============================================================
STATE_DIM = 64
NUM_ACTIONS = 4
HIDDEN1 = 150
HIDDEN2 = 100

EPOCHS = 2000
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.1
LEARNING_RATE = 1e-3
BUFFER_SIZE = 2000
BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 50  # 每 N 個 epoch 更新 target 網路

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


# ============================================================
# Experience Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.cat(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# 網路架構
# ============================================================

class StandardQNetwork(nn.Module):
    """標準 Q 網路 (用於 Standard DQN 和 Double DQN)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, NUM_ACTIONS)
        )
    
    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN 網路架構
    
    核心概念:
    - 將 Q(s,a) 分解為 V(s) + A(s,a)
    - V(s): 狀態價值 — 不管採取什麼動作，這個狀態本身的好壞
    - A(s,a): 動作優勢 — 在這個狀態下，某個動作比平均好多少
    
    好處:
    - 有些狀態不管怎麼動都差不多（例如遠離目標），
      這時只學 V(s) 就夠了，不需要精確估計每個動作
    - 加速學習，因為 V(s) 的更新對所有動作都有效
    
    合併公式:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    減去平均值是為了唯一性（identifiability）
    """
    def __init__(self):
        super().__init__()
        # 共享特徵提取層
        self.feature = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN1),
            nn.ReLU(),
        )
        # Value Stream: 輸出 1 個值 — V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1)
        )
        # Advantage Stream: 輸出 N 個值 — A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, NUM_ACTIONS)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)          # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, num_actions]
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# ============================================================
# 通用訓練器
# ============================================================
class DQNTrainer:
    def __init__(self, network_type='standard', use_double=False):
        """
        network_type: 'standard' 或 'dueling'
        use_double: 是否使用 Double DQN 更新
        """
        self.use_double = use_double
        
        if network_type == 'dueling':
            self.online_net = DuelingQNetwork()
            self.target_net = DuelingQNetwork()
        else:
            self.online_net = StandardQNetwork()
            self.target_net = StandardQNetwork()
        
        # 初始化 target 網路 = online 網路
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return np.random.randint(0, NUM_ACTIONS)
        with torch.no_grad():
            q_values = self.online_net(state)
        return np.argmax(q_values.data.numpy())
    
    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # 當前 Q 值
        q_values = self.online_net(states)
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            if self.use_double:
                # ===== Double DQN =====
                # 關鍵差異: 用 online 網路選動作，用 target 網路評估 Q 值
                # 這解決了 Q 值高估問題:
                #   標準 DQN: max_a Q_target(s', a) 
                #     → 同一個網路既選又估，容易過度樂觀
                #   Double DQN: Q_target(s', argmax_a Q_online(s', a))
                #     → 分離選擇和評估，減少高估
                best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                max_next_q = self.target_net(next_states).gather(1, best_actions).squeeze()
            else:
                # ===== Standard DQN =====
                max_next_q = self.target_net(next_states).max(dim=1)[0]
        
        target = rewards + GAMMA * max_next_q * (1 - dones)
        
        loss = self.loss_fn(q_action, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        """將 online 網路的權重複製到 target 網路"""
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def train(self, mode='player', epochs=EPOCHS, verbose=True):
        epsilon = EPSILON_START
        losses = []
        
        for epoch in range(epochs):
            game = Gridworld(size=4, mode=mode)
            state = torch.from_numpy(
                game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            ).float()
            
            total_loss = 0
            steps = 0
            done = False
            
            while not done:
                action_idx = self.select_action(state, epsilon)
                game.makeMove(ACTION_SET[action_idx])
                
                next_state = torch.from_numpy(
                    game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
                ).float()
                
                reward = game.reward()
                done = abs(reward) == 10
                
                self.replay_buffer.push(state, action_idx, reward, next_state, float(done))
                state = next_state
                
                loss = self.update()
                total_loss += loss
                steps += 1
                
                if steps > 50:
                    done = True
            
            avg_loss = total_loss / max(steps, 1)
            losses.append(avg_loss)
            
            # 定期更新 target 網路
            if epoch % TARGET_UPDATE_FREQ == 0:
                self.update_target()
            
            if epsilon > EPSILON_MIN:
                epsilon -= (1 / epochs)
            
            if verbose and epoch % 200 == 0:
                print("  Epoch %4d/%d | Loss: %.4f | Eps: %.3f" % (epoch, epochs, avg_loss, epsilon))
        
        return losses
    
    def evaluate(self, mode='player', num_games=100):
        wins = 0
        for _ in range(num_games):
            game = Gridworld(size=4, mode=mode)
            state = torch.from_numpy(
                game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            ).float()
            
            for step in range(15):
                with torch.no_grad():
                    q_values = self.online_net(state)
                action_idx = q_values.argmax().item()
                game.makeMove(ACTION_SET[action_idx])
                
                state = torch.from_numpy(
                    game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
                ).float()
                
                reward = game.reward()
                if reward != -1:
                    if reward > 0:
                        wins += 1
                    break
        
        win_rate = wins / num_games * 100
        return win_rate


# ============================================================
# 主程式: 三種方法比較
# ============================================================
if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("HW3-2: DQN Variants Comparison (Player Mode)")
    print("=" * 60)
    
    MODE = 'player'
    TRAIN_EPOCHS = 2000
    
    configs = [
        ("Standard DQN",   'standard', False),
        ("Double DQN",     'standard', True),
        ("Dueling DQN",    'dueling',  False),
    ]
    
    all_results = {}
    
    for name, net_type, use_double in configs:
        print("\n" + "=" * 50)
        print("[Train] %s" % name)
        print("=" * 50)
        
        trainer = DQNTrainer(network_type=net_type, use_double=use_double)
        losses = trainer.train(mode=MODE, epochs=TRAIN_EPOCHS)
        
        results = {}
        for test_mode in ['static', 'player', 'random']:
            wr = trainer.evaluate(mode=test_mode, num_games=200)
            print("  [Result] %-8s win rate: %.1f%%" % (test_mode, wr))
            results[test_mode] = wr
        
        all_results[name] = {'losses': losses, 'win_rates': results}
    
    # ============================================================
    # 繪製比較圖
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss 曲線
    colors = ['#2196F3', '#E91E63', '#4CAF50']
    for (name, _, _), color in zip(configs, colors):
        # 使用移動平均平滑
        losses = all_results[name]['losses']
        window = 50
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, label=name, alpha=0.8, color=color)
    
    axes[0].set_title('Training Loss Comparison (Player Mode)', fontsize=13)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (Moving Avg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 勝率柱狀圖
    modes = ['static', 'player', 'random']
    x = np.arange(len(modes))
    width = 0.25
    
    for i, ((name, _, _), color) in enumerate(zip(configs, colors)):
        wr = [all_results[name]['win_rates'][m] for m in modes]
        axes[1].bar(x + i * width, wr, width, label=name, color=color, alpha=0.8)
    
    axes[1].set_title('Win Rate Comparison', fontsize=13)
    axes[1].set_xlabel('Game Mode')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([m.capitalize() for m in modes])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('hw3_2_dqn_comparison.png', dpi=150)
    print("\nComparison chart saved to hw3_2_dqn_comparison.png")

# -*- coding: utf-8 -*-
"""
HW3-1: DQN with Experience Replay Buffer [30%]
=================================================
改進版 DQN，加入 Experience Replay Buffer。
解決 Naive DQN 的兩個核心問題:
1. 連續經驗高度相關 → 隨機取樣打破相關性
2. 只使用最新經驗 → 重複利用過去經驗
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
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Gridworld import Gridworld

# ============================================================
# 超參數
# ============================================================
L1, L2, L3, L4 = 64, 150, 100, 4
EPOCHS = 1000
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.1
LEARNING_RATE = 1e-3
BUFFER_SIZE = 1000   # Experience Replay Buffer 容量
BATCH_SIZE = 200     # 每次從 buffer 取樣的批次大小
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


# ============================================================
# Experience Replay Buffer
# ============================================================
class ExperienceReplayBuffer:
    """
    經驗回放緩衝區
    
    核心概念:
    - 儲存 agent 的經驗 (s, a, r, s', done)
    - 訓練時隨機取樣一個 mini-batch
    - 打破經驗之間的時間相關性
    - 讓每筆經驗可以被多次使用
    """
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """儲存一筆經驗"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """隨機取樣一個 mini-batch"""
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
# Q 網路
# ============================================================
def build_model():
    return nn.Sequential(
        nn.Linear(L1, L2), nn.ReLU(),
        nn.Linear(L2, L3), nn.ReLU(),
        nn.Linear(L3, L4)
    )


# ============================================================
# 訓練函數 (帶 Experience Replay)
# ============================================================
def train_dqn_with_replay(mode='static', epochs=EPOCHS, verbose=True):
    """
    帶 Experience Replay 的 DQN 訓練
    
    與 Naive DQN 的關鍵差異:
    1. 每步經驗先存入 buffer
    2. 從 buffer 隨機取樣 batch 來訓練
    3. 一次更新使用多筆不相關的經驗
    """
    model = build_model()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ExperienceReplayBuffer(BUFFER_SIZE)
    
    epsilon = EPSILON_START
    losses = []
    
    for epoch in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state = torch.from_numpy(
            game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        ).float()
        
        status = 1
        epoch_loss = 0
        steps = 0
        
        while status == 1:
            # ε-greedy 選擇動作
            qval = model(state)
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = np.argmax(qval.data.numpy())
            
            game.makeMove(ACTION_SET[action_idx])
            
            next_state = torch.from_numpy(
                game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            ).float()
            
            reward = game.reward()
            done = abs(reward) == 10
            
            # 📌 關鍵: 將經驗存入 buffer
            replay_buffer.push(state, action_idx, reward, next_state, float(done))
            
            state = next_state
            
            # 📌 關鍵: 從 buffer 取樣訓練
            if len(replay_buffer) >= BATCH_SIZE:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                    replay_buffer.sample(BATCH_SIZE)
                
                # 計算當前 Q 值
                q_values = model(batch_states)
                q_action = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
                # 計算目標 Q 值
                with torch.no_grad():
                    next_q_values = model(batch_next_states)
                    max_next_q = torch.max(next_q_values, dim=1)[0]
                
                # 若遊戲結束，目標就是 reward；否則 reward + γ * max Q(s')
                target = batch_rewards + GAMMA * max_next_q * (1 - batch_dones)
                
                loss = loss_fn(q_action, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss = loss.item()
            
            if done:
                status = 0
            steps += 1
            if steps > 50:  # 防止無限迴圈
                status = 0
        
        losses.append(epoch_loss)
        
        if epsilon > EPSILON_MIN:
            epsilon -= (1 / epochs)
        if verbose and epoch % 200 == 0:
            print("  Epoch %4d/%d | Loss: %.4f | Eps: %.3f | Buffer: %d" %
                  (epoch, epochs, epoch_loss, epsilon, len(replay_buffer)))
    
    return model, losses


# ============================================================
# 測試與評估
# ============================================================
def test_model(model, mode='static', display=False, max_moves=15):
    game = Gridworld(size=4, mode=mode)
    state = torch.from_numpy(
        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    ).float()
    
    for i in range(max_moves):
        qval = model(state)
        action_idx = np.argmax(qval.data.numpy())
        game.makeMove(ACTION_SET[action_idx])
        
        state = torch.from_numpy(
            game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        ).float()
        
        reward = game.reward()
        if reward != -1:
            return reward > 0
    return False


def evaluate_model(model, mode='static', num_games=100):
    wins = sum(test_model(model, mode=mode) for _ in range(num_games))
    win_rate = wins / num_games * 100
    print("  Mode: %-8s | Win rate: %d/%d = %.1f%%" % (mode, wins, num_games, win_rate))
    return win_rate


# ============================================================
# 主程式
# ============================================================
if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("HW3-1: DQN with Experience Replay Buffer")
    print("=" * 60)
    
    results = {}
    
    for mode in ['static', 'player', 'random']:
        print("\n[Training] Mode: %s" % mode)
        print("-" * 40)
        model, losses = train_dqn_with_replay(mode=mode, epochs=1000)
        
        print("\n[Test] %s mode:" % mode)
        wr = evaluate_model(model, mode=mode, num_games=100)
        results[mode] = {'model': model, 'losses': losses, 'win_rate': wr}
    
    # 繪製 Loss 曲線比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for idx, mode in enumerate(['static', 'player', 'random']):
        axes[idx].plot(results[mode]['losses'], alpha=0.7, color=colors[idx])
        axes[idx].set_title(f'Experience Replay DQN - {mode.capitalize()} Mode\n'
                           f'Win Rate: {results[mode]["win_rate"]:.1f}%', fontsize=12)
        axes[idx].set_xlabel('Epochs')
        axes[idx].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('hw3_1_replay_dqn_loss.png', dpi=150)
    print("\nLoss curves saved to hw3_1_replay_dqn_loss.png")

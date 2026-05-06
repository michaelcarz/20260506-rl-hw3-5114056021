# -*- coding: utf-8 -*-
"""
HW3-1: Naive DQN for Static Mode [30%]
========================================
基本 DQN 實作，不包含 Experience Replay Buffer。
在 static mode 下訓練，觀察學習效果。

環境: Gridworld 4x4
- 狀態: 4x4x4 → 展平為 64 維向量
- 動作: 4 個 (上/下/左/右)
- 獎勵: Goal=+10, Pit=-10, 其他=-1
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
import os
import sys

# 確保可以 import Gridworld
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Gridworld import Gridworld

# ============================================================
# 1. 超參數設定
# ============================================================
L1 = 64    # 輸入層: 4x4x4 = 64 (棋盤狀態展平)
L2 = 150   # 第一隱藏層
L3 = 100   # 第二隱藏層
L4 = 4     # 輸出層: 4 個動作的 Q 值

EPOCHS = 1000
GAMMA = 0.9         # 折扣因子: 未來獎勵的衰減率
EPSILON_START = 1.0  # ε-greedy 的初始探索率
EPSILON_MIN = 0.1    # 最低探索率
LEARNING_RATE = 1e-3

# 動作對應表
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# ============================================================
# 2. Q 網路定義
# ============================================================
def build_model():
    """
    建立 Q 網路 (全連接神經網路)
    架構: 64 → 150 → ReLU → 100 → ReLU → 4
    
    輸入: 64 維狀態向量 (4x4x4 棋盤展平)
    輸出: 4 個 Q 值，對應 4 個動作
    """
    model = nn.Sequential(
        nn.Linear(L1, L2),   # 第一層: 64 → 150
        nn.ReLU(),
        nn.Linear(L2, L3),   # 第二層: 150 → 100
        nn.ReLU(),
        nn.Linear(L3, L4)    # 輸出層: 100 → 4
    )
    return model


# ============================================================
# 3. 訓練函數
# ============================================================
def train_naive_dqn(mode='static', epochs=EPOCHS, verbose=True):
    """
    Naive DQN 訓練迴圈（無 Experience Replay）
    
    每一步直接用「最新的一筆經驗」更新網路，
    這導致：
    1. 連續的經驗高度相關 → 學習不穩定
    2. 只看到最新經驗，忘記過去的經驗
    """
    model = build_model()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    epsilon = EPSILON_START
    losses = []
    
    for epoch in range(epochs):
        # 每個 epoch 開一局新遊戲
        game = Gridworld(size=4, mode=mode)
        
        # 取得初始狀態並加入少量雜訊（防止過擬合固定棋盤）
        state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state).float()
        
        status = 1  # 1=遊戲進行中
        
        while status == 1:
            # ε-greedy 選擇動作
            qval = model(state)
            qval_np = qval.data.numpy()
            
            if random.random() < epsilon:
                action_idx = np.random.randint(0, 4)  # 探索: 隨機動作
            else:
                action_idx = np.argmax(qval_np)        # 利用: 選 Q 值最大的動作
            
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            
            # 取得新狀態
            state2 = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2 = torch.from_numpy(state2).float()
            
            reward = game.reward()
            
            # 計算目標 Q 值
            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)
            
            if reward == -1:
                # 遊戲未結束: Q = r + γ * max_a' Q(s', a')
                Y = reward + (GAMMA * maxQ)
            else:
                # 遊戲結束（勝利或落敗）: Q = r
                Y = torch.tensor([reward]).float()
            
            Y = torch.tensor([Y]).detach()
            X = qval.squeeze()[action_idx]  # 當前動作的預測 Q 值
            
            loss = loss_fn(X, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = state2
            
            if abs(reward) == 10:
                status = 0  # 遊戲結束
        
        losses.append(loss.item())
        
        # 遞減 ε (探索率)
        if epsilon > EPSILON_MIN:
            epsilon -= (1 / epochs)
        
        if verbose and epoch % 200 == 0:
            print("Epoch %4d/%d | Loss: %.4f | Epsilon: %.3f" % (epoch, epochs, loss.item(), epsilon))
    
    return model, losses


# ============================================================
# 4. 測試函數
# ============================================================
def test_model(model, mode='static', display=True, max_moves=15):
    """測試訓練好的模型"""
    game = Gridworld(size=4, mode=mode)
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state).float()
    
    if display:
        print("初始狀態:")
        print(game.display())
    
    for i in range(max_moves):
        qval = model(state)
        action_idx = np.argmax(qval.data.numpy())
        action = ACTION_SET[action_idx]
        
        if display:
            print(f"步驟 {i}: 動作={action}")
        
        game.makeMove(action)
        state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state).float()
        
        if display:
            print(game.display())
        
        reward = game.reward()
        if reward != -1:
            if reward > 0:
                if display:
                    print("WIN! Reward: %s" % reward)
                return True
            else:
                if display:
                    print("LOST! Reward: %s" % reward)
                return False
    
    if display:
        print("Too many moves, LOST.")
    return False


def evaluate_model(model, mode='static', num_games=100):
    """統計勝率"""
    wins = sum(test_model(model, mode=mode, display=False) for _ in range(num_games))
    win_rate = wins / num_games * 100
    print("Mode: %s | Win rate: %d/%d = %.1f%%" % (mode, wins, num_games, win_rate))
    return win_rate


# ============================================================
# 5. 主程式
# ============================================================
if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("HW3-1: Naive DQN (No Experience Replay)")
    print("=" * 60)
    
    # --- Exp 1: Static Mode ---
    print("\n[Exp 1] Static Mode Training")
    print("-" * 40)
    model_static, losses_static = train_naive_dqn(mode='static', epochs=1000)
    
    print("\n[Test] Static Mode:")
    evaluate_model(model_static, mode='static', num_games=100)
    
    print("\n[Test] Static model on random mode:")
    evaluate_model(model_static, mode='random', num_games=100)
    
    # --- Exp 2: Random Mode ---
    print("\n[Exp 2] Random Mode Training (Naive DQN)")
    print("-" * 40)
    model_random, losses_random = train_naive_dqn(mode='random', epochs=1000)
    
    print("\n[Test] Random Mode:")
    evaluate_model(model_random, mode='random', num_games=100)
    
    # --- 繪製 Loss 曲線 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(losses_static, alpha=0.7)
    axes[0].set_title('Naive DQN - Static Mode Loss', fontsize=13)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(losses_random, alpha=0.7, color='orange')
    axes[1].set_title('Naive DQN - Random Mode Loss', fontsize=13)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('hw3_1_naive_dqn_loss.png', dpi=150)
    print("\nLoss curves saved to hw3_1_naive_dqn_loss.png")

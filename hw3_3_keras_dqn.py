# -*- coding: utf-8 -*-
"""
HW3-3: Keras DQN with Training Tips for Random Mode [30%]
===========================================================
將 PyTorch DQN 轉換為 Keras (TensorFlow) 實作
並加入多種訓練技巧以穩定/提升 random mode 下的學習效果

訓練技巧:
1. Target Network (目標網路)
2. Gradient Clipping (梯度裁剪)
3. Learning Rate Scheduling (學習率調度)
4. Soft Update (軟更新 target 網路)
5. Huber Loss (比 MSE 更穩健)
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os, sys

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TF 訊息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Gridworld import Gridworld

# ============================================================
# 超參數
# ============================================================
STATE_DIM = 64
NUM_ACTIONS = 4
HIDDEN1 = 150
HIDDEN2 = 100

EPOCHS = 1500
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 1 / 1200
LEARNING_RATE = 1e-3
BUFFER_SIZE = 5000
BATCH_SIZE = 256
TAU = 0.01            # Soft update 的混合比例
LR_DECAY_RATE = 0.995  # 每 100 epoch 乘以此值

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
            np.vstack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# Keras Dueling DQN 模型
# ============================================================
def build_dueling_dqn():
    """
    使用 Keras Functional API 建立 Dueling DQN
    
    架構:
    Input(64) → Dense(150, ReLU) → [分支]
      ├→ Value:     Dense(100, ReLU) → Dense(1)       → V(s)
      └→ Advantage: Dense(100, ReLU) → Dense(4)       → A(s,a)
    Output: Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    inputs = keras.Input(shape=(STATE_DIM,))
    
    # 共享特徵層
    x = layers.Dense(HIDDEN1, activation='relu', name='shared')(inputs)
    
    # Value Stream
    v = layers.Dense(HIDDEN2, activation='relu', name='value_hidden')(x)
    v = layers.Dense(1, name='value')(v)
    
    # Advantage Stream
    a = layers.Dense(HIDDEN2, activation='relu', name='advantage_hidden')(x)
    a = layers.Dense(NUM_ACTIONS, name='advantage')(a)
    
    # 合併: Q = V + (A - mean(A))
    # 使用 Lambda 層實現
    q = layers.Lambda(
        lambda va: va[0] + (va[1] - tf.reduce_mean(va[1], axis=1, keepdims=True)),
        name='q_values'
    )([v, a])
    
    model = keras.Model(inputs=inputs, outputs=q)
    return model


# ============================================================
# Keras DQN Agent (含所有訓練技巧)
# ============================================================
class KerasDQNAgent:
    def __init__(self):
        # Training Tip 1: Dual Network (Target Network)
        self.online_net = build_dueling_dqn()
        self.target_net = build_dueling_dqn()
        self.target_net.set_weights(self.online_net.get_weights())
        
        # Training Tip 2: Learning Rate Scheduling
        self.lr = LEARNING_RATE
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=500,
            decay_rate=LR_DECAY_RATE,
            staircase=True
        )
        
        # Training Tip 3: Gradient Clipping
        # clipnorm=1.0: 將梯度的 L2 範數限制在 1.0
        # 防止梯度爆炸，穩定訓練
        self.optimizer = optimizers.Adam(
            learning_rate=self.lr_schedule,
            clipnorm=1.0  # ← 梯度裁剪
        )
        
        # Training Tip 4: Huber Loss
        # MSE 對大誤差的懲罰太強 → 梯度不穩定
        # Huber Loss = MSE (小誤差) + MAE (大誤差)
        self.loss_fn = losses.Huber(delta=1.0)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(0, NUM_ACTIONS)
        # Use __call__ instead of predict() for speed (avoid session overhead)
        state_tensor = tf.constant(state, dtype=tf.float32)
        q_values = self.online_net(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        單步訓練 (使用 tf.function 加速)
        
        結合 Double DQN + Dueling DQN + Gradient Clipping + Huber Loss
        """
        # Double DQN: online 選動作, target 估值
        next_q_online = self.online_net(next_states, training=False)
        best_actions = tf.argmax(next_q_online, axis=1)
        next_q_target = self.target_net(next_states, training=False)
        
        # 用 tf.one_hot 取出對應動作的 Q 值
        best_next_q = tf.reduce_sum(
            next_q_target * tf.one_hot(best_actions, NUM_ACTIONS), axis=1
        )
        
        # 目標 Q 值
        targets = rewards + GAMMA * best_next_q * (1.0 - dones)
        
        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            action_masks = tf.one_hot(actions, NUM_ACTIONS)
            q_action = tf.reduce_sum(q_values * action_masks, axis=1)
            
            loss = self.loss_fn(targets, q_action)
        
        # Gradient clipping is set in optimizer
        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))
        
        return loss
    
    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.int32)
        rewards = tf.constant(rewards, dtype=tf.float32)
        next_states = tf.constant(next_states, dtype=tf.float32)
        dones = tf.constant(dones, dtype=tf.float32)
        
        loss = self._train_step(states, actions, rewards, next_states, dones)
        return loss.numpy()
    
    def soft_update_target(self):
        """
        Training Tip 5: Soft Update
        
        θ_target = τ * θ_online + (1 - τ) * θ_target
        
        不像 Hard Update 直接複製，Soft Update 漸進式更新
        讓 target 網路的變化更平滑 → 訓練更穩定
        """
        online_weights = self.online_net.get_weights()
        target_weights = self.target_net.get_weights()
        
        new_weights = []
        for ow, tw in zip(online_weights, target_weights):
            new_weights.append(TAU * ow + (1 - TAU) * tw)
        
        self.target_net.set_weights(new_weights)
    
    def train(self, mode='random', epochs=EPOCHS, verbose=True):
        losses = []
        win_rates = []
        
        for epoch in range(epochs):
            game = Gridworld(size=4, mode=mode)
            state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            
            total_loss = 0
            steps = 0
            done = False
            
            while not done:
                action_idx = self.select_action(state)
                game.makeMove(ACTION_SET[action_idx])
                
                next_state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
                reward = game.reward()
                done = abs(reward) == 10
                
                self.replay_buffer.push(state, action_idx, reward, next_state, float(done))
                state = next_state
                
                loss = self.update()
                total_loss += loss
                steps += 1
                
                # Soft update every step
                if steps % 5 == 0:
                    self.soft_update_target()
                
                if steps > 50:
                    done = True
            
            avg_loss = total_loss / max(steps, 1)
            losses.append(avg_loss)
            
            # Epsilon decay
            if self.epsilon > EPSILON_MIN:
                self.epsilon -= EPSILON_DECAY
                self.epsilon = max(self.epsilon, EPSILON_MIN)
            
            # Periodic evaluation
            if epoch % 500 == 0 and epoch > 0:
                wr = self.evaluate(mode=mode, num_games=50)
                win_rates.append((epoch, wr))
                if verbose:
                    try:
                        lr_val = float(self.lr_schedule(self.optimizer.iterations))
                    except:
                        lr_val = float(self.optimizer.learning_rate)
                    print("  Epoch %4d/%d | Loss: %.4f | Eps: %.3f | LR: %.6f | Win: %.1f%%" % 
                          (epoch, epochs, avg_loss, self.epsilon, lr_val, wr), flush=True)
            elif verbose and epoch % 100 == 0:
                try:
                    lr_val = float(self.lr_schedule(self.optimizer.iterations))
                except:
                    lr_val = float(self.optimizer.learning_rate)
                print("  Epoch %4d/%d | Loss: %.4f | Eps: %.3f | LR: %.6f" % 
                      (epoch, epochs, avg_loss, self.epsilon, lr_val), flush=True)
        
        return losses, win_rates
    
    def evaluate(self, mode='random', num_games=50):
        wins = 0
        for _ in range(num_games):
            game = Gridworld(size=4, mode=mode)
            state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            
            for step in range(15):
                state_tensor = tf.constant(state, dtype=tf.float32)
                q_values = self.online_net(state_tensor, training=False)
                action_idx = int(tf.argmax(q_values[0]).numpy())
                game.makeMove(ACTION_SET[action_idx])
                
                state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
                reward = game.reward()
                if reward != -1:
                    if reward > 0:
                        wins += 1
                    break
        
        return wins / num_games * 100


# ============================================================
# 主程式
# ============================================================
if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("HW3-3: Keras Dueling Double DQN + Training Tips")
    print("        (Random Mode)")
    print("=" * 60)
    print()
    print("Training Tips:")
    print("  1. Target Network")
    print("  2. Double DQN")
    print("  3. Dueling DQN")
    print("  4. Gradient Clipping (clipnorm=1.0)")
    print("  5. Learning Rate Scheduling (Exponential Decay)")
    print("  6. Soft Update (tau=0.01)")
    print("  7. Huber Loss")
    print()
    
    agent = KerasDQNAgent()
    
    print("[Start] Training (Random Mode)...")
    print("-" * 50)
    losses, win_rates = agent.train(mode='random', epochs=EPOCHS)
    
    print("\n" + "=" * 50)
    print("[Final Test Results]")
    print("=" * 50)
    for mode in ['static', 'player', 'random']:
        wr = agent.evaluate(mode=mode, num_games=100)
        print("  %-8s win rate: %.1f%%" % (mode, wr), flush=True)
    
    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲線 (平滑)
    window = 100
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, color='#E91E63', alpha=0.8)
    else:
        axes[0].plot(losses, color='#E91E63', alpha=0.8)
    axes[0].set_title('Keras Dueling Double DQN\nTraining Loss (Random Mode)', fontsize=13)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (Moving Avg)')
    axes[0].grid(True, alpha=0.3)
    
    # 勝率變化
    if win_rates:
        wr_epochs, wr_values = zip(*win_rates)
        axes[1].plot(wr_epochs, wr_values, 'o-', color='#4CAF50', markersize=8)
        axes[1].set_title('Win Rate Progress (Random Mode)', fontsize=13)
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hw3_3_keras_dqn.png', dpi=150)
    print("\nCharts saved to hw3_3_keras_dqn.png")

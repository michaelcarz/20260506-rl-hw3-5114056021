# HW3-1 理解報告：Naive DQN 與 Experience Replay DQN

---

## 一、前言

本報告針對深度強化學習課程提供的程式碼進行分析與實驗，涵蓋 **Naive DQN**（程式 3.2~3.4）以及 **Experience Replay Buffer DQN**（程式 3.5~3.7）兩種方法。透過在 Gridworld 4×4 環境下的實驗，驗證兩者的學習效果差異，並深入理解 DQN 的核心機制。

---

## 二、環境說明

Gridworld 是一個 4×4 的網格世界，包含四個物件：

| 物件 | 符號 | 說明 |
|------|------|------|
| Player | P | 代理人（由 DQN 控制） |
| Goal | + | 目標位置，到達獲得 +10 獎勵 |
| Pit | - | 陷阱位置，掉入獲得 -10 獎勵 |
| Wall | W | 牆壁，無法通過 |

每走一步獲得 -1 的獎勵，鼓勵 agent 盡快到達目標。

**三種模式：**
- **Static**：所有物件位置固定（Player 在 (0,3)，Goal 在 (0,0)）
- **Player**：只有 Player 位置隨機，其餘固定
- **Random**：所有物件位置皆隨機

**狀態表示：** 棋盤以 4×4×4 的三維陣列表示（4 個 channel 分別代表 Player、Goal、Pit、Wall 的位置），展平為長度 64 的向量作為神經網路輸入。

---

## 三、DQN 核心機制解析

### 3.1 Q 網路架構

程式碼使用三層全連接神經網路作為 Q 值的函數近似器：

```
輸入層 (64) → 隱藏層 (150, ReLU) → 隱藏層 (100, ReLU) → 輸出層 (4)
```

- **輸入**：64 維狀態向量（4×4×4 棋盤展平）
- **輸出**：4 個 Q 值，分別對應上、下、左、右四個動作
- **優化器**：Adam（學習率 lr = 0.001）
- **損失函數**：MSE（均方誤差）

### 3.2 Q-Learning 更新公式

DQN 的核心是 Bellman 方程式：

$$Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a')$$

在神經網路的框架下：
- **預測值（X）**：網路對當前狀態 s 和動作 a 的 Q 值輸出 → `Q(s, a)`
- **目標值（Y）**：
  - 若遊戲未結束：`Y = r + γ × max Q(s', a')`，其中 γ = 0.9
  - 若遊戲結束（到達 Goal 或掉入 Pit）：`Y = r`
- **損失**：`Loss = MSE(X, Y)`

透過最小化此損失來更新網路權重，使 Q 值逐漸逼近真實的動作價值。

### 3.3 ε-greedy 探索策略

```python
if random.random() < epsilon:
    action = 隨機選擇動作       # 探索 (Exploration)
else:
    action = argmax Q(s, a)     # 利用 (Exploitation)
```

- **初始 ε = 1.0**：完全隨機探索，確保 agent 能接觸到各種狀態
- **逐步遞減**：每個 epoch 減少 `1/epochs`
- **最低 ε = 0.1**：保留 10% 的探索機率，避免陷入局部最優

**目的**：在訓練初期大量探索以收集多樣經驗，後期主要利用已學到的策略。

### 3.4 torch.no_grad() 的作用

```python
with torch.no_grad():
    newQ = model(state2)
```

計算目標 Q 值時使用 `torch.no_grad()`：
- 目標值僅作為參考標準，不需要計算梯度
- 節省記憶體和計算資源
- 避免目標值的計算影響到梯度更新的方向

### 3.5 折扣因子 γ（Gamma = 0.9）

γ 控制 agent 對未來獎勵的重視程度：
- γ = 0.9 表示下一步的獎勵價值為當前的 90%
- γ 越大，agent 越會考慮長期收益（有遠見）
- γ 越小，agent 越注重眼前獎勵（短視）

在 Gridworld 中，γ = 0.9 使 agent 有足夠的動機穿越多步到達 Goal，而不是只避開 Pit。

---

## 四、實驗結果與分析

### 4.1 實驗一：Naive DQN

Naive DQN 的特點是**每一步直接用最新的一筆經驗更新網路**，不儲存歷史經驗。

| 訓練模式 | 測試模式 | 勝率 |
|---------|---------|:----:|
| Static (1000 epochs) | Static | **100%** |
| Static (1000 epochs) | Random | 22% |
| Random (1000 epochs) | Random | 27% |

**Loss 曲線觀察：**

- **Static 模式**：Loss 在前 200 個 epoch 快速下降至接近 0，之後穩定。因為棋盤固定，agent 只需記住一條固定路徑。
- **Random 模式**：Loss 在整個訓練過程中劇烈震盪（100~400 之間），始終無法收斂。

### 4.2 Naive DQN 失敗的原因

**問題一：經驗相關性（Temporal Correlation）**

連續的訓練資料 (s₁→s₂→s₃→...) 來自同一局遊戲，具有高度的時間相關性。這違反了隨機梯度下降（SGD）對**獨立同分布（i.i.d.）**資料的基本假設，導致：
- 網路過度擬合到最近的經驗軌跡
- 梯度方向不穩定，Loss 劇烈震盪

**問題二：災難性遺忘（Catastrophic Forgetting）**

每次更新只使用最新一筆經驗，之前學到的知識會被新經驗覆蓋。在 Random 模式下，每局棋盤配置不同，agent 需要在各種情境下都能做出正確決策，但單筆更新無法同時兼顧所有情境。

### 4.3 實驗二：Experience Replay Buffer DQN

Experience Replay 的核心機制：
1. 將每步經驗 `(s, a, r, s', done)` 存入固定大小的 buffer（容量 1000）
2. 訓練時從 buffer 中**隨機取樣一個 mini-batch**（大小 200）
3. 用這個 batch 來更新網路

| 訓練模式 | 測試模式 | 勝率 |
|---------|---------|:----:|
| Static (1000 epochs) | Static | **100%** |
| Player (1000 epochs) | Player | **100%** |
| Random (1000 epochs) | Random | **71%** |

### 4.4 Experience Replay 為何有效？

| 比較項目 | Naive DQN | Experience Replay DQN |
|---------|-----------|----------------------|
| 訓練資料 | 最新 1 筆 | 隨機取樣 200 筆 |
| 經驗相關性 | 高（連續步驟） | 低（隨機取樣打破相關性） |
| 資料利用率 | 每筆只用 1 次 | 每筆可被多次取樣使用 |
| 梯度穩定性 | 差（單筆雜訊大） | 好（batch 平均降低變異）|
| Random 模式勝率 | 27% | **71%** |

**隨機取樣**打破了時間相關性，使訓練資料近似 i.i.d.，滿足 SGD 的理論假設。
**重複利用**過去的經驗，使 agent 能同時學習各種棋盤配置下的策略，避免災難性遺忘。

### 4.5 Loss 曲線比較

對比兩種方法在 Random 模式下的 Loss 曲線：
- **Naive DQN**：Loss 在 100~400 之間劇烈震盪，無收斂趨勢
- **Replay DQN**：Loss 在前 200 epoch 後穩步下降，最終穩定在 0.2~1.0 附近

這直觀地反映了 Experience Replay 對訓練穩定性的顯著改善。

---

## 五、程式碼中值得注意的細節

### 5.1 狀態雜訊

```python
state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
```

在狀態向量中加入微小的隨機雜訊（0~0.1），目的是：
- 防止網路對完全一樣的輸入產生過擬合
- 增加訓練的正規化效果
- 幫助 agent 在相似但不完全相同的狀態間泛化

### 5.2 終止條件判斷

```python
if abs(reward) == 10:
    status = 0  # 遊戲結束
```

只有當獲得 +10（到達 Goal）或 -10（掉入 Pit）時遊戲才結束。每走一步的 -1 獎勵不會終止遊戲，這驅使 agent 在有限步數內找到最短路徑。

---

## 六、結論

1. **Naive DQN** 在固定環境（Static）下表現良好，但無法應對隨機變化的環境，根本原因是經驗相關性和災難性遺忘。
2. **Experience Replay Buffer** 透過隨機取樣和重複利用歷史經驗，有效解決了上述兩個問題，將 Random 模式的勝率從 27% 提升至 71%。
3. Experience Replay 是 DQN 從理論走向實用的關鍵改進之一，也是後續 Double DQN、Dueling DQN 等進階方法的重要基礎。

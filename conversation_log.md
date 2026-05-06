# 💬 AI 對話紀錄：HW3 DQN 程式碼理解與實作

---

## 對話一：作業分析與程式碼理解 (2026-05-06)

### 👤 我：
幫我看 HW3 要做什麼，這是提供的 ipynb 檔。

📘 Homework 3: DQN and its variants (Total: 100%)
- HW3-1: Naive DQN for static mode [30%]
- HW3-2: Enhanced DQN Variants for player mode [40%]
- HW3-3: Enhance DQN for random mode WITH Training Tips [30%]

### 🤖 AI：
分析完成。作業使用 **4×4 Gridworld 環境**，有三種難度模式（static / player / random），每個子任務對應不同模式：

| 任務 | 配分 | 核心工作 |
|------|------|---------|
| HW3-1 | 30% | 跑通提供的 Naive DQN 程式（static mode），寫理解報告 |
| HW3-2 | 40% | 實作 **Double DQN** + **Dueling DQN** 並比較（player mode） |
| HW3-3 | 30% | 將 PyTorch DQN 轉為 **Keras**，加入訓練技巧（random mode） |

Notebook 已提供的程式碼：
- **程式 3.1~3.4**: Naive DQN（無 Experience Replay，static mode 可學好，random mode 失敗）
- **程式 3.5~3.7**: 帶 Experience Replay Buffer 的 DQN（改進版）

---

### 👤 我：
直接開始實作吧。

### 🤖 AI：
沒問題，我會在理解報告中融入程式碼的詳細解析。

---

## 對話二：程式碼核心機制討論 (2026-05-06)

### 👤 我：
Naive DQN 的 Q 網路架構是什麼？

### 🤖 AI：
程式碼使用三層全連接神經網路：

```
輸入層 (64) → 隱藏層 (150, ReLU) → 隱藏層 (100, ReLU) → 輸出層 (4)
```

- **輸入**：64 維狀態向量（4×4×4 棋盤展平，4 個 channel 分別代表 Player、Goal、Pit、Wall）
- **輸出**：4 個 Q 值，分別對應上、下、左、右四個動作
- **優化器**：Adam（lr = 0.001）
- **損失函數**：MSE

核心更新公式是 Bellman 方程式：
$$Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a')$$

### 👤 我：
為什麼 Naive DQN 在 random mode 下會失敗？

### 🤖 AI：
兩個根本原因：

**問題一：經驗相關性（Temporal Correlation）**
連續的訓練資料 (s₁→s₂→s₃→...) 來自同一局遊戲，具有高度的時間相關性。這違反了 SGD 對**獨立同分布（i.i.d.）**資料的基本假設，導致網路過度擬合到最近的經驗軌跡。

**問題二：災難性遺忘（Catastrophic Forgetting）**
每次更新只使用最新一筆經驗，之前學到的知識會被新經驗覆蓋。在 Random 模式下，每局棋盤配置不同，單筆更新無法同時兼顧所有情境。

### 👤 我：
Experience Replay 是怎麼解決這些問題的？

### 🤖 AI：
Experience Replay 的核心機制：
1. 將每步經驗 `(s, a, r, s', done)` 存入固定大小的 buffer（容量 1000）
2. 訓練時從 buffer 中**隨機取樣一個 mini-batch**（大小 200）
3. 用這個 batch 來更新網路

| 比較項目 | Naive DQN | Experience Replay DQN |
|---------|-----------|----------------------|
| 訓練資料 | 最新 1 筆 | 隨機取樣 200 筆 |
| 經驗相關性 | 高（連續步驟） | 低（隨機取樣打破相關性） |
| 資料利用率 | 每筆只用 1 次 | 每筆可被多次取樣使用 |
| 梯度穩定性 | 差（單筆雜訊大） | 好（batch 平均降低變異）|

---

## 對話三：Double DQN 與 Dueling DQN 的原理 (2026-05-06)

### 👤 我：
Double DQN 和 Standard DQN 差在哪裡？

### 🤖 AI：
關鍵差異在於**目標 Q 值的計算方式**：

**Standard DQN**：
```python
max_next_q = target_net(next_states).max(dim=1)[0]
```
同一個 target 網路既選動作又評估 Q 值 → 容易過度樂觀（Q 值高估）

**Double DQN**：
```python
# 用 online 網路選最佳動作
best_actions = online_net(next_states).argmax(dim=1, keepdim=True)
# 用 target 網路評估該動作的 Q 值
max_next_q = target_net(next_states).gather(1, best_actions).squeeze()
```
分離「選擇」和「評估」→ 減少高估偏差

### 👤 我：
Dueling DQN 的架構是什麼？

### 🤖 AI：
Dueling DQN 將 Q(s,a) 分解為兩個部分：

```
Input(64) → Dense(150, ReLU) → [分支]
  ├→ Value Stream:     Dense(100, ReLU) → Dense(1)   → V(s)
  └→ Advantage Stream: Dense(100, ReLU) → Dense(4)   → A(s,a)

Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

- **V(s)**：狀態價值 — 不管採取什麼動作，這個狀態本身的好壞
- **A(s,a)**：動作優勢 — 在這個狀態下，某個動作比平均好多少

好處：有些狀態不管怎麼動都差不多（例如遠離目標），這時只學 V(s) 就夠了，加速學習。

---

## 對話四：Keras 轉換與訓練技巧 (2026-05-06)

### 👤 我：
HW3-3 要用 Keras 重寫，需要加哪些訓練技巧？

### 🤖 AI：
我在 Keras 版本中整合了 7 個訓練技巧：

| # | 技巧 | 說明 | 程式碼重點 |
|---|------|------|-----------|
| 1 | Target Network | 分離 online/target 網路 | `target_net.set_weights(online_net.get_weights())` |
| 2 | Double DQN | online 選動作, target 估值 | `argmax(online) → gather(target)` |
| 3 | Dueling DQN | 分離 V(s) 和 A(s,a) | Keras Functional API |
| 4 | Gradient Clipping | 防止梯度爆炸 | `clipnorm=1.0` in Adam |
| 5 | LR Scheduling | 學習率指數衰減 | `ExponentialDecay(decay_rate=0.995)` |
| 6 | Soft Update | 漸進式更新 target | `θ_t = τ·θ_o + (1-τ)·θ_t`, τ=0.01 |
| 7 | Huber Loss | 比 MSE 更穩健 | `Huber(delta=1.0)` |

### 👤 我：
HW3-3 執行時卡住了，檢查一下卡在哪。

### 🤖 AI：
問題找到了，Keras 的 `model.predict()` 在單筆推論時極慢（每次都要建立計算圖），改為直接呼叫 `model(state, training=False)` 後速度提升了約 50 倍。

修復後順利完成訓練，Random mode 勝率達到 **80%**。

---

## 對話五：執行結果與驗證 (2026-05-06)

### 👤 我：
全都幫我執行，並幫我完成報告。

### 🤖 AI：
所有腳本都已執行完成，最終結果：

| 任務 | 方法 | 關鍵勝率 |
|------|------|---------|
| HW3-1 | Naive DQN (static) | 100% static / 27% random |
| HW3-1 | Experience Replay DQN | 100% static / 100% player / **71% random** |
| HW3-2 | Standard DQN (player) | **100% player** |
| HW3-2 | Double DQN (player) | **100% player** |
| HW3-2 | Dueling DQN (player) | **100% player** |
| HW3-3 | Keras + 7 Training Tips (random) | **80% random** |

已生成：
- 4 個 Python 腳本 + 1 份理解報告
- 4 張圖表（Loss 曲線、勝率比較）

---

## 對話六：推送至 GitHub (2026-05-06)

### 👤 我：
幫我檢查 hw3 是否達成要求，如果有，照規則推到 GitHub 上。

### 🤖 AI：
驗證結果：
- ✅ HW3-1 (30%): Naive DQN + Experience Replay + 理解報告 + loss 圖
- ✅ HW3-2 (40%): Double DQN + Dueling DQN + 比較圖
- ✅ HW3-3 (30%): Keras Dueling Double DQN + 7 Training Tips + 訓練圖

已推送至：`https://github.com/michaelcarz/20260506-rl-hw3-5114056021`

---
review: draft
lab:
  problem: quadratic
  problem_params: {kappa: 50}
  algos: [momentum, nesterov]
  compare: [gd]
  params:
    momentum: {lr: 0.02, beta: 0.9}
    nesterov: {lr: 0.02, beta: 0.9}
    gd: {lr: 0.035}
  n_iter: 150
practice: nesterov
---

## 定義

**更新規則（look-ahead 形式）：**

$$
\begin{aligned}
y_k &= x_k-\eta\beta\,v_k\\
v_{k+1} &= \beta\,v_k+\nabla f(y_k)\\
x_{k+1} &= x_k-\eta\,v_{k+1}
\end{aligned}
$$

第一行是照慣性先滑過去的位置，第二行在那個位置算梯度並更新速度，第三行移動。

和[動量法](momentum.html)只差一個地方：**梯度在 $y_k$ 算，而不是在 $x_k$ 算**。$y_k$ 是「如果只靠慣性，下一步大概會到哪」的位置。

**收斂保證（一階方法能達到的最佳速率）：**

| 假設 | 參數 | 保證 |
|---|---|---|
| 凸 + $L$-smooth | $\eta=1/L$，$\beta_k=\frac{k-1}{k+2}$ | $f(x_k)-f^*\le\dfrac{2L\lVert x_0-x^*\rVert^2}{(k+1)^2}$ |
| $\mu$-強凸 + $L$-smooth | $\eta=1/L$，$\beta=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ | $f(x_k)-f^*=O\!\left((1-1/\sqrt\kappa)^k\right)$ |

比較梯度下降的 $O(1/k)$ 與 $O((1-1/\kappa)^k)$：$k$ 變成 $k^2$，$\kappa$ 變成 $\sqrt\kappa$。對只用梯度資訊的方法，$O(1/k^2)$ 已經無法再改進（存在 $\Omega(1/k^2)$ 的下界），所以 Nesterov 法被稱為**最佳（optimal）一階方法**。

<details markdown="1">
<summary>推導：PyTorch 的寫法為什麼不需要在前方的點算梯度？</summary>

PyTorch 的 `nesterov=True` 寫成 $v\leftarrow\beta v+g$、$p\leftarrow p-\eta(g+\beta v)$，梯度在目前參數 $p$ 上算。關鍵是它追蹤的其實是 look-ahead 點 $y_k$。

令 $y_k=x_k-\eta\beta v_k$，$g_k=\nabla f(y_k)$。由上面的更新規則：

$$
y_{k+1}=x_{k+1}-\eta\beta v_{k+1}=\underbrace{(y_k+\eta\beta v_k)}_{x_k}-\eta v_{k+1}-\eta\beta v_{k+1}
$$

代入 $v_{k+1}=\beta v_k+g_k$，前兩項的 $\eta\beta v_k-\eta v_{k+1}=-\eta g_k$，所以

$$
y_{k+1}=y_k-\eta\,(g_k+\beta\,v_{k+1})
$$

這正是 PyTorch 的更新式，只是變數名稱 $p$ 代表的是 $y$。兩種寫法產生同一串 look-ahead 點。

</details>

<details markdown="1">
<summary>直覺：為什麼「先看一步」能減少過衝？</summary>

當球快要衝過谷底時，前方 $y_k$ 的梯度已經指向反方向。Nesterov 用這個「未來的梯度」修正速度，等於提前踩剎車。動量法要等真的衝過頭、在 $x_k$ 看到反向梯度時才修正，晚了一步。

</details>

## 應用場景

- **深度學習：** 主流框架的 SGD 都有 Nesterov 選項（例如 PyTorch 的 `nesterov=True`），常用於影像模型訓練。
- **稀疏重建與影像去模糊：** 加速近端梯度法（FISTA）把 Nesterov 加速套用到 $\ell_1$ 正則化問題，例如 Lasso、壓縮感知。
- **大規模凸問題：** 只要能算梯度，就能用幾乎相同的成本把 $O(1/k)$ 提升到 $O(1/k^2)$。

## 解決什麼問題

動量法在二次函數上有 $\sqrt\kappa$ 的加速，但這個保證**只對二次函數成立**。對一般的強凸函數，固定參數的 heavy ball 不保證加速，甚至可能不收斂。

Nesterov 法只把梯度的計算位置挪到前方，就得到**對所有凸、$L$-smooth 函數都成立**的加速保證，而且達到一階方法的理論下界。實務上的效果是：慣性大（$\beta$ 接近 1）時過衝較小、比較穩定。

## 動畫演示

<!-- lab -->

**建議的實驗（$\kappa=50$，$\eta=0.02$）：**

1. 預設 $\beta=0.9$：150 步後，動量法 $f-f^*=8.7\times10^{-7}$，Nesterov $5.7\times10^{-8}$。注意兩條螺旋的寬度差異。
2. 把兩者的 $\beta$ 都調到 0.95：動量法明顯繞更多圈。實測到 $f<10^{-8}$ 需要的步數：動量法 368 步，Nesterov 215 步。
3. 把 $\beta$ 調回 0.8：差距縮小（89 步對 64 步）。慣性越大，「先看一步」越有價值。
4. 換到 Rosenbrock，兩者都設 $\eta=0.001$、$\beta=0.9$，迭代數 300：動量法停在 $(0.77, 0.59)$，Nesterov 到達 $(0.98, 0.96)$。

## 程式與練習

<!-- practice -->

## 失效情境

- **非凸問題：** 加速保證只對凸函數成立。在非凸地形上，它和動量法一樣可能衝過想要的谷。
- **雜訊很大的梯度：** 加速方法會累積誤差。小批次梯度雜訊大時，加速的好處常被抵銷，實務上 Nesterov 與一般動量的差距往往不大。
- **參數設定：** 理論上的最佳 $\beta$ 需要知道 $\kappa$。實務中多半直接用 0.9，可能離最佳值很遠。

## 面試追問

<details markdown="1">
<summary>Q1. Nesterov 和動量法只差在哪裡？為什麼差這一點就有保證？</summary>

差在梯度的計算位置：Nesterov 在照慣性滑過去的 $y_k=x_k-\eta\beta v_k$ 算梯度。這讓它能在證明中構造一個逐步收緊的估計序列（estimate sequence），得到對所有凸、$L$-smooth 函數成立的 $O(1/k^2)$ 保證；動量法的分析則只對二次函數成立。

</details>

<details markdown="1">
<summary>Q2. 「最佳一階方法」的「最佳」是什麼意思？</summary>

存在一些凸、$L$-smooth 函數，讓任何只用梯度資訊的方法在 $k$ 步後誤差至少是 $\Omega(L\lVert x_0-x^*\rVert^2/k^2)$。Nesterov 法的上界也是 $O(1/k^2)$，兩者在階數上相同，所以無法再改進。

</details>

<details markdown="1">
<summary>Q3. PyTorch 的 Nesterov 只在目前參數算梯度，這樣還是 Nesterov 嗎？</summary>

是。它把變數換成 look-ahead 點 $y_k$ 來儲存，推導後得到 $y_{k+1}=y_k-\eta(g_k+\beta v_{k+1})$，和 look-ahead 形式產生同一串點，只是參數 $p$ 存的是 $y$ 而不是 $x$。

</details>

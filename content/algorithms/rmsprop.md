---
review: draft
lab:
  problem: quadratic
  problem_params: {kappa: 10}
  algos: [adagrad, rmsprop]
  compare: [adam, gd]
  params:
    adagrad: {lr: 0.5}
    rmsprop: {lr: 0.05, rho: 0.99}
  n_iter: 150
practice: rmsprop
---

## 定義

**更新規則（逐座標）：**

$$
s_{k}=\rho\,s_{k-1}+(1-\rho)\,g_k^2,\qquad
x_{k+1}=x_k-\frac{\eta}{\sqrt{s_k}+\varepsilon}\,g_k
$$

和 [AdaGrad](adagrad.html) 只差一行：$s$ 從「累加」改成**指數移動平均**（EMA），$\rho\in[0,1)$ 是衰減係數。PyTorch 預設 $\rho=0.99$，Keras 預設 $\rho=0.9$。

<details markdown="1">
<summary>推導：ρ 決定「記憶長度」</summary>

把遞迴展開（$s_0=0$）：

$$
s_k=(1-\rho)\sum_{j=1}^{k}\rho^{\,k-j}\,g_j^2
$$

權重 $\rho^{k-j}$ 以等比數列衰減，有效窗口約 $1/(1-\rho)$ 步：$\rho=0.9$ 約記住 10 步，$\rho=0.99$ 約記住 100 步。超過窗口的舊梯度幾乎被遺忘，所以學習率**不會像 AdaGrad 一樣單調縮到零**。

</details>

<details markdown="1">
<summary>推導：初始偏差，第一步會走得太遠</summary>

$s_0=0$ 時，$s_1=(1-\rho)g_1^2$，第一步的位移是

$$
\frac{\eta\,g_1}{\sqrt{(1-\rho)}\,|g_1|}=\frac{\eta}{\sqrt{1-\rho}}\,\mathrm{sign}(g_1)
$$

$\rho=0.99$ 時是 $10\eta$。原因是 $s$ 從 0 起算，前幾步嚴重低估梯度平方的平均。[Adam](adam.html) 用偏差修正 $s/(1-\rho^k)$ 解決這個問題。

</details>

<details markdown="1">
<summary>推導：靠近最小值時為什麼會「彈來彈去」</summary>

若梯度大小在窗口內大致不變，$s\approx g^2$，步長就接近 $\eta\,\mathrm{sign}(g)$，**不會因為靠近最小值而變小**。梯度很小時，只要 $s$ 也跟著變小，相除後仍是 $O(\eta)$ 的步長。

窗口越短（$\rho$ 越小），$s$ 跟得越緊，這個現象越明顯。窗口長時，$s$ 還記得先前較大的梯度，比值 $g/\sqrt s$ 會變小，步長才會跟著縮小。

</details>

## 應用場景

- **循環神經網路（RNN）：** 梯度大小在訓練中變化劇烈，逐座標正規化能穩定訓練。
- **深度強化學習：** 目標會隨策略變動（非平穩），需要「只記得最近」的尺度估計。
- **作為 Adam 的一半：** Adam 的二階動差就是 RMSProp 的 $s$。

## 解決什麼問題

AdaGrad 的 $s$ 只增不減，學習率最後會降到零，長時間訓練會停滯。對**非平穩**的問題更糟：地形改變後，舊梯度的影響永遠留著。

RMSProp 用指數移動平均取代累加，只記住最近約 $1/(1-\rho)$ 步，於是：

- 學習率不會單調衰減到零。
- 地形改變時，尺度估計能跟著更新。

代價是它不再有 AdaGrad 那種自動衰減。用固定學習率時，靠近最小值會在大約 $\eta$ 的尺度上來回彈動，通常需要搭配學習率排程。

## 動畫演示

<!-- lab -->

**建議的實驗（$\kappa=10$）：**

1. 預設下比較 AdaGrad（$\eta=0.5$）和 RMSProp（$\eta=0.05$、$\rho=0.99$）。RMSProp 的第一步很大：$s$ 從 0 起算，第一步走了 $10\eta=0.5$。
2. 把 $\rho$ 調到 0.9：150 步中，$f-f^*$ 最低曾到 $1.5\times10^{-8}$，最後卻是 $6.8\times10^{-3}$。把滑鼠移到收斂曲線上，可以看到它先降下去又彈起來。
3. 把 $\rho$ 調回 0.99：150 步後 $f-f^*=1.9\times10^{-7}$，而且一路下降。記憶較長，靠近最小值時步長才會縮小。
4. 打開 Adam 比較：Adam 加上了動量和偏差修正，第一步不會跳太遠。

## Scratch practice

<!-- practice -->

## 失效情境

- **固定學習率下不收斂到精確解：** 靠近最小值時步長仍是 $O(\eta)$，會在最小值附近振盪（實驗 2）。
- **初始幾步過大：** 沒有偏差修正，$\rho$ 越接近 1，第一步越大（實驗 1）。
- **座標系不對齊：** 和 AdaGrad 一樣只做對角縮放，旋轉過的峽谷無法處理。

## 面試追問

<details markdown="1">
<summary>Q1. RMSProp 和 AdaGrad 差在哪裡？為什麼這一點很重要？</summary>

AdaGrad 累加所有梯度平方，學習率單調衰減到零；RMSProp 用指數移動平均，只記得最近約 $1/(1-\rho)$ 步。這讓長時間訓練不會停滯，也能適應非平穩的目標。

</details>

<details markdown="1">
<summary>Q2. ρ = 0.99 時，第一步會走多遠？</summary>

$s_1=(1-\rho)g_1^2=0.01\,g_1^2$，所以位移是 $\eta/\sqrt{0.01}=10\eta$。這是 $s$ 從 0 起算造成的初始偏差，Adam 用除以 $1-\rho^k$ 修正。

</details>

<details markdown="1">
<summary>Q3. 為什麼 RMSProp 搭配固定學習率很難收斂到很精確？</summary>

步長大約是 $\eta\cdot g/\sqrt{s}$，而靠近最小值時 $g$ 和 $\sqrt s$ 一起變小，比值仍是 $O(1)$，所以步長停在 $O(\eta)$，會在最小值附近振盪。需要學習率衰減才能收斂到更精確的解。

</details>

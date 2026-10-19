---
review: draft
lab:
  problem: quadratic
  problem_params: {kappa: 10}
  algos: [gd, sgd]
  compare: [momentum, adam]
  params:
    gd: {lr: 0.15}
    sgd: {lr: 0.15, sigma: 1.0, decay: 0.0}
  n_iter: 150
practice: sgd
predict:
  - id: p1
    prompt: "把雜訊 σ 調到 0，SGD（η = 0.15）150 步內 f − f* 能降到 10⁻¹⁰ 以下嗎？"
    scenario: {params: {sgd: {sigma: 0.0}}, n_iter: 150}
    subject: sgd
    show: [gd, sgd]
    options:
      - {label: "能", when: {reach_within: 150}}
      - {label: "不能", when: {not_reach_within: 150}}
    explain: "沒有雜訊時，SGD 每步用的就是真梯度，更新式和 GD 一模一樣，兩條軌跡完全重合（都在第 78 步達到 10⁻¹⁰）。SGD 與 GD 唯一的差別就是梯度裡的雜訊。"
  - id: p2
    prompt: "雜訊只剩 σ = 0.1，η = 0.15 固定，跑 500 步。f − f* 能降到 10⁻¹⁰ 以下嗎？"
    scenario: {params: {sgd: {sigma: 0.1}}, n_iter: 500}
    subject: sgd
    options:
      - {label: "能，雜訊這麼小，多跑幾步就行", when: {reach_within: 500}}
      - {label: "不能，會一直在最小值附近亂跳", when: {not_reach_within: 500}}
    explain: "步長固定時，每步都注入 η × 雜訊，迭代點最後停在最小值附近的「雜訊球」裡，球的大小不隨步數縮小（小 η 時 f − f* 約與 ησ² 成正比）。同一組亂數下，500 步後 f − f* = 2.6 × 10⁻³，σ = 1 時是 0.26：σ 縮小 10 倍，f − f* 剛好縮小 100 倍，但永遠不會是零。"
  - id: p3
    prompt: "σ = 1，改用衰減步長 ηₖ = η₀/(1 + ck)，c = 0.05，跑 500 步。結果會？"
    scenario: {params: {sgd: {decay: 0.05}}, n_iter: 500}
    subject: sgd
    options:
      - {label: "500 步內 f − f* 降到 10⁻¹⁰ 以下", when: {reach_within: 500}}
      - {label: "到不了 10⁻¹⁰，但最後 f − f* < 10⁻²", when: {not_reach_within: 500, final_gap_below: 1.0e-2}}
      - {label: "和固定步長差不多：最後 f − f* ≥ 10⁻²", when: {not_reach_within: 500, final_gap_above: 1.0e-2}}
    explain: "步長遞減讓雜訊球跟著縮小，500 步後 f − f* 降到 1.6 × 10⁻³（固定步長時約 0.26）。但步長按 1/k 變小，誤差也只能以大約 1/k 的次線性速率下降，遠到不了 10⁻¹⁰：精確收斂和速度不能兼得，這是 SGD 的根本取捨。"
---

## 定義

**問題形式（有限和）：** 機器學習與資料擬合的目標函數通常是很多項的平均，

$$
f(x)=\frac1N\sum_{i=1}^{N} f_i(x)
$$

$f_i$ 是第 $i$ 筆資料的誤差。精確梯度要掃過全部 $N$ 筆資料，$N$ 很大時非常昂貴。

**更新規則：** 每步隨機抽一小批 $B_k$（大小 $b$），用它的平均梯度代替真正的梯度：

$$
\tilde g_k=\frac1b\sum_{i\in B_k}\nabla f_i(x_k),\qquad x_{k+1}=x_k-\eta_k\,\tilde g_k
$$

$\tilde g_k$ 是**不偏估計**：$\mathbb E[\tilde g_k]=\nabla f(x_k)$，變異數大約與 $1/b$ 成正比。

**本站的模擬方式：** 動畫中用 $\tilde g_k=\nabla f(x_k)+\sigma\,\xi_k$、$\xi_k\sim\mathcal N(0,I)$ 代表小批次的估計誤差，並固定亂數種子讓結果可重現。學習率排程為 $\eta_k=\eta_0/(1+c\,k)$。

<details markdown="1">
<summary>推導：固定學習率只會收斂到一個「雜訊球」</summary>

取一維 $f(x)=\tfrac{\lambda}{2}x^2$，更新變成

$$
x_{k+1}=(1-\eta\lambda)\,x_k-\eta\sigma\,\xi_k
$$

這是一個 AR(1) 過程。穩態變異數滿足 $V=(1-\eta\lambda)^2V+\eta^2\sigma^2$，解得

$$
V=\frac{\eta\,\sigma^2}{\lambda\,(2-\eta\lambda)},\qquad
\mathbb E[f]=\frac{\lambda}{2}V=\frac{\eta\,\sigma^2}{2\,(2-\eta\lambda)}
$$

所以 $\mathbb E[f]-f^*$ **不會趨近 0，而是停在與 $\eta$ 大約成正比的水準**。對本站的二次碗（$\lambda=1$ 和 $10$，$\eta=0.15$、$\sigma=1$），兩個方向相加的理論值是 $0.040+0.150=0.19$；實測 400 步、取最後 200 步平均為 $0.174$。

</details>

<details markdown="1">
<summary>收斂條件：學習率要怎麼衰減？</summary>

經典的 Robbins–Monro 條件：

$$
\sum_k \eta_k=\infty,\qquad \sum_k \eta_k^2<\infty
$$

第一個條件保證走得夠遠，能到達任何地方；第二個保證雜訊的總影響有限。$\eta_k=\eta_0/(1+ck)$ 兩者都滿足。對強凸問題，適當的 $\eta_k\propto1/k$ 可得 $\mathbb E[f(x_k)]-f^*=O(1/k)$。

</details>

## 應用場景

- **深度學習：** 幾乎所有神經網路都以小批次 SGD（或其變形）訓練。資料量可達數十億筆，不可能每步算完整梯度。
- **線上學習：** 資料一筆一筆到達，例如推薦系統即時更新模型。
- **大量觀測的參數校準：** 觀測資料很多時（例如長期的多站水位或流量紀錄），每次抽一部分觀測計算誤差梯度，可大幅降低每步成本。

## 解決什麼問題

梯度下降每步都要掃過全部資料。$N=10^9$ 時，一次梯度可能要算好幾個小時，但一開始離解很遠，根本不需要那麼精確的方向。

SGD 用**便宜但有雜訊**的梯度估計換取極低的每步成本：同樣的計算量可以走成千上萬步。雜訊本身在非凸問題中還可能有好處，例如幫助離開鞍點和尖銳的局部最小值。

代價是**固定學習率無法精確收斂**，只會在最小值附近一個大小與 $\eta\sigma^2$ 相關的區域內打轉。要精確收斂，就必須讓學習率衰減，或增大批次以降低 $\sigma$。

## 動畫演示

<!-- lab -->

**建議的實驗（$\kappa=10$，$\eta_0=0.15$，$\sigma=1$）：**

1. 預設衰減 $c=0$：SGD 很快進入谷底附近，然後在那裡亂跳。150 步中最後 50 步的平均 $f-f^*$ 是 $0.26$。
2. 把衰減 $c$ 調到 0.05：最後 50 步的平均降到 $6.5\times10^{-3}$，路徑越來越平穩。
3. 把 $\sigma$ 調到 0：SGD 的軌跡和 GD 完全重合。雜訊是唯一的差別。
4. 固定 $c=0$，把 $\eta$ 從 0.15 降到 0.05：雜訊球變小（實測 400 步的最後 200 步平均從 0.174 降到 0.039），但前期也變慢。這是 SGD 最核心的取捨。

## 程式與練習

<!-- practice -->

## 失效情境

- **學習率固定：** 只會收斂到雜訊球，無法得到精確解（實驗 1）。
- **學習率衰減太快：** $\sum\eta_k<\infty$ 時，可能在到達最小值前就停住。
- **病態問題：** SGD 繼承了 GD 對條件數的敏感，雜訊還會讓大曲率方向更不穩定。
- **梯度雜訊太大：** 批次太小或資料極度不平衡時，梯度方向幾乎是隨機的，需要更大批次或降低學習率。

## 面試追問

<details markdown="1">
<summary>Q1. 為什麼 SGD 用的是「有雜訊的梯度」還能收斂？</summary>

小批次梯度是真實梯度的不偏估計，平均來說方向正確。只要學習率滿足 $\sum\eta_k=\infty$、$\sum\eta_k^2<\infty$，雜訊的影響會被逐漸平均掉，同時又能走得夠遠。

</details>

<details markdown="1">
<summary>Q2. 固定學習率的 SGD 最後會停在哪裡？</summary>

停在最小值附近的一個區域，平均誤差大約與 $\eta\sigma^2$ 成正比。對一維二次函數可以精確算出 $\mathbb E[f]-f^*=\eta\sigma^2/(2(2-\eta\lambda))$。

</details>

<details markdown="1">
<summary>Q3. 批次大小加倍，對訓練有什麼影響？</summary>

梯度變異數大約減半，雜訊球變小，可以用較大的學習率；但每步計算量也加倍。在固定計算預算下，批次大到一定程度後，多出來的計算不再換得等比例的進步。

</details>

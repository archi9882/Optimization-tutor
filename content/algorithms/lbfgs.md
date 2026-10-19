---
review: draft
lab:
  problem: quadratic
  problem_params: {kappa: 10}
  algos: [bfgs, lbfgs]
  compare: [gd]
  params:
    lbfgs: {m: 5}
  n_iter: 60
practice: lbfgs
predict:
  - id: p1
    prompt: "二次碗（κ = 10），L-BFGS 只記 m = 2 組 (s, y)，何時達到 f − f* < 10⁻¹⁰？"
    scenario: {params: {lbfgs: {m: 2}}, n_iter: 60}
    subject: lbfgs
    show: [bfgs, lbfgs]
    options:
      - {label: "12 步內", when: {reach_within: 12}}
      - {label: "13～60 步", when: {reach_within: 60, not_reach_within: 12}}
      - {label: "60 步內到不了", when: {not_reach_within: 60}}
    explain: "2 維問題只有 2 個曲率方向，記住最近 2 組 (s, y) 就差不多夠了：第 10 步達到 10⁻¹⁰（m = 5 是第 8 步，完整 BFGS 第 4 步）。實務上 m 取 5～20，遠小於維度 n，記憶體只要 O(mn)。"
  - id: p2
    prompt: "把 m 降到 1（只記上一步），同一個二次碗會？"
    scenario: {params: {lbfgs: {m: 1}}, n_iter: 60}
    subject: lbfgs
    show: [bfgs, lbfgs]
    options:
      - {label: "12 步內", when: {reach_within: 12}}
      - {label: "13～60 步", when: {reach_within: 60, not_reach_within: 12}}
      - {label: "60 步內到不了", when: {not_reach_within: 60}}
    explain: "只記 1 組 (s, y) 時，只知道上一步那個方向的曲率，其他方向只能用 γI 頂替，近似的反 Hessian 一直不完整：第 23 步才達到 10⁻¹⁰。"
  - id: p3
    prompt: "換到 Rosenbrock，m = 1，跑 300 步，會？"
    scenario: {problem: rosenbrock, params: {lbfgs: {m: 1}}, n_iter: 300}
    subject: lbfgs
    show: [bfgs, lbfgs]
    options:
      - {label: "100 步內達到 10⁻¹⁰", when: {reach_within: 100}}
      - {label: "101～300 步達到", when: {reach_within: 300, not_reach_within: 100}}
      - {label: "300 步內到不了", when: {not_reach_within: 300}}
    explain: "Rosenbrock 的谷是彎的，曲率隨位置一直改變；m = 1 學到的曲率很快就過時，300 步後 f − f* 仍有 6.1 × 10⁻¹⁰。m = 3 時第 39 步就達到，和 BFGS 的第 36 步相近：記憶不用多，但不能太少。"
---

## 定義

**想法：** BFGS 的 $H$ 其實完全由初始矩陣和所有的 $(s_i,y_i)$ 決定。L-BFGS **只保留最近 $m$ 組** $(s_i,y_i)$，每次需要 $H\nabla f$ 時，直接用它們遞推出結果，從不存 $H$。

**兩迴圈遞推**（計算 $r=H_k\,g$）：

1. $q\leftarrow g$。由新到舊，對每組 $(s_i,y_i)$：$a_i=\rho_i s_i^\top q$，$q\leftarrow q-a_i y_i$。
2. $r\leftarrow\gamma\,q$，其中 $\gamma=s^\top y/y^\top y$（最新一組），作為初始 Hessian 的尺度估計。
3. 由舊到新：$b=\rho_i y_i^\top r$，$r\leftarrow r+(a_i-b)\,s_i$。

方向 $d=-r$，之後和 BFGS 一樣做線搜尋，並把新的 $(s,y)$ 放進記憶、丟掉最舊的。

**成本：** 記憶體 $O(mn)$，每步計算 $O(mn)$。$m$ 通常取 3～20。

<details markdown="1">
<summary>推導：為什麼兩個迴圈就能算出 H·g？</summary>

把 BFGS 更新寫成 $H_{k+1}=V_k^\top H_k V_k+\rho_k s_k s_k^\top$，其中 $V_k=I-\rho_k y_k s_k^\top$。反覆展開 $m$ 次：

$$
H_k=(V_{k-1}^\top\cdots V_{k-m}^\top)\,H_k^0\,(V_{k-m}\cdots V_{k-1})+\sum(\text{秩一修正項})
$$

把它乘上 $g$ 時，右側的 $V$ 連乘由新到舊作用在 $g$ 上（第一迴圈），中間乘上 $H_k^0=\gamma I$，左側的 $V^\top$ 連乘再由舊到新作用回來，同時補上秩一修正（第二迴圈）。全程只有向量內積和加減，沒有任何矩陣。

</details>

## 應用場景

- **大規模平滑最佳化：** 數百萬參數仍可使用，是科學計算中最常用的擬牛頓法。
- **PINN 訓練：** 常見流程是先用 Adam 快速降低損失，再切換到 L-BFGS 精細收斂。
- **地球物理反演：** 全波形反演等問題中，以伴隨法算梯度、L-BFGS 更新模型是常見組合。
- **機器學習中的凸問題：** 例如大規模邏輯斯迴歸。

## 解決什麼問題

BFGS 收斂快，但 $n\times n$ 的 $H$ 在 $n$ 很大時存不下。L-BFGS 只記住最近幾步的曲率資訊：

- 記憶體從 $O(n^2)$ 降到 $O(mn)$。
- 保留了大部分擬牛頓的加速效果。
- 只需要梯度，實作也簡單。

代價是只看得到最近 $m$ 步的曲率，$m$ 太小時效果會打折；收斂速度通常是線性的（但常數很好）。

## 動畫演示

<!-- lab -->

**建議的實驗（二次碗，$\kappa=10$）：**

1. 預設 $m=5$：L-BFGS 第 8 步達到 $f<10^{-10}$，BFGS 第 4 步。差別來自 L-BFGS 用 $\gamma I$ 當初始矩陣，而且記憶會被淘汰。
2. 把 $m$ 調到 1：要到第 23 步才達到 $f<10^{-10}$，曲率資訊太少。調到 2：第 10 步，在 2 維問題上已經接近 $m=5$ 的效果。
3. 換到 Rosenbrock：$m=5$ 第 46 步達到 $f<10^{-10}$；$m=1$ 時 300 步內仍到不了 $10^{-10}$。

## 程式與練習

<!-- practice -->

## 失效情境

- **m 太小：** 曲率資訊不足，接近一般的梯度法（實驗 2、3）。
- **雜訊梯度：** 小批次訓練時 $(s,y)$ 充滿雜訊，曲率估計失準，可能比 Adam 還差，所以 L-BFGS 多用於全批次或接近收斂的階段。
- **非平滑或不連續的目標函數：** 和 BFGS 一樣，割線條件會失準。
- **曲率條件不成立：** $s^\top y\le0$ 時必須跳過該組，否則方向可能不是下降方向。

## 面試追問

<details markdown="1">
<summary>Q1. L-BFGS 的記憶體和計算成本是多少？和 BFGS 比呢？</summary>

L-BFGS 存 $m$ 組長度 $n$ 的向量對，記憶體與每步計算都是 $O(mn)$；BFGS 要存 $n\times n$ 的 $H$，是 $O(n^2)$。$n$ 很大時差別是決定性的。

</details>

<details markdown="1">
<summary>Q2. 為什麼 PINN 常先用 Adam 再換 L-BFGS？</summary>

訓練初期損失地形崎嶇、梯度變化大，Adam 的逐座標縮放比較穩健；接近收斂時地形較平滑，L-BFGS 利用曲率資訊能把損失壓得更低。全批次的 PINN 訓練也正好符合 L-BFGS 需要準確梯度的條件。

</details>

<details markdown="1">
<summary>Q3. 兩迴圈遞推中的 γ 有什麼作用？</summary>

$\gamma=s^\top y/y^\top y$ 是沿最近一步方向的「平均曲率倒數」，用來縮放初始矩陣 $H^0=\gamma I$。它讓第一次嘗試的步長尺度合理，線搜尋通常直接接受 $\alpha=1$。

</details>

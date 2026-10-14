---
review: draft
lab:
  problem: quadratic
  problem_params: {kappa: 10}
  algos: [gd]
  compare: [momentum, nesterov, adam]
  n_iter: 150
practice: gd
---

## 定義

**問題：** 給定可微函數 $f:\mathbb{R}^n\to\mathbb{R}$，求 $\min_x f(x)$。

**更新規則：**

$$
x_{k+1} = x_k - \eta\,\nabla f(x_k)
$$

其中 $\eta>0$ 是學習率（步長）。每步只需要一次梯度計算，記憶體 $O(n)$。

**為什麼是負梯度？** 一階 Taylor 展開 $f(x+d)\approx f(x)+\nabla f(x)^\top d$。在固定步長 $\lVert d\rVert=\varepsilon$ 下，由 Cauchy–Schwarz 不等式，讓 $\nabla f^\top d$ 最小的方向正是 $d=-\varepsilon\,\nabla f/\lVert\nabla f\rVert$，也就是「最陡下降」方向。

**兩個關鍵假設：**

| 名稱 | 數學條件 | 直覺 |
|---|---|---|
| $L$-smooth | $\lVert\nabla f(x)-\nabla f(y)\rVert\le L\lVert x-y\rVert$ | 曲率最多是 $L$，地形不會突然變陡 |
| $\mu$-強凸 | $f(y)\ge f(x)+\nabla f(x)^\top(y-x)+\tfrac{\mu}{2}\lVert y-x\rVert^2$ | 曲率至少是 $\mu$，碗底不會太平 |

兩者的比值 $\kappa=L/\mu$ 稱為**條件數**：碗在最陡方向和最平方向的曲率比。

**收斂保證：**

| 假設 | 步長 | 保證 |
|---|---|---|
| $L$-smooth（可非凸） | $1/L$ | $\min_{k<K}\lVert\nabla f(x_k)\rVert^2\le \dfrac{2L\,(f(x_0)-f^*)}{K}$ |
| 凸 + $L$-smooth | $1/L$ | $f(x_K)-f^*\le \dfrac{L\lVert x_0-x^*\rVert^2}{2K}$ |
| $\mu$-強凸 + $L$-smooth | $2/(\mu+L)$ | $\lVert x_K-x^*\rVert\le\left(\dfrac{\kappa-1}{\kappa+1}\right)^K\lVert x_0-x^*\rVert$ |

<details markdown="1">
<summary>推導第 1 層：下降引理（為什麼步長要小於 2/L）</summary>

$L$-smooth 給出二次上界 $f(y)\le f(x)+\nabla f(x)^\top(y-x)+\tfrac{L}{2}\lVert y-x\rVert^2$。代入 $y=x-\eta\nabla f(x)$：

$$
f(x_{k+1})\le f(x_k)-\eta\left(1-\frac{L\eta}{2}\right)\lVert\nabla f(x_k)\rVert^2
$$

只要 $0<\eta<2/L$，括號為正，函數值就一定下降；$\eta=1/L$ 時保證的下降量 $\lVert\nabla f\rVert^2/(2L)$ 最大。把這個不等式從 $k=0$ 加到 $K-1$，就得到上表第一列。

</details>

<details markdown="1">
<summary>推導第 2 層：在二次函數上精確算出收斂率</summary>

取 $f(x)=\tfrac12 x^\top A x$，$A$ 對稱正定，特徵值介於 $\mu$ 與 $L$。更新規則變成線性遞迴：

$$
x_{k+1}=(I-\eta A)\,x_k
$$

沿 $A$ 的第 $i$ 個特徵向量，分量每步乘上 $(1-\eta\lambda_i)$。要所有分量都衰減，需要 $|1-\eta\lambda_i|<1$，也就是 $0<\eta<2/L$。最慢的分量決定收斂率：

$$
\rho(\eta)=\max\left(|1-\eta\mu|,\;|1-\eta L|\right)
$$

兩項相等時 $\rho$ 最小，解得 $\eta^*=2/(\mu+L)$、$\rho^*=(\kappa-1)/(\kappa+1)$。$\kappa$ 很大時 $\rho^*\approx1-2/\kappa$，要把誤差縮小 $e$ 倍大約需要 $\kappa/2$ 步：**迭代數與條件數成正比**。

</details>

<details markdown="1">
<summary>類比：GD 就是梯度流的顯式 Euler 法</summary>

梯度流 $\dot x=-\nabla f(x)$ 用前向 Euler 離散、時間步長取 $\eta$，就是梯度下降。二次函數上 $\dot x=-Ax$ 是剛性（stiff）線性系統，前向 Euler 的穩定條件 $\eta<2/\lambda_{\max}$ 正是上面的 $\eta<2/L$。這和顯式有限差分解擴散方程時受 $\Delta t\le \Delta x^2/(2D)$ 限制是同一件事：**最快的模態限制步長，最慢的模態決定要走多久**。

</details>

## 應用場景

- **機器學習：** 線性迴歸、邏輯斯迴歸的訓練；所有深度學習優化器（SGD、Momentum、Adam）都是它的變形。
- **地球科學反演：** 以伴隨法（adjoint）算出目標函數對地下參數的梯度，再用梯度類方法更新速度或滲透率模型。
- **水文模型校準：** 對可微的地下水或降雨逕流模型，以敏感度（梯度）資訊調整參數。
- **PINN 訓練：** 損失函數同時包含資料誤差與 PDE 殘差，用梯度法最小化。

## 解決什麼問題

最佳化的「理想答案」是直接解 $\nabla f(x)=0$，但大多數問題沒有解析解。牛頓法雖然收斂快，卻需要 $n\times n$ 的 Hessian 與 $O(n^3)$ 的線性求解，在百萬參數時不可行。

梯度下降只需要梯度、記憶體 $O(n)$，讓**高維問題變得可解**。代價是它只看局部斜率、不看曲率，所以在窄長峽谷（$\kappa$ 大）裡會鋸齒狀震盪、前進緩慢。這個缺點正是後續演算法的出發點：

- 用慣性抵銷震盪 → [動量法](momentum.html)、[Nesterov](nesterov.html)
- 每個座標各自調整步長 → [AdaGrad](adagrad.html)、[Adam](adam.html)
- 直接利用曲率 → [牛頓法](newton.html)、[L-BFGS](lbfgs.html)

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 在 $\kappa=10$ 的二次碗上，把 $\eta$ 從 0.05 調到 0.15，再調到 0.21：觀察從緩慢、最快，到發散的轉變。理論上限是 $2/L=0.2$。
2. 把 $\kappa$ 調到 50：看鋸齒有多嚴重，再打開 Momentum 比較。
3. 換成 Rosenbrock，試 $\eta=0.001$、$0.002$、$0.003$。實測（150 步）：0.001 時還困在左上方的谷壁附近，0.002 時已接近 $(1,1)$，0.003 就發散。**能用的步長區間非常窄**，這正是自適應方法要解決的問題。

## Scratch practice

<!-- practice -->

## 失效情境

- **步長太大（$\eta>2/L$）：** 在高曲率方向來回放大，直接發散。
- **病態問題（$\kappa$ 很大）：** 不會發散，但需要約 $\kappa$ 量級的步數，實務上等於停滯。
- **非凸地形：** 只保證走到梯度為零的點，可能是局部最小值或鞍點。
- **不可微函數：** 例如 $|x|$ 在 0 處，固定步長會在最小值兩側來回跳動。

## 面試追問

<details markdown="1">
<summary>Q1. 為什麼學習率不能超過 2/L？能用最簡單的例子說明嗎？</summary>

取一維 $f(x)=\tfrac{L}{2}x^2$，更新 $x_{k+1}=(1-\eta L)x_k$。當 $\eta>2/L$，$|1-\eta L|>1$，每步誤差都被放大並變號。高維時，最陡的方向（最大特徵值 $L$）最先出事。

</details>

<details markdown="1">
<summary>Q2. 條件數加倍，GD 需要的迭代數大約怎麼變？</summary>

強凸情況下迭代數約為 $O(\kappa\log(1/\varepsilon))$，所以大約也加倍。動量類加速方法可以降到 $O(\sqrt\kappa\log(1/\varepsilon))$。

</details>

<details markdown="1">
<summary>Q3. 梯度下降和數值解 ODE 有什麼關係？</summary>

GD 是梯度流 $\dot x=-\nabla f(x)$ 的前向 Euler 離散。步長上限 $2/L$ 就是顯式方法對剛性系統的穩定條件，條件數大則對應剛性強。

</details>

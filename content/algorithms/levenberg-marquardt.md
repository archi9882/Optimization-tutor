---
review: draft
lab:
  problem: himmelblau
  algos: [gauss_newton, lm]
  compare: [newton]
  params:
    gauss_newton: {alpha: 1.0}
    lm: {lam0: 1.0}
  n_iter: 60
practice: lm
---

## 定義

**更新規則：** 在高斯-牛頓的正規方程加上阻尼項 $\lambda I$：

$$
(J^\top J+\lambda I)\,d=-J^\top r
$$

並依結果調整 $\lambda$：

- 試走 $x+d$，若 $f$ 下降：**接受**，並把 $\lambda$ 縮小（例如除以 10）。
- 若 $f$ 沒有下降：**拒絕**（$x$ 不動），並把 $\lambda$ 放大（例如乘以 10）。

<details markdown="1">
<summary>推導：λ 在兩種方法之間連續切換</summary>

- $\lambda\to0$：方程回到 $(J^\top J)d=-J^\top r$，就是**高斯-牛頓**。
- $\lambda\to\infty$：$d\approx-J^\top r/\lambda=-\nabla f/(2\lambda)$，是步長很小的**梯度下降**。

所以 LM 在模型可信時大步走（像高斯-牛頓），在模型不可信時小步走（像梯度下降）。$\lambda>0$ 時 $J^\top J+\lambda I$ 一定正定，方向永遠存在，而且一定是下降方向。

</details>

<details markdown="1">
<summary>推導：LM 其實是信賴域法</summary>

考慮在半徑 $\Delta$ 內最小化線性化殘差：

$$
\min_d\ \lVert r+Jd\rVert^2\quad\text{s.t.}\quad\lVert d\rVert\le\Delta
$$

由 KKT 條件，最佳解滿足 $(J^\top J+\lambda I)d=-J^\top r$，其中 $\lambda\ge0$ 是半徑限制的 Lagrange 乘數。所以「調大 $\lambda$」等同「縮小信賴半徑」。這也解釋了為什麼 LM 的接受/拒絕規則和[信賴域法](trust-region.html)如此相似。

</details>

**常見變形：** Marquardt 建議改用 $\lambda\,\mathrm{diag}(J^\top J)$，讓阻尼隨各參數的尺度調整，對參數單位不敏感。

## 應用場景

- **參數反演與模型校準：** 地下水、地表水模式的參數校準軟體普遍採用 LM 類方法，因為它在病態、參數相關性高的問題上很穩健。
- **曲線擬合：** 科學計算套件中非線性最小平方擬合的預設方法之一。
- **電腦視覺與機器人：** 姿態估計、地圖建構（SLAM）中的非線性最小平方問題。

## 解決什麼問題

高斯-牛頓有兩個弱點：$J$ 不滿秩時方向不存在；步子太大時 $f$ 可能上升。

LM 用同一個參數 $\lambda$ 同時解決兩者：

- $\lambda I$ 讓矩陣永遠可逆，方向永遠存在。
- 接受/拒絕規則保證 $f$ 單調下降。
- 接近解時 $\lambda\to0$，回到高斯-牛頓的快速收斂。

代價是可能比高斯-牛頓慢：它拒絕任何讓 $f$ 上升的步子，即使那一步其實是捷徑。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 預設的 Himmelblau，起點 $(-0.5,-0.5)$：高斯-牛頓因為 Jacobian 奇異而完全不動；LM 第 8 步就達到 $f<10^{-10}$，收斂到 $(-2.81,3.13)$。途中有 2 步被拒絕，路徑上會看到原地停留的點。
2. 把 $\lambda_0$ 調到 0.001 或 1000：分別要 11 步和 8 步，LM 對初始 $\lambda$ 很不敏感，因為 $\lambda$ 會自動調整。
3. 換到 Rosenbrock：高斯-牛頓 2 步，LM 卻要 51 步。原因是高斯-牛頓的第 1 步經過 $f=3910$ 的點才跳到答案，LM 拒絕這種上升的步子，只能沿著谷慢慢前進。**穩健與快速之間有取捨。**

## 程式與練習

<!-- practice -->

## 失效情境

- **λ 調整太保守：** 每次只乘除 10，在需要大幅改變 λ 的地方會浪費很多步（實驗 3）。
- **局部最小值：** LM 仍是局部方法，起點決定收斂到哪個谷。
- **大殘差問題：** 接近解時退化成高斯-牛頓，若殘差大，收斂仍可能只有線性。
- **參數尺度差異大：** 用 $\lambda I$ 時，尺度大的參數被過度阻尼；改用 $\lambda\,\mathrm{diag}(J^\top J)$ 可以改善。

## 面試追問

<details markdown="1">
<summary>Q1. λ 很大和很小時，LM 分別像什麼方法？</summary>

λ 很小時接近高斯-牛頓，λ 很大時接近步長 $1/(2\lambda)$ 的梯度下降。LM 依「這一步有沒有讓 $f$ 下降」自動在兩者之間切換。

</details>

<details markdown="1">
<summary>Q2. 為什麼說 LM 是信賴域法？</summary>

在 $\lVert d\rVert\le\Delta$ 限制下最小化線性化殘差，KKT 條件給出 $(J^\top J+\lambda I)d=-J^\top r$，λ 是限制的 Lagrange 乘數。調整 λ 等同調整信賴半徑，λ 越大半徑越小。

</details>

<details markdown="1">
<summary>Q3. LM 一定比高斯-牛頓好嗎？</summary>

不一定。LM 更穩健（方向永遠存在、保證下降），但會拒絕暫時讓 $f$ 上升的捷徑，在 Rosenbrock 上就比高斯-牛頓慢得多。實務上穩健通常比快重要，所以參數校準多用 LM。

</details>

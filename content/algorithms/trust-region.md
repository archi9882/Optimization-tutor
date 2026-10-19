---
review: draft
lab:
  problem: rosenbrock
  algos: [newton, trust_region]
  compare: [bfgs]
  params:
    newton: {alpha: 1.0}
    trust_region: {radius0: 0.5}
  n_iter: 60
practice: trust_region
predict:
  - id: p1
    prompt: "Δ₀ 只有 0.05（很小的信賴範圍），何時達到 f − f* < 10⁻¹⁰？"
    scenario: {params: {trust_region: {radius0: 0.05}}, n_iter: 60}
    subject: trust_region
    options:
      - {label: "40 步內", when: {reach_within: 40}}
      - {label: "41～60 步", when: {reach_within: 60, not_reach_within: 40}}
      - {label: "60 步內到不了", when: {not_reach_within: 60}}
    explain: "每一步比較實際下降 ÷ 模型預測下降：比值 ≤ 0.1 就拒絕這一步並縮小 Δ；模型預測得準（比值 > 0.75）而且步子碰到邊界時，Δ 就加倍。半徑會自我調整，Δ₀ = 0.05 只讓頭幾步短一點，第 28 步就達到，和預設 Δ₀ = 0.5 的第 27 步差不多。"
  - id: p2
    prompt: "Δ₀ 調到 3（一開始幾乎不限制牛頓步）。會像牛頓法一樣 10 步內達到 f − f* < 10⁻¹⁰ 嗎？"
    scenario: {params: {trust_region: {radius0: 3.0}}, n_iter: 60}
    subject: trust_region
    show: [newton, trust_region]
    options:
      - {label: "會，10 步內", when: {reach_within: 10}}
      - {label: "不會，要 11～60 步", when: {reach_within: 60, not_reach_within: 10}}
      - {label: "60 步內到不了", when: {not_reach_within: 60}}
    explain: "牛頓法第 6 步達到，但途中跳到 f = 793 的點再折回來；信賴域法拒絕不可信的大步（這次被拒絕 5 次：3 次是 f 真的上升，2 次是 Hessian 不正定時二次模型本身就預測不會下降），每次拒絕把 Δ 縮成 1/4，之後只能沿著谷前進，第 31 步才達到。穩健的代價是放棄了這條「先上升」的捷徑。"
  - id: p3
    prompt: "換到 Himmelblau，從預設起點 (−0.5, −0.5)（牛頓法會停在局部最大值），信賴域法會？"
    scenario: {problem: himmelblau, n_iter: 60}
    subject: trust_region
    show: [newton, trust_region]
    options:
      - {label: "收斂到一個最小值（f − f* < 10⁻¹⁰）", when: {reach_within: 60}}
      - {label: "梯度歸零而停止，但 f − f* ≥ 1", when: {status: converged, final_gap_above: 1}}
      - {label: "發散", when: {status: diverged}}
    explain: "Hessian 不是正定時，本頁的 dogleg 改沿最陡下降方向走到圓的邊界，加上只接受讓 f 下降的步，所以不會往上爬：第 8 步收斂到最小值 (−2.81, 3.13)。"
---

## 定義

**想法：** 線搜尋法先決定方向、再決定走多遠；信賴域法**先決定能走多遠**（半徑 $\Delta$），再在這個範圍內找最好的一步。

每一步在 $x_k$ 附近建立二次模型，並在半徑內最小化：

$$
\min_{\lVert p\rVert\le\Delta_k}\ m_k(p)=f(x_k)+g_k^\top p+\tfrac12\,p^\top H_k\,p
$$

然後比較**實際下降**與**模型預測的下降**：

$$
\rho_k=\frac{f(x_k)-f(x_k+p_k)}{m_k(0)-m_k(p_k)}
$$

- $\rho_k<0.25$：模型不準，縮小半徑 $\Delta\leftarrow\Delta/4$。
- $\rho_k>0.75$ 且步子碰到邊界：模型很準，放大半徑 $\Delta\leftarrow\min(2\Delta,\,10)$（本站設上限 10）。
- $\rho_k>0.1$ 才接受這一步，否則 $x$ 不動。

<details markdown="1">
<summary>子問題怎麼解？dogleg 折線法</summary>

當 $H$ 正定時，dogleg 用一條兩段的折線近似子問題的解：

1. 若牛頓步 $p_N=-H^{-1}g$ 在半徑內，就直接用它。
2. 否則先沿 $-g$ 走到模型的最低點 $p_U=-\dfrac{g^\top g}{g^\top Hg}\,g$（柯西點）。
3. 若 $p_U$ 已經超出半徑，就沿 $-g$ 走到邊界為止。
4. 否則沿折線 $p_U\to p_N$ 走，直到碰到邊界（解一個一元二次方程）。

當 $H$ 不正定時，本站改沿 $-g$ 走到邊界。半徑小時這一步像梯度下降，半徑大時像牛頓法。

</details>

<details markdown="1">
<summary>為什麼信賴域法可以放心用不正定的 Hessian？</summary>

子問題有 $\lVert p\rVert\le\Delta$ 的限制，即使模型在某個方向往下無限延伸（Hessian 有負特徵值），解仍然有界。比值 $\rho$ 會檢查這一步是否真的讓 $f$ 下降，所以不會像純牛頓法一樣被吸到最大值或鞍點。在合理條件下，信賴域法具有**全域收斂**保證，而且接近解、半徑夠大時會自動變成牛頓法，保有二次收斂。

</details>

## 應用場景

- **非線性最小平方：** Levenberg-Marquardt 本質上就是信賴域法。
- **科學計算套件：** 許多通用最佳化器提供以信賴域為核心的方法，特別適合有精確 Hessian 的中小型問題。
- **有限制最佳化：** SQP 等方法常以信賴域控制每一步的大小，避免線性化失準。

## 解決什麼問題

牛頓法只在解附近可靠：離解太遠時二次模型失準，可能大幅跳躍、上升，甚至收斂到最大值。

信賴域法讓「模型可信的範圍」成為演算法的一部分：

- 模型準就放大範圍，接近牛頓法的速度。
- 模型不準就縮小範圍，退化成小步的最陡下降。
- Hessian 不正定也能安全使用。

代價是每一步要解一個帶限制的子問題，被拒絕的步會浪費一次函數計算。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 預設的 Rosenbrock，起點 $(-1.5,2.5)$：信賴域法第 27 步達到 $f<10^{-10}$，途中有 3 步被拒絕。播放時注意**虛線圓**：它就是這一步的信賴範圍，半徑在 0.06 到 1 之間變化。
2. 比較牛頓法：第 6 步就達到 $f<10^{-10}$，但中途跳到圖外 $f=793$ 的點。信賴域法的每一步都在圓內，而且被接受的步一定讓 $f$ 下降。
3. 把初始半徑 $\Delta_0$ 調到 3：第 31 步達到 $f<10^{-10}$，被拒絕 5 次，半徑會先被快速縮小再慢慢放大。調到 0.05：第 28 步達到 $f<10^{-10}$，半徑會逐步放大到 0.8。**初始半徑不太影響結果**，因為它會自我調整。
4. 換到 Himmelblau：從預設起點，牛頓法收斂到最大值，信賴域法收斂到最小值 $(-2.81,3.13)$，第 8 步達到 $f<10^{-10}$。

## 程式與練習

<!-- practice -->

## 失效情境

- **子問題解得不好：** dogleg 只在 Hessian 正定時效果好；Hessian 嚴重不定時，需要更精確的子問題解法。
- **半徑規則調得不好：** 縮放常數不合適時，可能反覆拒絕、浪費大量計算。
- **高維問題：** 和牛頓法一樣需要 Hessian 或其近似；大型問題常改用截斷共軛梯度（Steihaug CG）解子問題。
- **局部最小值：** 全域收斂指的是收斂到駐點，不保證是全域最小值。

## 面試追問

<details markdown="1">
<summary>Q1. 信賴域法和線搜尋法的差別是什麼？</summary>

線搜尋法先決定方向，再沿方向決定步長；信賴域法先決定可以走多遠（半徑），再在範圍內同時決定方向和步長。方向因此會隨半徑改變：半徑小時接近最陡下降，半徑大時接近牛頓方向。

</details>

<details markdown="1">
<summary>Q2. 比值 ρ 有什麼作用？</summary>

ρ 是實際下降與模型預測下降的比值，用來衡量模型在目前範圍內是否可信。ρ 接近 1 表示模型很準，可以放大半徑；ρ 很小或為負表示模型不準，要縮小半徑並拒絕這一步。

</details>

<details markdown="1">
<summary>Q3. 為什麼信賴域法不怕 Hessian 不正定？</summary>

因為子問題有 $\lVert p\rVert\le\Delta$ 的限制，模型即使往下無界，解仍然有界；而且 ρ 會檢查實際下降，不會接受讓 $f$ 上升的步。牛頓法則會直接跳到二次模型的駐點，可能是最大值。

</details>

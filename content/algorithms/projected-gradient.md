---
review: draft
lab:
  problem: fert_disk
  problems: [fert_disk, reservoir, fert_budget, rosen_disk]
  algos: [projected_gradient]
  compare: [frank_wolfe]
  n_iter: 200
practice: projected_gradient
predict:
  - id: p1
    prompt: "把起點設在圓盤外的 (3, 3)（η = 0.1）。投影梯度會？"
    scenario: {x0: [3.0, 3.0]}
    show: [projected_gradient]
    options:
      - {label: "無法開始：起點不可行，第 1 步就停止", when: {stops_within: 1, final_dist_above: 1.0e-6}}
      - {label: "第一步被拉回圓盤，之後收斂到 x*", when: {final_dist_below: 1.0e-6}}
      - {label: "發散", when: {status: diverged}}
    explain: "第一步就被投影回圓周，第 32 步停止，距離 3.5 × 10⁻¹³。投影梯度每步都把點投影回可行集，所以起點可以不可行；這點和需要嚴格可行起點的障礙法不同。"
  - id: p2
    prompt: "圓盤上把 η 調到 0.5，超過 GD 的穩定上限 2/L = 2/6 ≈ 0.33。投影梯度會？"
    scenario: {params: {projected_gradient: {lr: 0.5}}}
    show: [projected_gradient]
    options:
      - {label: "收斂：200 步內停止，且 ‖x − x*‖ < 10⁻⁶", when: {stops_within: 200, final_dist_below: 1.0e-6}}
      - {label: "不收斂：200 步內停不下來", when: {not_stops_within: 200}}
      - {label: "發散", when: {status: diverged}}
    explain: "仍收斂，第 35 步停止，每一步都折返。在 x* 附近，沿圓周的誤差每步乘上 (1 − η·τᵀ∇²fτ)/(1 + 2ηλ*) ≈ (1 − 0.5 × 4.32)/(1 + 0.5 × 2 × 1.63) ≈ −0.44：投影回彎曲的圓周時，乘數 λ* 那一項把切線方向的誤差也縮小了。"
  - id: p3
    prompt: "換成「施肥預算（三角形）」，η = 0.35。投影梯度會？"
    scenario: {problem: fert_budget, params: {projected_gradient: {lr: 0.35}}}
    show: [projected_gradient]
    options:
      - {label: "收斂：200 步內停止，且 ‖x − x*‖ < 10⁻⁶", when: {stops_within: 200, final_dist_below: 1.0e-6}}
      - {label: "不收斂：200 步內停不下來", when: {not_stops_within: 200}}
      - {label: "發散", when: {status: diverged}}
    explain: "收斂，第 33 步停止。x* 在預算線 x₁ + x₂ = 3 上；貼著這條線時，投影只是去掉法向分量，等於在線上做 GD。沿線方向 d = (1, −1)/√2 的曲率 dᵀ∇²f d = (2 + 6)/2 = 4，誤差每步乘上 1 − 0.35 × 4 = −0.4：翻面但縮小。"
  - id: p4
    prompt: "同一個三角形，把 η 再調到 0.5。投影梯度會？"
    scenario: {problem: fert_budget, params: {projected_gradient: {lr: 0.5}}}
    show: [projected_gradient]
    options:
      - {label: "收斂：200 步內停止，且 ‖x − x*‖ < 10⁻⁶", when: {stops_within: 200, final_dist_below: 1.0e-6}}
      - {label: "不收斂：200 步內停不下來", when: {not_stops_within: 200}}
      - {label: "發散", when: {status: diverged}}
    explain: "不收斂：誤差每步乘上 1 − 0.5 × 4 = −1，大小不變只翻面，迭代點在預算線上的 (0.5, 2.5) 與 (2.5, 0.5) 之間來回跳，距離一直是 1.41。投影只把點留在可行集內，不能取代合適的步長；圓盤上同樣的 η 能收斂，是因為彎曲的邊界多了縮小的效果。"
---

## 定義

**問題：** $\min_{x\in C}f(x)$，$C$ 是一個「投影很便宜」的閉凸集合，例如箱型、圓盤（球）、單純形。

**一句話：** 照常走一步梯度下降；走出 $C$ 了，就換成 $C$ 裡離它最近的點。

$$
x_{k+1}=P_C\big(x_k-\eta\,\nabla f(x_k)\big),\qquad P_C(y)=\arg\min_{x\in C}\lVert x-y\rVert
$$

**常見集合的投影：**

| 集合 | $P_C(y)$ |
|---|---|
| 箱型 $l\le x\le u$ | 逐座標截斷：$\min(u_i,\max(l_i,y_i))$ |
| 圓盤 $\lVert x-c\rVert\le r$ | 在圓內不動；否則 $c+r\,\dfrac{y-c}{\lVert y-c\rVert}$ |
| 凸多邊形 | 在內部不動；否則取各邊上最近點中最近的一個（本站寫法，只適合二維小例子） |

本站的動畫用固定步長 $\eta$，迭代點不再移動（$\lVert x_{k+1}-x_k\rVert<10^{-12}$）時停止。

<details markdown="1">
<summary>推導第 1 層：不動點就是 KKT 點</summary>

投影的特徵：$p=P_C(y)$ 若且唯若對所有 $z\in C$，$(y-p)^\top(z-p)\le0$，也就是 $y-p$ 落在 $C$ 在 $p$ 的**法錐**裡。

若 $x^*=P_C(x^*-\eta\nabla f(x^*))$，代入 $y=x^*-\eta\nabla f(x^*)$、$p=x^*$：

$$
-\nabla f(x^*)^\top(z-x^*)\le0\quad\forall z\in C
$$

也就是「沿任何可行方向，$f$ 的一階變化都不是下降」。當 $C$ 由 $g_i\le0$ 描述時，這等價於 KKT 條件：$-\nabla f(x^*)$ 是起作用限制梯度的非負組合。所以

$$
G_\eta(x)=\frac{x-P_C(x-\eta\nabla f(x))}{\eta}
$$

（梯度映射）可以當作停止條件，它在無限制時就是 $\nabla f(x)$。動畫的步驟說明顯示的就是 $\lVert G_\eta\rVert$。

</details>

<details markdown="1">
<summary>推導第 2 層：為什麼投影不會讓情況變糟</summary>

投影到凸集合是**非擴張**的：$\lVert P_C(a)-P_C(b)\rVert\le\lVert a-b\rVert$。把 $x^*=P_C(x^*-\eta\nabla f(x^*))$ 和更新式相減：

$$
\lVert x_{k+1}-x^*\rVert\le\big\lVert (x_k-\eta\nabla f(x_k))-(x^*-\eta\nabla f(x^*))\big\rVert
$$

右邊正是無限制 GD 的誤差遞迴。所以強凸、$L$-smooth 時，投影梯度保有和 [GD](gd.html) 一樣的線性收斂率（例如 $\eta=2/(\mu+L)$ 時每步誤差乘上 $(\kappa-1)/(\kappa+1)$）。施肥例子 $f=(x_1-3)^2+3(x_2-2)^2$ 的 $\mu=2$、$L=6$，GD 的步長上限是 $2/L\approx0.33$。

</details>

<details markdown="1">
<summary>類比：撞到邊界的水流</summary>

梯度下降像水往低處流；投影就是遇到河岸時水不會穿過去，只會沿著河岸滑動。在 $x^*$ 處，重力（$-\nabla f$）正好垂直壓在河岸上，沿岸方向沒有分量，所以停住，這就是法錐條件。

</details>

## 應用場景

- **水文模式參數校準：** 參數有物理上合理的上下限（例如滲透係數、蓄水容量），箱型投影就是把每個參數截回範圍內，幾乎不增加計算量。
- **非負限制：** 非負最小平方、非負矩陣分解、濃度或排放量反演中的 $x\ge0$，投影是 $\max(0,x)$。
- **機率與比例：** 混合比例、作物面積分配比例、源解析的貢獻比例落在單純形上，單純形投影有 $O(n\log n)$ 的排序演算法。
- **機器學習：** 以範數球限制權重大小（$\ell_2$ 球投影是縮放、$\ell_\infty$ 球是截斷）。

## 解決什麼問題

[懲罰法](penalty.html)與[增廣 Lagrangian 法](augmented-lagrangian.html)允許迭代點越界，要靠額外的外迴圈把它拉回來。當可行集合簡單到可以直接算出最近點時，這些都是多餘的：

- 每個迭代點都可行，沒有懲罰係數要調。
- 程式只比 GD 多一行，收斂分析也幾乎照搬 GD。

代價是：只適用於投影便宜的集合。一般的多面體或非線性限制，求投影本身就是一個二次規劃；這時可以改用只需要線性子問題的 [Frank-Wolfe](frank-wolfe.html)。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. **預設（土壤養分圓盤，$\eta=0.1$）：** 第 34 步停止，距離 $7.4\times10^{-13}$。前兩步在圓內照常走 GD，第 3 步碰到圓周，之後每一步都被拉回圓周，沿著邊界滑到 $x^*$。
2. **調步長：** 圓盤上 $\eta=0.01$ 要 337 步（把迭代數調高），$\eta=0.2$ 只要 14 步，$\eta=0.5$ 要 35 步。$\eta=0.5$ 已超過 GD 的上限 $2/L\approx0.33$，在圓盤上仍收斂，因為投影把來回跳動的點拉回圓周。
3. **換成「施肥預算（三角形）」：** $\eta=0.1$ 時 53 步，$\eta=0.35$ 時 33 步，$\eta=0.5$ 時就不收斂了（在程式中跑 2000 步後距離仍是 1.41；動畫上限 500 步），迭代點在預算線上的 $(0.5,\,2.5)$ 與 $(2.5,\,0.5)$ 之間來回跳。**投影不能取代合適的步長**。
4. **換成「兩座水庫放水（箱型）」：** $\eta=0.1$ 時 76 步停在 $x_2=1$ 這條邊上的 $(5/3,\,1)$。投影只是把 $x_2$ 截在 1，$x_1$ 方向照常做 GD。
5. **點擊圓盤外面設成起點：** 例如 $(3,3)$，第一步就被投影回圓周，32 步停止。投影梯度不需要可行的起點。
6. **換成「Rosenbrock + 圓盤（非凸）」：** $\eta=0.1$ 完全不收斂。$\eta=0.003$ 要 809 步才停止，$\eta=0.002$ 要 1061 步（500 步時距離還有 0.41）；$\eta=0.004$ 跑 5000 步也沒有收斂（距離 0.3）。超過 500 步的數字超出動畫的迭代上限，需在程式中執行。可用的步長區間很窄，而且病態問題上投影梯度和 GD 一樣慢。
7. **勾選 Frank-Wolfe 比較：** 在圓盤上 200 步後距離 $1.3\times10^{-5}$，比投影梯度的 34 步慢；Frank-Wolfe 的優勢在投影很貴的集合，不在這裡。

## 程式與練習

<!-- practice -->

## 失效情境

- **投影很貴：** 一般的多面體 $Ax\le b$ 或非線性限制，投影本身就要解一個最佳化問題，每一步的成本可能比梯度還高。
- **非凸集合：** 最近點可能不唯一，非擴張性也不成立，收斂分析失效。
- **步長：** 投影不會修好過大的步長（實驗 3）；步長太小則和 GD 一樣慢（實驗 6）。
- **病態：** 收斂率仍由條件數決定，邊界上的窄長峽谷一樣會鋸齒狀前進。

## 面試追問

<details markdown="1">
<summary>Q1. 投影梯度法的停止條件為什麼不能用 ‖∇f‖ 很小？</summary>

有限制的最優解通常在邊界上，那裡 $\nabla f\ne0$，被限制的推力抵銷而已。正確的量是梯度映射 $G_\eta(x)=(x-P_C(x-\eta\nabla f(x)))/\eta$，它為 0 若且唯若 $x$ 是投影的不動點，也就是滿足 KKT 條件；無限制時它就等於 $\nabla f$。

</details>

<details markdown="1">
<summary>Q2. 投影梯度的收斂率和 GD 相比如何？為什麼？</summary>

在凸、$L$-smooth 的情況下收斂率和 GD 相同。關鍵是投影到凸集合是非擴張的，$\lVert P(a)-P(b)\rVert\le\lVert a-b\rVert$，加上最優解是投影的不動點，所以誤差遞迴不會比無限制的 GD 差。

</details>

<details markdown="1">
<summary>Q3. 什麼時候該用投影梯度，什麼時候該用 Frank-Wolfe？</summary>

投影便宜（箱型、球、單純形）時用投影梯度，程式簡單且收斂快。投影很貴、但線性子問題 $\min_{s\in C}g^\top s$ 很便宜時（例如一般多面體、核範數球、流量多面體）用 Frank-Wolfe，它不需要投影，而且迭代點是少數頂點的凸組合。

</details>

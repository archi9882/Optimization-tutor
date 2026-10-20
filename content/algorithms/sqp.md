---
review: draft
lab:
  problem: fert_disk
  problems: [fert_disk, fert_budget, fert_eq, reservoir, rosen_disk, maratos]
  algos: [sqp, sqp_bfgs]
  compare: [sqp_raw, penalty, augmented_lagrangian]
  n_iter: 30
practice: sqp
predict:
  - id: p1
    prompt: "預設（土壤養分圓盤，限制是彎的）：SQP（Lagrangian Hessian）會在第幾步停止？（懲罰法要 308 步、增廣 Lagrangian 要 208 步）"
    scenario: {}
    show: [sqp, sqp_bfgs]
    options:
      - {label: "10 步內", when: {stops_within: 10}}
      - {label: "11～30 步", when: {stops_within: 30, not_stops_within: 10}}
      - {label: "30 步內停不下來", when: {not_stops_within: 30}}
    explain: "第 7 步停止，距離 2.2 × 10⁻¹⁶。看收斂圖的最後幾步：距離 0.051 → 0.0012 → 7.6 × 10⁻⁷ → 1.3 × 10⁻¹²，每一步的誤差大約是上一步的平方，這就是牛頓法的二次收斂。阻尼 BFGS 版第 10 步停止：一開始 B = I 不準，要幾步才學到曲率。"
  - id: p2
    prompt: "換成「施肥預算（三角形）」：目標是二次函數、限制都是線性的。SQP（Lagrangian Hessian）會在第幾步停止？"
    scenario: {problem: fert_budget}
    show: [sqp, sqp_bfgs]
    options:
      - {label: "2 步內", when: {stops_within: 2}}
      - {label: "3～10 步", when: {stops_within: 10, not_stops_within: 2}}
      - {label: "10 步內停不下來", when: {not_stops_within: 10}}
    explain: "第 2 步停止。二次目標 + 線性限制時，QP 子問題就是原問題本身：第 1 步直接落在 x* = (1.5, 1.5)，得到 λ = (3, 0, 0)；第 2 步算出 d = 0，確認是 KKT 點後停止。BFGS 版從 B = I 出發，模型不準，要到第 4 步。"
  - id: p3
    prompt: "換成「Rosenbrock + 圓盤（非凸）」，起點 (−0.6, 0.4)，看「SQP（Hessian 不修正）」：直接用 ∇²L 當 B，會怎樣？"
    scenario: {problem: rosen_disk, x0: [-0.6, 0.4]}
    subject: sqp_raw
    show: [sqp, sqp_raw]
    options:
      - {label: "和修正版一樣收斂（‖x − x*‖ < 10⁻⁶）", when: {final_dist_below: 1.0e-6}}
      - {label: "走到半路停下（10⁻⁶ ≤ ‖x − x*‖ < 1）", when: {final_dist_above: 1.0e-6, final_dist_below: 1.0}}
      - {label: "幾乎沒前進（‖x − x*‖ ≥ 1）", when: {final_dist_above: 1.0}}
    explain: "第 1 步就停在起點，距離仍是 1.40：Rosenbrock 在 (−0.6, 0.4) 的 Hessian 特徵值是 −5.84 與 480，QP 的解 d 不是價值函數的下降方向，回溯 40 次都不被接受。修正版加上 τ = 5.84 讓 B 正定，第 13 步停止。"
  - id: p4
    prompt: "換成「圓上等式（Maratos 例）」，起點 (0.99875, 0.04998) 幾乎就在圓上、離 x* = (1, 0) 只有 0.05。SQP（Lagrangian Hessian）會在第幾步停止？"
    scenario: {problem: maratos, x0: [0.99875, 0.04998]}
    show: [sqp]
    options:
      - {label: "5 步內（每步都走整步，二次收斂）", when: {stops_within: 5}}
      - {label: "6～15 步", when: {stops_within: 15, not_stops_within: 5}}
      - {label: "15 步內停不下來", when: {not_stops_within: 15}}
    explain: "第 10 步停止。前 6 步的步長 t 是 0.5、0.125、0.125、0.25、0.25、0.5，距離每步只縮小到 0.5～0.88 倍；第 7 步起才走整步，之後才二次收斂。原因是整步讓違反量 |h| 從 4.4 × 10⁻⁷ 升到 1.6 × 10⁻⁴，ℓ1 價值函數變大而被拒絕，這就是 Maratos 效應。同一個起點若每步都走整步，第 5 步就停止。"
---

## 定義

**問題：** $\min f(x)$，限制 $g_i(x)\le0$（不等式）與 $h_j(x)=0$（等式），$f$、$g$、$h$ 都平滑。

**一句話：** 在目前的點用一個**二次規劃（QP）**近似原問題：目標換成二次模型，限制換成一階線性化。解這個 QP 得到方向 $d$，沿 $d$ 走一步，重複。

$$
\min_d\ \tfrac12 d^\top B_k d+\nabla f(x_k)^\top d
\quad\text{s.t.}\quad g_i(x_k)+\nabla g_i(x_k)^\top d\le0,\quad h_j(x_k)+\nabla h_j(x_k)^\top d=0
$$

- $B_k$ 是 **Lagrangian** $L(x,\lambda)=f(x)+\sum_i\lambda_ig_i(x)$ 的 Hessian（或它的擬牛頓近似），**不是** $f$ 的 Hessian：限制的曲率要算進去。
- QP 的乘數就是下一步的 $\lambda$ 估計。
- 二維時最多兩個限制同時起作用，本站逐一試「工作集」並解 KKT 方程組（見示範程式的 `qp_subproblem`），這是 [有效集法](qp-active-set.html) 的暴力版。

**全域化：** 單純走整步只在 $x^*$ 附近可靠。本站用 **ℓ1 價值函數**（merit function）做回溯線搜尋：

$$
\phi_\mu(x)=f(x)+\mu\Big(\sum_i\max(0,g_i(x))+\sum_j|h_j(x)|\Big),\qquad \mu\leftarrow\max\big(\mu,\ 2\max_i|\lambda_i|\big)
$$

$t$ 從 1 開始減半，直到 $\phi_\mu(x+td)\le\phi_\mu(x)+10^{-4}\,t\,D$，$D=\nabla f^\top d-\mu\cdot(\text{違反量})$。$\lVert d\rVert<10^{-10}$ 時停止。

**$B$ 的兩種選法（動畫的兩列）：**

| | $B$ | 本站做法 |
|---|---|---|
| Lagrangian Hessian | $\nabla^2f+\sum\lambda_i\nabla^2g_i$ | 用上一步的 $\lambda$；最小特徵值小於 $10^{-4}$ 時加 $\tau I$（**Hessian 修正**） |
| 阻尼 BFGS | 擬牛頓近似，從 $I$ 開始 | $y=\nabla_xL(x_{k+1},\lambda_{k+1})-\nabla_xL(x_k,\lambda_{k+1})$；$s^\top y<0.2\,s^\top Bs$ 時把 $y$ 往 $Bs$ 拉（Powell 阻尼），$B$ 一定保持正定 |

<details markdown="1">
<summary>推導第 1 層：SQP 就是對 KKT 條件做牛頓法</summary>

只看等式限制 $h(x)=0$。KKT 條件是一組非線性方程式：

$$
F(x,\lambda)=\begin{bmatrix}\nabla f(x)+\nabla h(x)^\top\lambda\\ h(x)\end{bmatrix}=0
$$

對 $(x,\lambda)$ 做一步[牛頓法](newton.html)，Jacobian 是

$$
\begin{bmatrix}\nabla^2_{xx}L & \nabla h^\top\\ \nabla h & 0\end{bmatrix}
\begin{bmatrix}d\\ \lambda_{+}-\lambda\end{bmatrix}
=-\begin{bmatrix}\nabla f+\nabla h^\top\lambda\\ h\end{bmatrix}
$$

把 $\nabla h^\top\lambda$ 移到兩邊消掉，得到

$$
\begin{bmatrix}\nabla^2_{xx}L & \nabla h^\top\\ \nabla h & 0\end{bmatrix}
\begin{bmatrix}d\\ \lambda_{+}\end{bmatrix}
=-\begin{bmatrix}\nabla f\\ h\end{bmatrix}
$$

這正是 QP 子問題（$B=\nabla^2_{xx}L$）的 KKT 方程組，也就是 `qp_subproblem` 解的那個矩陣。所以在 $x^*$ 附近、走整步時，SQP 繼承牛頓法的**二次收斂**。土壤圓盤上實測的距離：$0.051\to1.2\times10^{-3}\to7.6\times10^{-7}\to1.3\times10^{-12}$，比值 $e_{k+1}/e_k^2$ 依序是 0.48、0.51、2.3（最後一個已碰到浮點誤差）。

不等式限制：只要 QP 猜對了哪些限制起作用，剩下的就是上面的等式情形；接近解時起作用集合通常不再變。

</details>

<details markdown="1">
<summary>推導第 2 層：為什麼 μ 要大於 |λ|</summary>

$d$ 滿足線性化限制時，ℓ1 價值函數沿 $d$ 的方向導數不超過

$$
D=\nabla f^\top d-\mu\,\lVert c(x)\rVert_1
$$

（$c$ 是違反量）。用 QP 的 KKT 條件 $Bd+\nabla f+A^\top\lambda=0$ 代換 $\nabla f^\top d$，再用 $A d=-c$：

$$
D=-d^\top Bd+\lambda^\top c-\mu\lVert c\rVert_1\le-d^\top Bd-(\mu-\lVert\lambda\rVert_\infty)\lVert c\rVert_1
$$

$B$ 正定且 $\mu\ge\lVert\lambda\rVert_\infty$ 時 $D<0$，回溯一定找得到可接受的 $t$。這同時說明了 **B 必須正定**：$B$ 不正定時 $-d^\top Bd$ 可能是正的，$d$ 就不再是下降方向（預測題 3）。

</details>

<details markdown="1">
<summary>推導第 3 層：Maratos 效應</summary>

圓上等式的例子：$f=2(x_1^2+x_2^2-1)-x_1$，$h=x_1^2+x_2^2-1=0$，$x^*=(1,0)$，$\lambda^*=-3/2$。在圓上 $f=-x_1$。

從圓上的點出發，QP 的整步沿著切線走，切線在圓外，所以 $h$ 從 0 變成正的，大小約為 $\lVert d\rVert^2$。實測從 $(0.99875,\,0.04998)$ 出發（$h=-4.4\times10^{-7}$）：整步後 $f$ 從 $-0.998751$ 降到 $-0.999062$，但 $|h|$ 升到 $1.56\times10^{-4}$，$\mu\approx3$，$\phi$ 淨增加，整步被拒絕。

整步其實離 $x^*$ 更近（二次收斂的那一步），是價值函數把它擋掉了。常見補救是二階修正（second-order correction，再加一個把點拉回限制上的小步）或改用其他價值函數；本站沒有實作。

</details>

<details markdown="1">
<summary>類比：用拋物面近似地形、用切平面近似河岸</summary>

在山谷中找最低點又不能越過河岸：每走一步前，先用一個碗（二次模型）近似腳下的地形，用一條直線（切線）近似彎曲的河岸，算出「碗在直線內側的最低點」，再朝那裡走。河岸彎得厲害時，直線版的河岸會騙人，所以要用價值函數檢查走過去是不是真的比較好。

</details>

## 應用場景

- **工程設計：** 結構尺寸最佳化、軌跡最佳化（太空船、機器人）這類變數不多但限制非線性、函數計算昂貴的問題，SQP 用很少的迭代數找到精確可行的解。
- **能源：** 電力系統的最佳潮流（optimal power flow）有非線性的潮流方程式等式限制。
- **水資源：** 水庫調度若把水位—容量、發電量—落差寫成非線性關係，就是一個非線性限制的最佳化問題。
- **模型校準：** 地球物理或水文模式的反演，參數之間有非線性的物理限制（例如質量守恆）時，可在擬牛頓的框架下直接處理限制。

## 解決什麼問題

[懲罰法](penalty.html)和[增廣 Lagrangian 法](augmented-lagrangian.html)把限制塞進目標函數，再用一階的內迴圈慢慢解，同一個問題要幾百步。SQP 把[牛頓法](newton.html)的想法直接用在 KKT 條件上：

- 接近解時二次收斂（或 BFGS 版的超線性收斂），迭代數通常是個位數到幾十。
- 每步同時更新 $x$ 和乘數 $\lambda$，結束時直接得到影子價格。
- 不需要懲罰係數趨近無限大，也不需要可行的起點。

代價是每一步要解一個 QP，並需要二階資訊或擬牛頓近似；變數很多時 QP 本身就很貴。

## 動畫演示

<!-- lab -->

細線段是 QP 給的整步 $x\to x+d$，實心點是線搜尋接受的位置；兩者分開表示步長被截短了。步驟說明列出 $\lambda$、$\mu$、步長 $t$、$\lVert d\rVert$ 與 Hessian 修正量 $\tau$。

**建議的實驗：**

1. **預設（土壤養分圓盤）：** Lagrangian Hessian 版第 7 步停止，每一步都是整步（$t=1$）。第 1 步直接跳到圓外的 $(3,\,2)$（違反量 4.0）：起點的 $\lambda=0$，而在起點線性化的圓盤限制（一個半平面）包含 $(3,2)$；之後的迭代從圓外收斂回來，最後三步的距離 $7.6\times10^{-7}\to1.3\times10^{-12}\to2.2\times10^{-16}$；阻尼 BFGS 版第 10 步。
2. **勾選懲罰函數法與增廣 Lagrangian，迭代數調到 400：** 同一個圓盤，懲罰法第 308 步停止、距離 $1.1\times10^{-3}$；增廣 Lagrangian 第 208 步、距離 $4.0\times10^{-8}$。
3. **換成「施肥預算（三角形）」「預算全部用完（等式）」或「兩座水庫放水（箱型）」：** 都是第 2 步停止，第 1 步就落在 $x^*$。水庫問題的乘數 $\lambda=(0,\ 1.3333,\ 0,\ 0)$，和手算的 $4/3$ 一致，只有 $x_2\le1$ 起作用。
4. **換成「Rosenbrock + 圓盤（非凸）」：** 第 16 步停止。第 1 步的修正量 $\tau=43.0$：Rosenbrock 在起點 $(-0.6,\,0.6)$ 的 Hessian 有負特徵值 $-43.0$。阻尼 BFGS 版第 23 步。再勾選「Hessian 不修正」並點擊 $(-0.6,\,0.4)$ 當起點：不修正的版本第 1 步就停在原地（線搜尋失敗），修正版第 13 步停止。這一帶的起點不修正時常常失敗：以 $(-0.6,\,0.4)$ 為中心、間距 0.015 的 25 個起點中，有 12 個在第 1～2 步就停下（線搜尋失敗或 QP 無解），修正版全部在 13～14 步收斂。
5. **換成「圓上等式（Maratos 例）」：** 預設起點第 12 步停止，前 7 步的步長依序是 0.5、0.0625、0.125、0.125、0.125、0.25、0.5，第 8 步起才走整步。接近解時整步被價值函數拒絕，就是 Maratos 效應（推導第 3 層）。
6. **同一個問題，起點在 $x_1$ 軸的左半邊：** 起點 $(-0.5,\,0)$ 第 6 步停在 $(-1,\,0)$，距離 $x^*$ 是 2。$(-1,0)$ 也滿足 KKT 條件（在圓上 $f=-x_1$，它是最大值）：SQP 只保證收斂到 KKT 點。點擊很難剛好點在軸上，這個起點要在程式中執行；稍微離開軸，例如 $(-0.5,\,0.05)$，第 52 步就回到 $x^*$，因為最大值是不穩定的。

## 程式與練習

<!-- practice -->

## 失效情境

- **線性化的限制無解：** 圓上等式在圓心 $(0,0)$：$\nabla h=0$ 而 $h=-1$，線性化的限制變成 $-1=0$，QP 不可行，動畫顯示「QP 子問題無解」並停止（在程式中以起點 $(0,0)$ 執行；練習的單步測試也有這一題）。實務上的補救是放寬限制（elastic mode）或改用可行性恢復階段。
- **Hessian 不正定：** QP 可能無下界，或工作集的 KKT 解只是鞍點，$d$ 不是下降方向（實驗 4、預測題 3）。對策是加 $\tau I$ 修正，或用保持正定的阻尼 BFGS。
- **Maratos 效應：** 接近解時整步被 ℓ1 價值函數拒絕，失去二次收斂（實驗 5、預測題 4）。
- **只找到 KKT 點：** 非凸問題可能收斂到最大值或鞍點（實驗 6），結果依起點而定。
- **大規模問題：** 每步都要解 QP 並保存 $n\times n$ 的 $B$；變數上萬時要改用稀疏或有限記憶的版本。

## 面試追問

<details markdown="1">
<summary>Q1. SQP 和牛頓法是什麼關係？</summary>

SQP 是對 KKT 條件 $\nabla_xL=0$、$h=0$ 做牛頓法。牛頓法的線性方程組正好是 QP 子問題 $\min\frac12d^\top\nabla^2_{xx}L\,d+\nabla f^\top d$、$h+\nabla h\,d=0$ 的 KKT 方程組，所以在解附近走整步時是二次收斂。不等式限制則交給 QP 去決定哪些起作用。

</details>

<details markdown="1">
<summary>Q2. QP 子問題的 Hessian 為什麼用 Lagrangian 的，而不是 f 的？</summary>

限制的曲率會改變可行方向上的二次變化。只用 $\nabla^2f$ 等於假設限制是直的；在彎曲的限制上，正確的二次模型是 $\nabla^2f+\sum\lambda_i\nabla^2g_i$。例如圓上等式的例子，$\nabla^2f=4I$，但 $\nabla^2L=(4+2\lambda^*)I=I$，差了四倍。

</details>

<details markdown="1">
<summary>Q3. 為什麼需要價值函數？μ 怎麼選？</summary>

整步只在解附近可靠，離得遠時需要一個量來判斷「走過去是不是比較好」。單看 $f$ 會忽略可行性，單看違反量會忽略目標，ℓ1 價值函數 $f+\mu\cdot\text{違反量}$ 把兩者合起來。只要 $\mu$ 大於所有 $|\lambda_i|$ 且 $B$ 正定，QP 的方向就是它的下降方向；本站取 $\mu\ge2\max|\lambda_i|$。

</details>

<details markdown="1">
<summary>Q4. 什麼是 Maratos 效應？怎麼補救？</summary>

接近解時，整步沿著限制的切線走，會產生 $O(\lVert d\rVert^2)$ 的限制違反，讓 ℓ1 價值函數變大，線搜尋把本來會二次收斂的整步截短，收斂變慢。補救方法有二階修正（再加一個把點拉回限制上的小步）、非單調線搜尋（watchdog），或改用增廣 Lagrangian 型的價值函數。

</details>

<details markdown="1">
<summary>Q5. 什麼時候選 SQP，什麼時候選增廣 Lagrangian 或內點法？</summary>

SQP 適合變數不多、限制非線性、函數計算昂貴的問題：迭代數少，每步的 QP 相對便宜。變數很多、Hessian 很大，或只能用一階資訊時，增廣 Lagrangian 型方法比較容易擴充；不等式限制很多的大型問題，內點法通常比在 QP 裡猜起作用集合更穩定。

</details>

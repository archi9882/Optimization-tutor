---
review: draft
lab:
  problem: fert_budget
  problems: [fert_budget, fert_eq, fert_disk, reservoir, rosen_disk]
  algos: [penalty]
  compare: [augmented_lagrangian]
  n_iter: 400
practice: penalty
predict:
  - id: p1
    prompt: "預設（施肥預算，ρ 從 1 放大到 10⁴）：懲罰法停止時，最後一點的違反量（超出預算多少）大約是？"
    scenario: {}
    show: [penalty]
    options:
      - {label: "幾乎是 0：剛好落在預算線上", when: {final_violation_below: 1.0e-6}}
      - {label: "10⁻⁶～10⁻³：還在三角形外一點點", when: {final_violation_above: 1.0e-6, final_violation_below: 1.0e-3}}
      - {label: "10⁻³ 以上", when: {final_violation_above: 1.0e-3}}
    explain: "違反量 2.97 × 10⁻⁴。懲罰函數的最小點滿足 ∇f + ρ·v·∇g = 0，所以 ρ·v 近似乘數 λ* = 3，違反量 v ≈ λ*/ρ = 3/10⁴：懲罰法的誤差是 O(1/ρ)，只有 ρ → ∞ 才真正可行，迭代點一直從外面逼近。"
  - id: p2
    prompt: "同一個預設：停止時離 x* 多遠？（只看 ρ = 10⁴ 的懲罰子問題，精確解離 x* 是 2.4 × 10⁻⁴）"
    scenario: {}
    show: [penalty]
    options:
      - {label: "不到 10⁻³，接近 2.4 × 10⁻⁴", when: {final_dist_below: 1.0e-3}}
      - {label: "10⁻³～10⁻²", when: {final_dist_above: 1.0e-3, final_dist_below: 1.0e-2}}
      - {label: "10⁻² 以上", when: {final_dist_above: 1.0e-2}}
    explain: "‖x − x*‖ = 0.0050，約是子問題精確解的 20 倍。ρ 越大，懲罰函數的等高線越扁（病態），GD 內迴圈越慢：五輪的內圈步數是 43、24、100、100、100，後三輪都用完上限，沒有解到子問題的最小點。"
  - id: p3
    prompt: "跳過暖身，一開始就用 ρ₀ = 1000（γ = 10，所以兩輪就到 10⁴）。停止時離 x* 多遠？"
    scenario: {params: {penalty: {rho0: 1000.0}}}
    show: [penalty]
    options:
      - {label: "比預設更近（不到 0.005）", when: {final_dist_below: 0.005}}
      - {label: "0.005～0.1", when: {final_dist_above: 0.005, final_dist_below: 0.1}}
      - {label: "0.1 以上", when: {final_dist_above: 0.1}}
    explain: "距離 0.185，比預設差了 30 多倍，第 200 步就停止。ρ = 1000 的子問題從遠處的起點出發就很病態，GD 每輪 100 步走不完；所以 ρ 要從小放大，讓每一輪從上一輪的解（已經很近）暖啟動。"
  - id: p4
    prompt: "改成 ρ₀ = 1、放大倍數 γ = 100（三輪：ρ = 1、100、10⁴）。輪數變少，最後離 x* 會比預設（0.0050）近嗎？"
    scenario: {params: {penalty: {gamma: 100.0}}}
    show: [penalty]
    options:
      - {label: "更近（不到 0.0050）", when: {final_dist_below: 0.005}}
      - {label: "更遠（0.0050 以上）", when: {final_dist_above: 0.005}}
    explain: "更遠：第 243 步停止，距離 0.042。ρ 一次跳 100 倍，上一輪的解離新子問題的解太遠，暖啟動幫不上忙，病態的內迴圈又在步數上限內走不到。"
---

## 定義

**問題：** 有限制的最佳化

$$
\min_x f(x)\quad\text{s.t.}\quad g_i(x)\le 0\ (i=1,\dots,m),\qquad h_j(x)=0\ (j=1,\dots,p)
$$

**一句話：** 把「違反限制」換算成代價加進目標函數，再用任何無限制的方法去解。代價越重，解越靠近可行域。

**二次懲罰函數：**

$$
P_\rho(x)=f(x)+\frac{\rho}{2}\left[\sum_i \max\big(0,\,g_i(x)\big)^2+\sum_j h_j(x)^2\right]
$$

在可行域內懲罰項是 0，$P_\rho=f$；一旦越界，代價隨違反量平方增加。$\rho$ 是懲罰係數。

**演算法：** 取遞增的 $\rho_0<\rho_1<\cdots$（本站每輪乘上 $\gamma$），每一輪近似解 $\min_x P_{\rho_k}(x)$，並以上一輪的解當起點（暖啟動）。本站的內迴圈是帶 Armijo 回溯的梯度下降，$\lVert\nabla P_\rho\rVert<10^{-3}$ 或用完步數上限就進入下一輪。

<details markdown="1">
<summary>先複習：KKT 條件（有限制最優解要滿足什麼）</summary>

在適當的正則條件下，有限制的局部最小值 $x^*$ 存在乘數 $\lambda_i\ge0$、$\nu_j$ 使得：

| 條件 | 式子 | 意思 |
|---|---|---|
| 穩定性 | $\nabla f(x^*)+\sum_i\lambda_i\nabla g_i(x^*)+\sum_j\nu_j\nabla h_j(x^*)=0$ | 目標函數的下坡方向被限制的「推力」抵銷 |
| 原始可行 | $g_i(x^*)\le0$，$h_j(x^*)=0$ | 解要可行 |
| 對偶可行 | $\lambda_i\ge0$ | 不等式只能往可行域內推 |
| 互補鬆弛 | $\lambda_i\,g_i(x^*)=0$ | 沒碰到的限制，乘數為 0 |

**本站的施肥例子：** $f=(x_1-3)^2+3(x_2-2)^2$，預算 $x_1+x_2\le3$。只有預算限制起作用：

$$
2(x_1-3)+\lambda=0,\quad 6(x_2-2)+\lambda=0,\quad x_1+x_2=3
\;\Rightarrow\;\lambda^*=3,\ x^*=(1.5,\,1.5),\ f^*=3
$$

$\lambda^*=3$ 的意思是：預算每多 1 單位，最優損失大約減少 3（乘數是限制的「影子價格」）。

</details>

<details markdown="1">
<summary>推導第 1 層：懲罰解為什麼只差 O(1/ρ)</summary>

假設 $x_\rho$ 落在預算線外側（$g>0$），$P_\rho$ 的駐點條件是

$$
\nabla f(x_\rho)+\underbrace{\rho\,g(x_\rho)}_{\text{像乘數}}\,\nabla g(x_\rho)=0
$$

和 KKT 的穩定性條件長得一樣，只是乘數換成 $\rho\,g(x_\rho)$。施肥例子可以直接解：$x_1=3-\lambda/2$、$x_2=2-\lambda/6$、$\lambda=\rho\,g$，得到

$$
g(x_\rho)=\frac{6}{3+2\rho},\qquad \rho\,g(x_\rho)=\frac{6\rho}{3+2\rho}\to 3=\lambda^*
$$

所以違反量約為 $\lambda^*/\rho$：**ρ 放大 10 倍，誤差才縮小 10 倍**。代入公式（本 repo 的量測腳本實算）：$\rho=100$ 時 $\lVert x_\rho-x^*\rVert=0.0234$，$\rho=10^3$ 時 $0.00237$，$\rho=10^4$ 時 $0.000237$。另外，$x_\rho$ 一直在可行域**外面**：懲罰法是從外側逼近。

</details>

<details markdown="1">
<summary>推導第 2 層：ρ 越大，內迴圈越難解（病態）</summary>

在越界的一側，

$$
\nabla^2P_\rho=\nabla^2f+\rho\,\nabla g\,\nabla g^\top+\rho\,g\,\nabla^2g
$$

$\rho\,\nabla g\nabla g^\top$ 讓沿 $\nabla g$ 方向的曲率約為 $\rho$，其他方向不變，條件數約與 $\rho$ 成正比。施肥例子 $\nabla^2P_\rho=\mathrm{diag}(2,6)+\rho\begin{bmatrix}1&1\\1&1\end{bmatrix}$，實算條件數：$\rho=100$ 時 51.3、$10^3$ 時 501.3、$10^4$ 時 5001.3。

[梯度下降](gd.html)在條件數大的問題上需要更多步。實測（內迴圈不設上限，停在 $\lVert\nabla P\rVert<10^{-3}$，從上一輪的解暖啟動）：$\rho=100$、$10^3$、$10^4$ 分別用了 110、902、2541 步，隨 $\rho$ 明顯增加，但不是嚴格的正比。因為內迴圈只解到 $10^{-3}$，$\rho=10^4$ 那一輪結束時到 $x^*$ 的距離是 $3.9\times10^{-4}$，而不是精確懲罰解的 $2.37\times10^{-4}$。**精度要求逼著 ρ 變大，ρ 變大又讓子問題變難**，這是懲罰法的根本矛盾。

</details>

<details markdown="1">
<summary>類比：用彈簧把點拉回可行域</summary>

$\frac{\rho}{2}g^2$ 就是勁度係數 $\rho$ 的彈簧位能。目標函數的「力」$-\nabla f$ 把點往外拉，彈簧把它往回拉；兩者平衡時彈簧一定有伸長量 $g=\lambda^*/\rho$。彈簧越硬伸長越小，但系統也越「剛性」：和顯式時間積分遇到硬彈簧必須縮小時間步長是同一件事。

</details>

## 應用場景

- **PINN 與物理約束學習：** 邊界條件、守恆律常以「殘差平方 × 權重」加進損失函數，本質上就是固定 $\rho$ 的二次懲罰；權重不夠大時邊界條件只會近似成立。
- **水文與地下水模式校準：** 參數的合理範圍或水平衡條件，可以寫成懲罰項加進誤差函數，再交給無限制的最佳化器或演化演算法。
- **工程設計：** 應力、位移上限等設計規範在初步設計時常以懲罰項處理，讓同一套無限制求解器可以直接使用。
- **農業配置：** 肥料、灌溉水量的預算或環境上限（例如氮流失量），可以先以懲罰項處理，快速得到近似可行的配置。

## 解決什麼問題

無限制的方法（[GD](gd.html)、[BFGS](bfgs.html)、[Nelder-Mead](nelder-mead.html)）不知道什麼叫「不准越界」。懲罰法把限制翻譯成它們聽得懂的語言：**越界就變貴**。

- 不需要可行的起點，也不需要投影或特殊的子問題。
- 等式、不等式、非線性限制都用同一個寫法。
- 任何無限制求解器都能當內迴圈。

代價是：有限的 $\rho$ 只給出**近似可行**的解，而 $\rho$ 越大子問題越病態。後續方法各自補這個洞：

- 加上乘數，$\rho$ 不必趨近無限大 → [增廣 Lagrangian 法](augmented-lagrangian.html)
- 從內部逼近，迭代點永遠可行 → [障礙函數法](barrier.html)
- 可行集合簡單時直接投影 → [投影梯度法](projected-gradient.html)

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. **預設（施肥預算，$\rho_0=1$、$\gamma=10$、內圈上限 100）：** 第 367 步停止。實心大點是每一輪的終點，到 $x^*$ 的距離依序是 0.949、0.206、0.0235、0.0058、0.0050。注意迭代點都在三角形**外面**：最後違反量 $3.0\times10^{-4}$。五輪的內圈步數是 43、24、100、100、100：後三輪都用完上限，所以 $\rho=10^4$ 時距離只比上一輪少一點，沒有達到解析的 0.000237。
2. **把內圈步數上限調到 300、迭代數調到 500：** 500 步時還在 $\rho=10^4$ 那一輪，距離 0.0029。$\rho=10^3$ 那一輪用了 300 步，剛好用完上限，停在距離 0.0037。**大 ρ 的子問題是在消耗步數**。
3. **一開始就用 $\rho_0=1000$：** 第 200 步停止，距離 0.185。跳過小 $\rho$ 的暖身，病態的子問題從遠處出發，100 步走不到；這就是為什麼要讓 $\rho$ 逐步放大。
4. **$\gamma=100$：** 只有三輪（$\rho=1,100,10^4$），第 243 步停止，距離 0.042，比 $\gamma=10$ 還差：跳太快，暖啟動的起點離新的子問題解太遠。
5. **勾選增廣 Lagrangian：** $\rho=10$ 固定，第 209 步停止，距離 $2.3\times10^{-7}$、違反量 $9.9\times10^{-8}$。不需要大 $\rho$ 就達到可行。
6. **換成「Rosenbrock + 圓盤（非凸）」並把迭代數調到 500：** 第 476 步停止，距離 $6.2\times10^{-6}$。前兩輪（$\rho=1,10$）內圈用完 100 步還在圓內的谷底慢慢走：在可行域內懲罰項是 0，內迴圈就是普通的 GD。

## 程式與練習

<!-- practice -->

## 失效情境

- **病態：** $\rho$ 越大，$P_\rho$ 的條件數約與 $\rho$ 成正比，一階方法的內迴圈越走越慢（實驗 2、3）。
- **只有近似可行：** 有限的 $\rho$ 一定留下約 $\lambda^*/\rho$ 的違反量；限制必須嚴格成立時（例如物理守恆、法規上限），懲罰法的解不能直接用。
- **起點與 ρ 排程：** 一開始 $\rho$ 太大或放大太快，暖啟動失效（實驗 3、4）。
- **平滑度：** $\max(0,g)^2$ 只有一階連續，Hessian 在邊界上跳動，牛頓類的內迴圈在邊界附近可能來回震盪。
- **非凸問題：** 每一輪只找到 $P_\rho$ 的局部最小值，可能是不可行的局部最小值。

## 面試追問

<details markdown="1">
<summary>Q1. 二次懲罰法的解為什麼通常不可行？違反量大概多大？</summary>

$P_\rho$ 的駐點滿足 $\nabla f+\rho\,g\,\nabla g=0$，和 KKT 條件比較，$\rho\,g(x_\rho)$ 扮演乘數，所以 $g(x_\rho)\approx\lambda^*/\rho$。只要最優解的乘數 $\lambda^*\ne0$（限制真的有在擋），有限的 $\rho$ 就一定留下 $O(1/\rho)$ 的違反量。

</details>

<details markdown="1">
<summary>Q2. 為什麼不直接把 ρ 設成很大？</summary>

$\nabla^2P_\rho$ 沿限制梯度方向的曲率約為 $\rho$，條件數跟著放大，梯度類內迴圈需要的步數大約與條件數成正比，牛頓類方法則面臨數值誤差。從大 $\rho$ 直接開始，起點離子問題的解又遠，所以實務上讓 $\rho$ 逐步放大並暖啟動。

</details>

<details markdown="1">
<summary>Q3. 懲罰法和增廣 Lagrangian 法差在哪？</summary>

增廣 Lagrangian 在懲罰項之外多了一個乘數估計 $\lambda$，每輪用 $\lambda\leftarrow\lambda+\rho\,g$ 修正。乘數收斂到 $\lambda^*$ 後，有限的 $\rho$ 就能讓解精確可行，不需要讓 $\rho$ 趨近無限大，子問題也就不會越來越病態。

</details>

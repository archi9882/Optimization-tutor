---
review: draft
lab:
  problem: fert_lasso
  problems: [fert_lasso, fert_budget, fert_disk, reservoir]
  algos: [admm]
  compare: [augmented_lagrangian, projected_gradient]
  n_iter: 150
practice: admm
predict:
  - id: p1
    prompt: "預設（施肥稀疏，ρ = 1）：ADMM 會在第幾步停止？"
    scenario: {}
    show: [admm]
    options:
      - {label: "50 步內", when: {stops_within: 50}}
      - {label: "51～150 步", when: {stops_within: 150, not_stops_within: 50}}
      - {label: "150 步內停不下來", when: {not_stops_within: 150}}
    explain: "第 34 步停止，‖x − x*‖ = 6.0 × 10⁻¹⁰。看虛線的 z：第 1 步起 z₂ 就正好是 0（軟門檻把小於 l1/ρ = 3 的值直接變成 0）；實線的 x₂ 則在 0 的兩側來回，到最後還有 −4.1 × 10⁻¹⁰。稀疏性來自 z，x 只是在極限才等於 z。"
  - id: p2
    prompt: "換成「施肥預算（三角形）」（ρ = 1）。ADMM 的 x 迭代點（實線）會一直留在三角形內嗎？"
    scenario: {problem: fert_budget}
    show: [admm]
    options:
      - {label: "會，像投影梯度一樣每一步都可行", when: {stays_feasible: true}}
      - {label: "不會，有些 x 在三角形外", when: {stays_feasible: false}}
    explain: "不會：路徑上最大違反量 1.24，第 72 步停止時還有 1.3 × 10⁻⁹。x 的子問題完全不看限制，只被懲罰項拉向 z − u；z 才是每一步都投影回三角形的那一個（虛線）。x 與 z 的差就是原始殘差 r，它歸零時 x 才可行。"
  - id: p3
    prompt: "回到施肥稀疏，把 ρ 從 1 調到 100（懲罰 x ≠ z 更重）。會比 ρ = 1 的第 34 步更早停止嗎？"
    scenario: {params: {admm: {rho: 100.0}}}
    show: [admm]
    options:
      - {label: "更早：33 步內", when: {stops_within: 33}}
      - {label: "較晚：34～150 步", when: {stops_within: 150, not_stops_within: 33}}
      - {label: "150 步內停不下來", when: {not_stops_within: 150}}
    explain: "150 步內停不下來，距離還有 0.90（跑到 500 步是 6.2 × 10⁻⁴）。ρ 大時 x 和 z 幾乎黏在一起：從第 2 步起原始殘差 r 約 10⁻¹⁶，但對偶殘差 s 在第 10 步還是 1.6。軟門檻只有 l1/ρ = 0.03，z 每步只能挪一點點，乘數也就更新得很慢。"
  - id: p4
    prompt: "反過來把 ρ 調小到 0.1，迭代數 200。ADMM 會？"
    scenario: {params: {admm: {rho: 0.1}}, n_iter: 200}
    show: [admm]
    options:
      - {label: "比 ρ = 1 更早：33 步內停止", when: {stops_within: 33}}
      - {label: "34～200 步停止", when: {stops_within: 200, not_stops_within: 33}}
      - {label: "200 步內停不下來", when: {not_stops_within: 200}}
    explain: "第 167 步停止，比 ρ = 1 慢很多，但慢的原因和 ρ = 100 相反：軟門檻 l1/ρ = 30 很大，z 在第 1～20 步都卡在 (0, 0)，第 2～20 步的對偶殘差 s 都是 0，原始殘差 r 在第 20 步還有 1.4。x 幾乎不受 z 約束，要靠 u 一點一點累積才把 x、z 拉近。ρ 太大或太小都慢，要讓 r 和 s 一起下降。"
---

## 定義

**問題：** 把變數複製一份，拆成兩塊

$$
\min_{x,z}\ f(x)+g(z)\quad\text{s.t.}\quad x-z=0
$$

$f$ 平滑（本站是二次函數），$g$ 可以不平滑：可行集合的**指示函數**（在集合內為 0、外面為 $+\infty$），或 $\ell_1$ 範數 $\lambda\lVert z\rVert_1$。

**一句話：** 對這個拆開的問題寫出增廣 Lagrangian，但不同時對 $(x,z)$ 最小化，而是**先 $x$、再 $z$ 各最小化一次**，然後更新乘數。

**縮放形式**（$u$ = 乘數 / $\rho$）：

$$
\begin{aligned}
x_{k+1}&=\arg\min_x\ f(x)+\tfrac\rho2\lVert x-z_k+u_k\rVert^2\\
z_{k+1}&=\arg\min_z\ g(z)+\tfrac\rho2\lVert x_{k+1}-z+u_k\rVert^2=\operatorname{prox}_{g/\rho}(x_{k+1}+u_k)\\
u_{k+1}&=u_k+x_{k+1}-z_{k+1}
\end{aligned}
$$

| $g$ | $z$ 更新（prox） |
|---|---|
| 集合 $C$ 的指示函數 | 投影 $P_C(x+u)$，和[投影梯度法](projected-gradient.html)用的是同一個投影 |
| $\lambda\lVert z\rVert_1$ | **軟門檻**：每個座標往 0 收 $\lambda/\rho$，越過 0 就停在 0 |

**殘差與停止條件：** 原始殘差 $r_k=\lVert x_k-z_k\rVert$（離「$x=z$」多遠），對偶殘差 $s_k=\rho\lVert z_k-z_{k-1}\rVert$（$z$ 還在動多少）。本站兩者都小於 $10^{-9}$ 時停止；$z_0=x_0$、$u_0=0$，$\rho$ 固定。

**動畫的畫法：** 實線是 $x$，同色虛線與空心方塊是 $z$，收斂圖也各有一條。

<details markdown="1">
<summary>推導第 1 層：和增廣 Lagrangian 的關係</summary>

拆開的問題的增廣 Lagrangian（乘數 $y=\rho u$）：

$$
L_\rho(x,z,y)=f(x)+g(z)+y^\top(x-z)+\tfrac\rho2\lVert x-z\rVert^2
$$

[增廣 Lagrangian 法](augmented-lagrangian.html)每輪要對 $(x,z)$ **同時**最小化，再做 $y\leftarrow y+\rho(x-z)$。$f$ 與 $g$ 被二次項綁在一起，聯合最小化通常和原問題一樣難。

ADMM 只做**一次 Gauss–Seidel 掃描**：固定 $z$ 解 $x$、固定 $x$ 解 $z$，各自只碰一個函數，於是 $z$ 的子問題變成投影或軟門檻這種有公式的運算。乘數更新和增廣 Lagrangian 完全相同。代價是不再有「聯合最小化之後乘數線性收斂」的保證，通常需要更多迭代。

**不動點就是 KKT 點：** 若 $(x,z,u)$ 不再改變，$u$ 的更新給出 $x=z$；$x$ 的最佳性給出 $\nabla f(x)+\rho u=0$；$z$ 的最佳性給出 $\rho u\in\partial g(z)$。合起來是 $0\in\nabla f(x^*)+\partial g(x^*)$，也就是原問題的最佳性條件，而 $\rho u$ 就是乘數。

</details>

<details markdown="1">
<summary>推導第 2 層：軟門檻與施肥稀疏例子的 x*</summary>

一維的 $\min_z\ \lambda|z|+\frac\rho2(z-v)^2$：$z>0$ 時導數 $\lambda+\rho(z-v)=0$ 給出 $z=v-\lambda/\rho$，只在 $v>\lambda/\rho$ 時成立；負的一側對稱；其餘情形 $0\in\lambda[-1,1]-\rho v$，$z=0$。所以

$$
z=\operatorname{sign}(v)\max(|v|-\lambda/\rho,\ 0)
$$

**例子：** $s(x)=(x_1+x_2-3)^2+\tfrac12(x_1-2)^2$，$\lambda=3$。不加 $\ell_1$ 時最小值在 $(2,1)$。猜 $x_2=0$、$x_1>0$：

$$
\partial_{x_1}:\ 2(x_1-3)+(x_1-2)+3=0\Rightarrow x_1=\tfrac53
$$

再檢查 $x_2$ 的次梯度條件 $|\partial_{x_2}s|=|2(x_1-3)|=\tfrac83\le3$，成立，所以 $x^*=(5/3,\ 0)$。ADMM 結束時的 $\rho u=(3,\ 2.6667)$：第一個分量是 $\lambda\operatorname{sign}(x_1^*)=3$，第二個是 $-\partial_{x_2}s=8/3$，正好落在 $[-3,3]$ 之內。

</details>

<details markdown="1">
<summary>推導第 3 層：ρ 太大、太小各慢在哪裡</summary>

- $\rho$ 很大：$x$ 的子問題被 $\frac\rho2\lVert x-z+u\rVert^2$ 主導，$x\approx z$，$r$ 立刻變小；但 $z$ 的子問題也幾乎不准離開 $x$（軟門檻只有 $\lambda/\rho$），$z$ 每步只挪一點，$s$ 下降得慢。
- $\rho$ 很小：$x$ 幾乎只看 $f$，直接跑向不加限制的最小值；$z$ 被 $g$ 主導（軟門檻 $\lambda/\rho$ 很大，或投影把點拉得很遠），$r$ 很大，要靠 $u$ 慢慢累積才把兩者拉近。

實測（施肥稀疏，停止條件 $10^{-9}$）：

| ρ | 0.1 | 0.3 | 1 | 2 | 3 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| 停止步數 | 167 | 58 | 34 | 35 | 37 | 51 | 105 | 303 | 500 步未停（距離 6.2 × 10⁻⁴） |

實務上常在 $r\gg s$ 時放大 $\rho$、$s\gg r$ 時縮小 $\rho$（殘差平衡）；本站固定 $\rho$。

</details>

<details markdown="1">
<summary>類比：兩個測站分頭估計同一個量</summary>

兩個單位各自有自己的資料（$f$）與自己的規則（$g$，例如「用量不能是負的」或「能不用就不用」），要對同一組參數達成共識。每一輪各自算出最想要的值，再看彼此的差距（$r$），把差距記到帳上（$u$）；下一輪各自都會往對方靠一點。帳記得太重（$\rho$ 大），兩邊很快一致，但共識本身挪得很慢；記得太輕，兩邊各說各話很久。

</details>

## 應用場景

- **稀疏迴歸與訊號還原：** lasso、全變分去雜訊、壓縮感知，$z$ 的更新是軟門檻，$x$ 的更新是一個可以預先分解的線性方程組。
- **分散式最佳化：** 每台機器或每個子流域各自持有一份 $x_i$，共識限制 $x_i=z$；每台只解自己的子問題，再交換平均值。
- **能源系統：** 分區的電力調度、需量反應，各區自行最佳化，透過乘數協調聯絡線潮流。
- **影像與地球物理反演：** 平滑的資料擬合項加上不平滑的正則化（$\ell_1$、全變分、箱型限制），兩者交給不同的子問題。

## 解決什麼問題

[增廣 Lagrangian 法](augmented-lagrangian.html)每輪要解一個把所有項綁在一起的子問題；[投影梯度](projected-gradient.html)只能處理「平滑 + 投影便宜的集合」。ADMM 把目標拆成兩塊，各自用最方便的方式處理：

- 不平滑的 $g$（$\ell_1$、指示函數）只出現在 prox 裡，有公式解。
- 平滑的 $f$ 只出現在一個二次正則化的子問題裡，條件數被 $\rho I$ 改善。
- 拆成很多塊時可以平行計算，適合分散式資料。

代價是收斂通常只有線性速度、精度要求高時迭代數多，而且結果對 $\rho$ 很敏感（推導第 3 層）。

## 動畫演示

<!-- lab -->

步驟說明列出 $z$、乘數 $\rho u$、原始殘差 $r$ 與對偶殘差 $s$。

**建議的實驗：**

1. **預設（施肥稀疏，$\rho=1$）：** 第 34 步停止。前兩步的 $z$ 都是 $(0,0)$，之後 $z_2$ 一直是 0、只剩 $z_1$ 在動；實線的 $x_2$ 在 0 的兩側擺動（第 5 步 $-0.069$、第 15 步 $0.00025$）。殘差：第 10 步 $r=3.5\times10^{-3}$、$s=9.3\times10^{-3}$；第 20 步 $r=9.2\times10^{-6}$、$s=1.9\times10^{-6}$。
2. **調 ρ：** 推導第 3 層的表：$\rho=1$ 到 3 最快（34～37 步），$\rho=0.1$ 要 167 步、$\rho=30$ 要 303 步（把迭代數調到 500），$\rho=100$ 時 500 步還停不下來。看步驟說明：$\rho$ 小時 $r$ 大、$s$ 常是 0；$\rho$ 大時 $r$ 幾乎是 0、$s$ 降不下來。
3. **換成「施肥預算（三角形）」：** $\rho=1$ 第 72 步停止，$\rho=5$ 第 34 步。$z$ 永遠在三角形內，$x$ 從外面靠近（路徑上最大違反量 1.24）。結束時 $\rho u=(3,\ 3)$，就是預算限制的乘數 $\lambda^*=3$ 乘上它的梯度 $(1,1)$。
4. **勾選增廣 Lagrangian 與投影梯度比較（施肥預算）：** 增廣 Lagrangian（$\rho=10$）第 209 步、投影梯度（$\eta=0.1$）第 53 步。這個問題的投影很便宜，投影梯度比 ADMM（$\rho=1$ 時 72 步）快；ADMM 的優勢在 $g$ 不平滑或要分散計算時。
5. **換成「土壤養分圓盤」或「兩座水庫放水（箱型）」：** 圓盤上 $\rho=1$ 第 84 步、$\rho=5$ 第 32 步，$\rho u=(2.4773,\ 2.1098)$，等於 $\lambda^*\nabla g(x^*)$；水庫上 $\rho=1$ 第 63 步，$\rho u=(0,\ 1.3333)$，和手算的乘數 $4/3$ 一致。
6. **回到施肥稀疏並勾選比較：** 增廣 Lagrangian 與投影梯度的結果列顯示「不處理 ℓ1 項」：這兩個示範只看平滑部分的梯度，硬跑會收斂到錯的點 $(2,1)$，所以直接拒絕。

## 程式與練習

<!-- practice -->

## 失效情境

- **ρ 選得不好：** 太大或太小都會讓迭代數多好幾倍（實驗 2）；固定 $\rho$ 時需要試幾個值，或用殘差平衡自動調整。
- **高精度很貴：** 一般只有線性收斂；中等精度很快，要 $10^{-9}$ 就需要幾十到幾百步。
- **x 不可行：** 只有 $z$ 滿足限制，$x$ 只在極限才可行（預測題 2）；提早停止時要回報 $z$。
- **非凸問題：** 子問題可能有多個解，收斂沒有保證，結果依起點與 $\rho$ 而定。本站的 $x$ 更新用牛頓法，只對凸的 $f$ 可靠，所以沒有提供 Rosenbrock + 圓盤。
- **子問題不便宜：** 拆法的好處來自 prox 有公式；若 $g$ 的 prox 本身要迭代求解，ADMM 的每一步就不再便宜。

## 面試追問

<details markdown="1">
<summary>Q1. ADMM 和增廣 Lagrangian 法差在哪裡？</summary>

兩者的乘數更新相同。增廣 Lagrangian 法每輪對所有變數**聯合**最小化增廣 Lagrangian；ADMM 只做一次交替（Gauss–Seidel）：先固定 $z$ 解 $x$，再固定 $x$ 解 $z$。聯合最小化通常和原問題一樣難，交替之後每個子問題只碰一個函數，常常有公式解，代價是收斂變慢。

</details>

<details markdown="1">
<summary>Q2. 原始殘差和對偶殘差各代表什麼？為什麼兩個都要看？</summary>

原始殘差 $r=\lVert x-z\rVert$ 衡量限制 $x=z$ 違反多少；對偶殘差 $s=\rho\lVert z_k-z_{k-1}\rVert$ 衡量 $x$ 的最佳性條件還差多少（$z$ 還在動，乘數就還不對）。只看 $r$ 會在 $\rho$ 很大時誤判：$x$、$z$ 黏在一起但都還沒到 $x^*$。兩者都小，才接近 KKT 點。

</details>

<details markdown="1">
<summary>Q3. 為什麼 lasso 用 ADMM 時，z 是稀疏的而 x 不是？</summary>

$z$ 的更新是軟門檻，小於 $\lambda/\rho$ 的座標會被直接設成 0；$x$ 的更新是一個平滑的二次問題，解幾乎不會剛好是 0，只是在收斂時逼近 $z$。所以要稀疏的解應該回報 $z$。

</details>

<details markdown="1">
<summary>Q4. ρ 怎麼選？</summary>

$\rho$ 太大時 $r$ 很快變小但 $s$ 降得慢，太小時相反。常見做法是先試幾個數量級，或用殘差平衡：$r$ 遠大於 $s$ 就放大 $\rho$，反之縮小；改變 $\rho$ 時縮放形式的 $u$ 要跟著乘上 $\rho_{\text{舊}}/\rho_{\text{新}}$，乘數 $\rho u$ 才不變。

</details>

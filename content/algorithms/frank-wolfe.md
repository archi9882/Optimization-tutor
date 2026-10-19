---
review: draft
lab:
  problem: fert_budget
  problems: [fert_budget, reservoir, fert_disk, rosen_disk]
  algos: [frank_wolfe]
  compare: [projected_gradient]
  n_iter: 300
practice: frank_wolfe
predict:
  - id: p1
    prompt: "預設（施肥預算）：Frank-Wolfe 的迭代點會跑出三角形嗎？"
    scenario: {}
    show: [frank_wolfe]
    options:
      - {label: "每一步都留在三角形內", when: {stays_feasible: true}}
      - {label: "有些步會跑出三角形", when: {stays_feasible: false}}
    explain: "不會：x₊ = x + γ(s − x) 是 x 和頂點 s 的凸組合，0 ≤ γ ≤ 1，兩點都在凸集內，新點也一定在內。Frank-Wolfe 不需要投影，只需要在集合上解線性子問題。"
  - id: p2
    prompt: "300 步中，路徑有幾次折返（轉向超過 90°）？"
    scenario: {}
    show: [frank_wolfe]
    options:
      - {label: "10 次以下", when: {zigzag_max: 10}}
      - {label: "11～150 次", when: {zigzag_min: 11, zigzag_max: 150}}
      - {label: "151 次以上", when: {zigzag_min: 151}}
    explain: "299 次，幾乎每步都折返。x* = (1.5, 1.5) 在預算線的中間、不是頂點，線性子問題卻只會回傳頂點：它輪流選 (3, 0) 和 (0, 3)，迭代點就沿預算線來回拉扯。"
  - id: p3
    prompt: "100 步時離 x* 0.021。把迭代數加到 300（3 倍），最後離 x* 多遠？"
    scenario: {}
    show: [frank_wolfe, projected_gradient]
    options:
      - {label: "降到 0.0025 以下（比 1/k 快）", when: {final_dist_below: 0.0025}}
      - {label: "0.0025～0.014（比 k = 100 時小了 1.5～8 倍）", when: {final_dist_above: 0.0025, final_dist_below: 0.014}}
      - {label: "0.014 以上（幾乎沒進步）", when: {final_dist_above: 0.014}}
    explain: "0.0070，剛好約 1/3：步數 3 倍、距離縮成 1/3，是 O(1/k) 的速度。每步朝頂點走 γₖ = 2/(k + 2) 的比例，來回拉扯的幅度只按 1/k 縮小；同圖的投影梯度第 53 步就停止，距離 1.2 × 10⁻¹²。"
  - id: p4
    prompt: "換成「土壤養分圓盤」，同樣 300 步。最後離 x* 多遠？（三角形上是 0.0070）"
    scenario: {problem: fert_disk}
    show: [frank_wolfe]
    options:
      - {label: "10⁻⁴ 以下", when: {final_dist_below: 1.0e-4}}
      - {label: "10⁻⁴～10⁻²，和三角形差不多", when: {final_dist_above: 1.0e-4, final_dist_below: 1.0e-2}}
      - {label: "10⁻² 以上", when: {final_dist_above: 1.0e-2}}
    explain: "5.7 × 10⁻⁶，比三角形近上千倍，300 步只折返 3 次。圓盤的邊界是彎的，線性子問題的答案隨梯度連續移動，接近 x* 時 s 也接近 x*，不會在兩個頂點之間拉扯。"
---

## 定義

**問題：** $\min_{x\in C}f(x)$，$C$ 是有界的凸集合，而且**線性函數在 $C$ 上很好最小化**，例如多面體（答案在頂點）、圓盤、範數球。

**一句話：** 不做投影。每一步把 $f$ 線性化，在 $C$ 上找讓線性模型最小的點 $s_k$，再往 $s_k$ 走一小段。

$$
s_k=\arg\min_{s\in C}\nabla f(x_k)^\top s,\qquad
x_{k+1}=x_k+\gamma_k\,(s_k-x_k),\qquad \gamma_k=\frac{2}{k+2}
$$

找 $s_k$ 的子問題叫**線性最小化預言機**（LMO）。多邊形上只要把每個頂點代進 $\nabla f^\top v$ 比大小；圓盤上是從圓心沿 $-\nabla f$ 走到圓周。$x_{k+1}$ 是 $x_k$ 和 $s_k$ 的凸組合，所以**永遠可行**。本站在 Frank-Wolfe 間隙小於 $10^{-8}$ 時停止。

<details markdown="1">
<summary>推導第 1 層：Frank-Wolfe 間隙是免費的誤差上界</summary>

定義間隙 $\mathrm{gap}(x)=\nabla f(x)^\top(x-s)$，$s$ 是 LMO 的答案。$f$ 凸時，

$$
f^*=f(x^*)\ge f(x)+\nabla f(x)^\top(x^*-x)\ge f(x)+\min_{z\in C}\nabla f(x)^\top(z-x)=f(x)-\mathrm{gap}(x)
$$

所以 $f(x)-f^*\le\mathrm{gap}(x)$：不需要知道 $f^*$，每一步順便算出的間隙就是誤差的上界。$\mathrm{gap}(x)=0$ 時，沒有任何可行方向是下降方向，這正是有限制最優解的一階條件。

</details>

<details markdown="1">
<summary>推導第 2 層：為什麼是 O(1/k)</summary>

$L$-smooth 給出二次上界。令 $D$ 為 $C$ 的直徑，沿 $s_k-x_k$ 走 $\gamma$：

$$
f(x_{k+1})\le f(x_k)-\gamma\,\mathrm{gap}(x_k)+\frac{L D^2}{2}\gamma^2
$$

記 $e_k=f(x_k)-f^*\le\mathrm{gap}(x_k)$，得到 $e_{k+1}\le(1-\gamma_k)e_k+\tfrac{LD^2}{2}\gamma_k^2$。代入 $\gamma_k=2/(k+2)$ 用歸納法可證 $e_k\le\dfrac{2LD^2}{k+2}$（$k\ge1$）。步長不需要知道 $L$，但收斂只有次線性的 $O(1/k)$，比投影梯度在強凸問題上的線性收斂慢。

</details>

<details markdown="1">
<summary>為什麼最優解在邊上時會鋸齒狀前進</summary>

施肥例子的 $x^*=(1.5,\,1.5)$ 在預算線（三角形的一條邊）中間，但 LMO 只會回傳頂點 $(3,0)$ 或 $(0,3)$。要表示 $x^*$，迭代點只能在兩個頂點之間來回拉扯，每次拉的幅度是 $\gamma_k=2/(k+2)$，越來越小但總是過頭。實測到 $x^*$ 的距離：$k=10$ 時 0.193，$k=100$ 時 0.0210，$k=300$ 時 0.00705，$k=1000$ 時 0.00212，大約是 $2.1/k$。

</details>

## 應用場景

- **交通指派：** 用路人均衡問題的可行集合是路網流量的多面體，LMO 就是對每個起訖對找最短路徑，Frank-Wolfe 長期是這類問題的經典解法。
- **稀疏與低秩學習：** 在 $\ell_1$ 球上，LMO 只會回傳一個座標方向，$k$ 步後最多 $k+1$ 個非零係數；在核範數球上，LMO 是一個秩一矩陣（只需最大奇異向量），比投影需要的完整 SVD 便宜很多。
- **資源配置：** 肥料、灌溉水或預算在多個田區之間分配（單純形或運輸多面體），LMO 是「全部給目前邊際效益最高的那一塊」，每一步的方案都可行、而且容易解釋。

## 解決什麼問題

[投影梯度法](projected-gradient.html)每一步都要算投影，對一般多面體這是一個二次規劃。Frank-Wolfe 只需要**線性**子問題：

- 很多集合上線性最小化遠比投影便宜（最短路徑、最大奇異向量、挑一個頂點）。
- 迭代點是少數頂點的凸組合，天生稀疏或低秩。
- 永遠可行，而且間隙提供不需要 $f^*$ 的停止條件。

代價是：收斂只有 $O(1/k)$，最優解在面的中間時會鋸齒狀前進。加上「遠離步」（away step）等變體可以改善這一點。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. **預設（施肥預算）：** 細線連到這一步選中的頂點。第 1 步 $\gamma_0=1$，直接跳到頂點 $(0,3)$；之後 LMO 輪流回傳 $(3,0)$ 與 $(0,3)$，迭代點沿預算線來回：$(2,1)$、$(1,2)$、……。300 步後距離 0.0070，最後一步的間隙 0.060。收斂曲線是平滑下降的 $O(1/k)$，和投影梯度的直線（線性收斂）對比明顯。
2. **勾選投影梯度比較：** $\eta=0.1$ 時 53 步就停止，距離 $1.2\times10^{-12}$。在三角形這種投影容易的集合上，投影梯度快得多。
3. **點擊不同位置設成起點：** 因為 $\gamma_0=1$，第一步就把起點換成某個頂點；起點只透過「選哪個頂點」影響之後的路徑。
4. **換成「兩座水庫放水（箱型）」：** 第 21 步停止，距離剛好是 0。$\gamma_k=2/(k+2)$ 讓迭代點是頂點的有理數加權平均，這一題的 $x^*=(5/3,\,1)$ 恰好在第 20 個迭代點被精確命中，間隙變成 0。這是這組數字的巧合，不是 Frank-Wolfe 的一般性質。
5. **換成「土壤養分圓盤」：** 300 步後距離約 $5.7\times10^{-6}$（$k=100$ 時 $5.1\times10^{-5}$）。圓盤的邊界是彎的，LMO 的答案隨梯度連續移動，不會在兩個頂點之間拉扯，比多邊形快得多。
6. **換成「Rosenbrock + 圓盤（非凸）」：** 300 步後距離還有 0.46，1000 步後 0.22（1000 步超過動畫的迭代上限 500，需在程式中執行）。非凸又病態時 Frank-Wolfe 非常慢。

## 程式與練習

<!-- practice -->

## 失效情境

- **最優解在面的內部：** 只會回傳頂點的 LMO 讓迭代點鋸齒狀前進，收斂停在 $O(1/k)$（實驗 1）。
- **無界的集合：** LMO 可能沒有解（線性函數往無限遠下降），Frank-Wolfe 需要有界的 $C$。
- **高精度需求：** $O(1/k)$ 要多一位精度就要多約 10 倍步數；需要 $10^{-8}$ 的精度時通常改用其他方法。
- **非凸問題：** 間隙不再是 $f-f^*$ 的上界，只能保證找到一階駐點，而且可能很慢（實驗 6）。

## 面試追問

<details markdown="1">
<summary>Q1. Frank-Wolfe 為什麼不需要投影就能保持可行？</summary>

LMO 回傳的 $s_k$ 在 $C$ 裡，新的點 $x_{k+1}=(1-\gamma_k)x_k+\gamma_k s_k$ 是兩個可行點的凸組合，$C$ 是凸集合，所以 $x_{k+1}$ 也在 $C$ 裡。只要起點可行，每一步都可行。

</details>

<details markdown="1">
<summary>Q2. Frank-Wolfe 間隙是什麼？為什麼可以當停止條件？</summary>

$\mathrm{gap}(x)=\nabla f(x)^\top(x-s)$，$s$ 是線性子問題的解。由凸性 $f^*\ge f(x)+\nabla f(x)^\top(x^*-x)\ge f(x)-\mathrm{gap}(x)$，所以它是 $f(x)-f^*$ 的上界，而且每步都順便算出來，不需要知道 $f^*$。

</details>

<details markdown="1">
<summary>Q3. 什麼情況下 Frank-Wolfe 比投影梯度好？</summary>

當線性最小化比投影便宜很多時：例如核範數球上 LMO 只需要最大奇異向量，投影卻需要完整 SVD；交通網路上 LMO 是最短路徑，投影卻是大型二次規劃。另外，迭代點是少數頂點的組合，結果天生稀疏或低秩，容易解釋。

</details>

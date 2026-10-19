---
review: draft
lab:
  problem: fert_budget
  problems: [fert_budget, fert_eq, fert_disk, reservoir, rosen_disk]
  algos: [augmented_lagrangian]
  compare: [penalty]
  n_iter: 400
practice: augmented_lagrangian
predict:
  - id: p1
    prompt: "預設（施肥預算，ρ = 10 固定，不放大）：增廣 Lagrangian 停止時離 x* 多遠？（懲罰法要把 ρ 放大到 10⁴，還停在 0.0050）"
    scenario: {}
    show: [augmented_lagrangian, penalty]
    options:
      - {label: "不到 10⁻⁶", when: {final_dist_below: 1.0e-6}}
      - {label: "10⁻⁶～10⁻²，和懲罰法差不多", when: {final_dist_above: 1.0e-6, final_dist_below: 1.0e-2}}
      - {label: "10⁻² 以上", when: {final_dist_above: 1.0e-2}}
    explain: "第 209 步停止，距離 2.3 × 10⁻⁷、違反量 9.9 × 10⁻⁸。乘數 λ 每輪更新後往 λ* = 3 收斂，等於替懲罰項補上「該推多少」的資訊，所以不需要 ρ → ∞；子問題維持溫和的條件數。"
  - id: p2
    prompt: "把 ρ 從 10 提高到 100。會比 ρ = 10 的第 209 步更早停止嗎？"
    scenario: {params: {augmented_lagrangian: {rho: 100.0}}}
    show: [augmented_lagrangian]
    options:
      - {label: "更早：208 步內停止", when: {stops_within: 208}}
      - {label: "更晚：209～400 步才停止", when: {stops_within: 400, not_stops_within: 208}}
      - {label: "400 步內停不下來", when: {not_stops_within: 400}}
    explain: "第 367 步才停止，比 ρ = 10 晚：乘數更新只要 5 次（ρ = 10 時 8 次），但 ρ 大，增廣 Lagrangian 的等高線更扁，每輪 GD 內迴圈要走更多步。ρ 越大不一定越快。"
  - id: p3
    prompt: "再把 ρ 提高到 1000。400 步時離 x* 多遠？"
    scenario: {params: {augmented_lagrangian: {rho: 1000.0}}}
    show: [augmented_lagrangian]
    options:
      - {label: "不到 10⁻⁶", when: {final_dist_below: 1.0e-6}}
      - {label: "10⁻⁶～10⁻²", when: {final_dist_above: 1.0e-6, final_dist_below: 1.0e-2}}
      - {label: "10⁻² 以上", when: {final_dist_above: 1.0e-2}}
    explain: "400 步時還沒停止，距離 0.0595，只完成 3 次乘數更新。ρ = 1000 的子問題和懲罰法的大 ρ 一樣病態，GD 內迴圈每輪都用完 100 步的上限；增廣 Lagrangian 的好處正是用適中的 ρ 就能收斂。"
  - id: p4
    prompt: "回到 ρ = 10，把內圈步數上限從 100 降到 5：每輪 GD 最多 5 步就強制更新乘數，迭代數 500。會怎樣？"
    scenario: {params: {augmented_lagrangian: {rho: 10.0, inner: 5}}, n_iter: 500}
    show: [augmented_lagrangian]
    options:
      - {label: "比預設更早：208 步內停止", when: {stops_within: 208}}
      - {label: "比預設更晚：209～500 步才停止", when: {stops_within: 500, not_stops_within: 208}}
      - {label: "500 步內停不下來", when: {not_stops_within: 500}}
    explain: "第 73 步就停止（13 次乘數更新），距離 1.9 × 10⁻⁷，比預設的 209 步還快。乘數更新不必等子問題解得很準；子問題粗略地解、乘數早點更新，常常更省步數。"
---

## 定義

**一句話：** 懲罰法 + 乘數估計。每輪先在固定乘數 $\lambda$ 下近似最小化「增廣 Lagrangian」，再用限制的違反量修正 $\lambda$。乘數對了，有限的懲罰係數 $\rho$ 就能得到精確可行的解。又叫**乘數法**（method of multipliers）。

**等式限制** $h(x)=0$：

$$
L_A(x;\nu,\rho)=f(x)+\nu\,h(x)+\frac{\rho}{2}h(x)^2,\qquad
\nu\leftarrow\nu+\rho\,h(x)
$$

**不等式限制** $g(x)\le0$（本站使用的寫法，等式是它的特例）：令 $s=\max\big(0,\ \lambda+\rho\,g(x)\big)$，

$$
L_A(x;\lambda,\rho)=f(x)+\frac{s^2-\lambda^2}{2\rho},\qquad
\nabla_xL_A=\nabla f(x)+s\,\nabla g(x),\qquad \lambda\leftarrow s
$$

多個限制時各自加總、各自更新。本站的 $\rho$ 固定不變，內迴圈是帶 Armijo 回溯的梯度下降，$\lVert\nabla L_A\rVert<10^{-6}$ 時更新乘數，乘數的變化小於 $10^{-6}$ 時停止。

<details markdown="1">
<summary>推導第 1 層：為什麼乘數更新是 λ ← λ + ρh</summary>

$L_A$ 的駐點滿足

$$
\nabla f(x)+\big(\nu+\rho\,h(x)\big)\nabla h(x)=0
$$

和 KKT 的穩定性條件 $\nabla f+\nu^*\nabla h=0$ 比較，$\nu+\rho\,h(x)$ 就是目前對 $\nu^*$ 最好的估計，所以拿它當下一輪的乘數。

**關鍵差別：** 若 $\nu=\nu^*$，$x^*$ 本身就是 $L_A$ 的駐點（因為 $h(x^*)=0$，懲罰項的梯度為 0），**不論 ρ 多大都成立**。懲罰法沒有 $\nu$，只能靠 $\rho\,h\to\nu^*$，於是 $h$ 必須約為 $\nu^*/\rho$，永遠不為 0。

</details>

<details markdown="1">
<summary>推導第 2 層：施肥例子的乘數收斂率</summary>

預算限制 $a^\top x\le3$，$a=(1,1)$，$f$ 的 Hessian $H=\mathrm{diag}(2,6)$，無限制最小值 $(3,2)$ 的違反量 $g_u=2$。內迴圈若精確求解，限制起作用時

$$
x=x_u-H^{-1}a\,s,\qquad g(x)=g_u-(a^\top H^{-1}a)\,s=2-\tfrac23 s,\qquad s=\lambda+\rho\,g(x)
$$

解出 $s=\dfrac{\lambda+2\rho}{1+2\rho/3}$，所以

$$
\lambda_{k+1}-3=\frac{\lambda_k-3}{1+2\rho/3}
$$

固定點正是 $\lambda^*=3$，誤差每輪乘上 $1/(1+2\rho/3)$：$\rho=10$ 時 $3/23=0.1304$。實測前三個誤差比（第 2、3、4 次更新相對於前一次）都是 0.1304；之後內迴圈只解到 $10^{-6}$ 的誤差開始顯現，比值在 0.089 到 0.26 之間浮動。**ρ 越大收斂越快，但內迴圈越病態**，兩者要取捨。

另外，$\lambda=0$ 的第一輪就是 $\rho=10$ 的懲罰法：第一次更新得到 $\lambda=2.6087$，正是懲罰解的 $\rho\,g(x_\rho)=6\rho/(3+2\rho)=60/23$。

</details>

<details markdown="1">
<summary>不等式限制的寫法從哪來？</summary>

把 $g(x)\le0$ 改寫成 $g(x)+z=0$、$z\ge0$，對等式套用增廣 Lagrangian，再對 $z$ 解析地最小化（$z$ 只出現在二次式裡）。結果是 $z=\max(0,\,-\lambda/\rho-g)$，代回去整理就得到 $\frac{1}{2\rho}\big(\max(0,\lambda+\rho g)^2-\lambda^2\big)$。$\max(0,\cdot)$ 自動實現互補鬆弛：限制沒被碰到時 $s=0$，乘數歸零。

</details>

## 應用場景

- **有限元素與結構力學：** 接觸問題、不可壓縮條件常以增廣 Lagrangian 處理，避免純懲罰法需要極大的懲罰係數。
- **地球物理反演：** 需要嚴格滿足的資料或物理限制（例如質量守恆、已知井位的參數值）可以用乘數法加入，不必把權重調到無限大。
- **PINN 的硬限制訓練：** 以乘數更新取代手調的損失權重，讓邊界條件或守恆律在訓練過程中逐步被精確滿足。
- **分散式最佳化：** [ADMM](admm.html) 就是把增廣 Lagrangian 拆成兩塊交替最小化。

## 解決什麼問題

[懲罰法](penalty.html)的兩難：精度要求 $\rho\to\infty$，但 $\rho$ 越大子問題越病態。增廣 Lagrangian 法讓乘數 $\lambda$ 承擔「推回可行域」的工作：

- $\rho$ 固定在適中的值，子問題的條件數不會一路惡化。
- 乘數收斂後解是**精確可行**的，而且得到 $\lambda^*$（每個限制的影子價格）。
- 等式與不等式限制都適用，起點不必可行。

代價是多了一層外迴圈，$\rho$ 仍需要選：太小時乘數收斂慢，太大時又回到病態。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. **預設（施肥預算，$\rho=10$）：** 第 209 步停止，共 8 次乘數更新，距離 $2.3\times10^{-7}$、違反量 $9.9\times10^{-8}$。看步驟說明裡的 $\lambda$：2.609、2.949、2.993、2.999，往 $\lambda^*=3$ 收斂，誤差每次約縮小為 0.13 倍。每輪需要的內迴圈步數也從 45 步遞減到 6 步，因為上一輪的解已經很接近。
2. **勾選懲罰函數法比較：** 懲罰法要把 $\rho$ 放大到 $10^4$，第 367 步停止時距離還有 0.0050；增廣 Lagrangian 只用 $\rho=10$。
3. **ρ 的取捨：** $\rho=1$ 時要 29 次更新、1085 步才停止（400 步時還沒到）；$\rho=100$ 時 5 次更新、367 步，比 $\rho=10$ 的 209 步還多，因為每輪的內迴圈更病態。$\rho=1000$ 時 500 步內距離只到 0.038。
4. **換成「兩座水庫放水（箱型）」：** 第 196 步停止，$\lambda\approx(0,\ 1.3333,\ 0,\ 0)$。只有 $x_2\le1$ 起作用，其他三個限制的乘數保持 0，這就是互補鬆弛；手算值是 $4/3$。
5. **換成「預算全部用完（等式）」：** 第 208 步停止，乘數同樣收斂到 3。等式限制的 $s=\lambda+\rho h$ 不截斷，迭代點可以從線的兩側靠近。
6. **換成「Rosenbrock + 圓盤（非凸）」並把迭代數調到 500：** $\rho=10$ 時 500 步距離 $1.9\times10^{-6}$（完整跑完要 547 步）；$\rho=100$ 時第 346 步停止，距離 $1.1\times10^{-9}$。前兩輪內圈都用完 100 步，迭代點還在圓內，乘數一直是 0。

## 程式與練習

<!-- practice -->

## 失效情境

- **ρ 太小：** 乘數更新每次只縮小 $1/(1+c\rho)$ 倍左右的誤差，外迴圈很多（實驗 3）。
- **ρ 太大：** 子問題又變得病態，一階的內迴圈步數暴增（實驗 3）；實務上常從小 $\rho$ 開始，只在違反量下降不夠快時才放大。
- **ρ 太大又搭配太少的內圈步數會發散：** 「預算全部用完（等式）」、$\rho=1000$、內圈上限 5 時，每次乘數更新用的是遠未收斂的 $x$，$\lambda$ 依序變成 1611、−2576、5425、−10755……正負交替、每次約翻倍，第 121 步發散。同樣設定在「施肥預算」上不會發散，但 50 次更新後仍未收斂，第 300 步因達到外圈上限停止，距離 0.094（結果列會顯示「達到外圈上限」）。
- **內迴圈解得太粗：** 乘數更新用的是不準的 $x$，收斂率會偏離理論值，甚至停滯。
- **非凸問題：** 只保證收斂到 KKT 點；若乘數初值很差，內迴圈可能落入不可行的局部最小值。

## 面試追問

<details markdown="1">
<summary>Q1. 為什麼增廣 Lagrangian 法不需要 ρ → ∞？</summary>

$L_A=f+\nu h+\frac\rho2h^2$ 在 $\nu=\nu^*$ 時，$x^*$ 已經是它的駐點，因為 $h(x^*)=0$ 讓懲罰項的梯度為 0，這對任何 $\rho$ 都成立。演算法只要讓乘數收斂到 $\nu^*$，有限的 $\rho$ 就足夠；懲罰法沒有乘數，只能靠 $\rho h\to\nu^*$，所以 $h$ 永遠約為 $\nu^*/\rho$。

</details>

<details markdown="1">
<summary>Q2. 乘數更新 λ ← λ + ρ·h(x) 是怎麼來的？</summary>

$L_A$ 的駐點滿足 $\nabla f+(\nu+\rho h)\nabla h=0$，和 KKT 條件 $\nabla f+\nu^*\nabla h=0$ 對照，$\nu+\rho h(x)$ 就是對 $\nu^*$ 的新估計。另一個看法：它是對偶函數上步長為 $\rho$ 的梯度上升，因為對偶函數對 $\nu$ 的梯度就是 $h(x)$。

</details>

<details markdown="1">
<summary>Q3. 不等式限制在增廣 Lagrangian 裡怎麼處理？</summary>

加入鬆弛變數 $g+z=0$、$z\ge0$，對 $z$ 解析地最小化後，得到 $\frac{1}{2\rho}(\max(0,\lambda+\rho g)^2-\lambda^2)$，乘數更新是 $\lambda\leftarrow\max(0,\lambda+\rho g)$。截斷在 0 保證 $\lambda\ge0$，而且限制沒被碰到時乘數自動歸零，滿足互補鬆弛。

</details>

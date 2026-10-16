---
review: draft
lab:
  problem: rosenbrock
  algos: [gd, bfgs]
  compare: [newton, lbfgs]
  params:
    gd: {lr: 0.002}
  n_iter: 100
practice: bfgs
---

## 定義

**想法：** 不計算 Hessian，而是在每一步用「梯度變了多少」來修正一個 Hessian 反矩陣的近似 $H_k$。

**更新規則：**

$$
d_k=-H_k\nabla f(x_k),\qquad x_{k+1}=x_k+\alpha_k d_k\quad(\alpha_k\text{ 由線搜尋決定})
$$

令 $s_k=x_{k+1}-x_k$、$y_k=\nabla f(x_{k+1})-\nabla f(x_k)$、$\rho_k=1/(y_k^\top s_k)$，更新

$$
H_{k+1}=(I-\rho_k s_k y_k^\top)\,H_k\,(I-\rho_k y_k s_k^\top)+\rho_k\,s_k s_k^\top
$$

起點通常取 $H_0=I$。

<details markdown="1">
<summary>推導：割線條件是什麼？</summary>

對二次函數，梯度差恰好等於 Hessian 乘位移：$y=\nabla^2f\,s$。所以 Hessian 反矩陣應該把 $y$ 映回 $s$：

$$
H_{k+1}\,y_k=s_k
$$

這叫**割線條件**，是一維割線法的多維推廣。在 $n$ 維時，這個條件只有 $n$ 個方程，無法唯一決定 $n^2$ 個元素。BFGS 選擇「滿足割線條件、保持對稱，而且和 $H_k$ 差距最小（在某個加權範數下）」的那一個，就得到上面的公式。

</details>

<details markdown="1">
<summary>推導：為什麼要求 sᵀy > 0？</summary>

若 $H_k$ 正定且 $s_k^\top y_k>0$，更新後的 $H_{k+1}$ 也正定，方向 $-H_{k+1}\nabla f$ 就一定是下降方向。$s^\top y>0$ 代表沿著 $s$ 的平均曲率為正。Wolfe 線搜尋的曲率條件可以保證這一點；本站用較簡單的 Armijo 回溯，所以在 $s^\top y\le0$ 時直接跳過更新。

</details>

**收斂性：** 在合適的條件下（強凸、Wolfe 線搜尋），BFGS 具有**超線性收斂**：$\lVert e_{k+1}\rVert/\lVert e_k\rVert\to0$，比任何線性收斂都快，但通常不到二次。

**成本：** 每步只需要一次梯度，更新 $H$ 是 $O(n^2)$，不需要解線性系統。

## 應用場景

- **中小型平滑最佳化：** 參數數量從幾個到幾千個時，BFGS 是許多科學計算套件的預設方法之一。
- **統計模型的最大概似估計：** 只能算梯度、算不出 Hessian 時的首選。
- **工程設計最佳化：** 目標函數由模擬程式計算、梯度由伴隨法提供時。

## 解決什麼問題

牛頓法收斂快，但每步要算 Hessian 並解線性系統；梯度下降只用梯度，卻被條件數拖慢。

BFGS 介於兩者之間：

- **只用梯度**，不需要二階導數。
- 透過累積的 $(s,y)$ 學到曲率，**超線性收斂**。
- $H$ 保持正定，方向永遠是下降方向（牛頓法做不到）。

缺點是 $H$ 是 $n\times n$ 稠密矩陣，$n$ 很大時存不下，這導向 [L-BFGS](lbfgs.html)。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 預設的 Rosenbrock：BFGS 第 36 步達到 $f<10^{-10}$，GD（$\eta=0.002$）100 步後仍有 $1.3\times10^{-3}$。注意 BFGS 前幾步很像梯度下降（$H_0=I$），之後越走越準。
2. 打開牛頓法比較：牛頓法 6 步，BFGS 36 步。BFGS 沒有 Hessian 資訊，要花一些步數「學」曲率，但每一步便宜得多。
3. 換到二次碗（$\kappa=10$）：BFGS 第 4 步就達到 $f<10^{-10}$。
4. 換到 Himmelblau：從預設起點，牛頓法收斂到最大值，BFGS 則正確地收斂到最小值 $(-2.81,3.13)$，因為它的 $H$ 保持正定。

## 程式與練習

<!-- practice -->

## 失效情境

- **高維問題：** $H$ 需要 $n^2$ 記憶體，$n=10^6$ 時需要約 8 TB，不可行。
- **線搜尋不當：** 若不滿足 $s^\top y>0$，$H$ 可能失去正定性，方向不再下降。
- **非平滑函數：** 割線條件基於梯度連續，在不可微點附近曲率估計會失準。
- **雜訊梯度：** 隨機梯度讓 $y$ 充滿雜訊，曲率估計不可靠，所以 BFGS 很少直接用在小批次訓練上。

## 面試追問

<details markdown="1">
<summary>Q1. BFGS 怎麼在不算 Hessian 的情況下得到曲率資訊？</summary>

每一步記錄位移 $s$ 和梯度變化 $y$，要求新的近似滿足割線條件 $H y=s$。二次函數上 $y=\nabla^2 f\,s$ 精確成立，所以累積幾步後，$H$ 就逼近真正的 Hessian 反矩陣。

</details>

<details markdown="1">
<summary>Q2. 為什麼 BFGS 不會像牛頓法一樣跑到最大值？</summary>

只要 $s^\top y>0$，更新保持 $H$ 正定，方向 $-H\nabla f$ 的方向導數 $-\nabla f^\top H\nabla f<0$，一定是下降方向。牛頓法的 Hessian 在非凸區域可能不正定，方向就可能上升。

</details>

<details markdown="1">
<summary>Q3. BFGS、牛頓法、梯度下降的每步成本與收斂速度怎麼比？</summary>

梯度下降每步 $O(n)$、線性收斂；BFGS 每步 $O(n^2)$、超線性收斂；牛頓法每步要算 Hessian 並解 $O(n^3)$ 的系統、二次收斂。BFGS 通常是中小型問題中總成本最低的選擇。

</details>

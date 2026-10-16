---
review: draft
lab:
  problem: rosenbrock
  algos: [newton, gauss_newton]
  compare: [lm]
  params:
    newton: {alpha: 1.0}
    gauss_newton: {alpha: 1.0}
  n_iter: 60
practice: gauss_newton
---

## 定義

**問題形式（非線性最小平方）：**

$$
\min_x\ f(x)=\sum_{i=1}^m r_i(x)^2=\lVert r(x)\rVert^2
$$

$r_i$ 是殘差，例如「模式預測值 − 觀測值」。令 $J$ 為殘差的 Jacobian，$J_{ki}=\partial r_k/\partial x_i$。

**更新規則：** 把殘差線性化 $r(x+d)\approx r+Jd$，求讓 $\lVert r+Jd\rVert^2$ 最小的 $d$，得到**正規方程**：

$$
(J^\top J)\,d=-J^\top r,\qquad x_{k+1}=x_k+\alpha\,d
$$

<details markdown="1">
<summary>推導：高斯-牛頓就是「丟掉一項」的牛頓法</summary>

$f=\sum r_i^2$ 的梯度與 Hessian 為

$$
\nabla f=2J^\top r,\qquad \nabla^2 f=2J^\top J+2\sum_i r_i\,\nabla^2 r_i
$$

高斯-牛頓直接丟掉第二項，用 $2J^\top J$ 近似 Hessian。代入牛頓方程 $\nabla^2f\,d=-\nabla f$，就得到 $(J^\top J)d=-J^\top r$。

丟掉的那一項在兩種情況下很小：**殘差很小**（模型能解釋資料），或**殘差接近線性**（$\nabla^2 r_i\approx 0$）。這時高斯-牛頓的表現接近牛頓法；殘差在解處為零時，局部收斂是二次的。

</details>

<details markdown="1">
<summary>性質：方向一定是下降方向（只要 J 滿秩）</summary>

$J^\top J$ 一定半正定，$J$ 滿秩時正定。方向導數

$$
\nabla f^\top d=2r^\top J\,d=-2\,r^\top J(J^\top J)^{-1}J^\top r\le 0
$$

所以不像牛頓法，高斯-牛頓不會被吸到最大值。但 $J$ 不滿秩時，$J^\top J$ 奇異，方向就不存在。

</details>

**成本：** 只需要一階導數（Jacobian），不需要任何二階導數。

## 應用場景

- **地震定位：** 由多個測站的到時反推震源位置與發震時刻，經典的 Geiger 法就是高斯-牛頓迭代。
- **曲線擬合與參數校準：** 例如用觀測水位擬合含水層參數、擬合衰減曲線或入滲模型參數。
- **衛星定位與大地測量：** 由多顆衛星的距離觀測解出接收器座標。
- **電腦視覺：** 相機校正與光束法平差（bundle adjustment）。

## 解決什麼問題

牛頓法需要二階導數，對複雜的物理模式往往算不出來。但很多問題本質上是「讓模型和資料的差距最小」，具有最小平方的結構。

高斯-牛頓利用這個結構，**只用一階導數就得到接近牛頓法的曲率資訊**：

- $J^\top J$ 半正定，方向不會指向最大值。
- 殘差小的問題上，收斂速度接近牛頓法。

缺點是殘差大或高度非線性時近似失準，而且 $J$ 不滿秩時方向不存在。[Levenberg-Marquardt](levenberg-marquardt.html) 在 $J^\top J$ 加上 $\lambda I$ 同時解決這兩個問題。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 預設的 Rosenbrock（可寫成 $r=(1-x,\ 10(y-x^2))$，解處殘差為零）：高斯-牛頓 **2 步**就到 $(1,1)$，牛頓法要 6 步。但注意第 1 步跳到圖外的 $(1,-5.25)$，那裡 $f=3910$，比起點大很多。
2. 換到 Himmelblau：從預設起點 $(-0.5,-0.5)$，高斯-牛頓**完全不動**。那裡的 Jacobian 是 $\begin{pmatrix}-1&1\\1&-1\end{pmatrix}$，行列式為 0，正規方程無解。
3. 承上，點擊圖面把起點改到 $(1,1)$ 附近：高斯-牛頓 5 步收斂到 $(3,2)$。
4. 打開 LM，回到預設起點：$\lambda I$ 讓矩陣可逆，LM 能正常前進。

## 程式與練習

<!-- practice -->

## 失效情境

- **Jacobian 不滿秩：** $J^\top J$ 奇異，無法求方向（實驗 2）。參數彼此無法區分時就會發生，例如兩個參數只以乘積形式出現。
- **大殘差問題：** 被丟掉的 $\sum r_i\nabla^2 r_i$ 很大時，近似失準，可能只有線性收斂甚至發散。
- **步子太大：** 純高斯-牛頓不保證每步下降（實驗 1 中間經過 $f=3910$）。實務上要加線搜尋或改用 LM。

## 面試追問

<details markdown="1">
<summary>Q1. 高斯-牛頓和牛頓法差在哪裡？什麼時候差別很小？</summary>

高斯-牛頓用 $2J^\top J$ 近似 Hessian，丟掉 $2\sum r_i\nabla^2 r_i$。殘差很小（模型能解釋資料）或殘差接近線性時，丟掉的項很小，兩者表現接近；殘差大時差別就明顯。

</details>

<details markdown="1">
<summary>Q2. 線性最小平方問題上，高斯-牛頓要幾步？</summary>

一步。殘差是線性的，$r(x+d)=r+Jd$ 完全精確，正規方程的解就是全域最小值。

</details>

<details markdown="1">
<summary>Q3. 參數校準時 J^⊤J 接近奇異，代表什麼？</summary>

代表有些參數組合幾乎不影響殘差，資料無法區分它們（不可辨識或高度相關）。數值上會出現很大的步長與不穩定；可以加正則化、固定部分參數，或用 LM 的阻尼項。

</details>

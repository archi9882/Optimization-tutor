---
review: draft
lab:
  problem: quadratic
  problem_params: {kappa: 50}
  algos: [gd, momentum]
  compare: [nesterov, adam]
  params:
    gd: {lr: 0.035}
    momentum: {lr: 0.02, beta: 0.8}
    nesterov: {lr: 0.02, beta: 0.8}
  n_iter: 150
practice: momentum
---

## 定義

**更新規則（Polyak heavy ball）：**

$$
x_{k+1}=x_k-\eta\,\nabla f(x_k)+\beta\,(x_k-x_{k-1})
$$

比梯度下降多了一項 $\beta(x_k-x_{k-1})$：**上一步的位移，打個折扣後再走一次**。$\beta\in[0,1)$ 稱為動量係數。

**深度學習框架的寫法（PyTorch 慣例）：** 引入速度 $v$，

$$
v_{k+1}=\beta\,v_k+\nabla f(x_k),\qquad x_{k+1}=x_k-\eta\,v_{k+1}
$$

<details markdown="1">
<summary>證明兩種寫法等價</summary>

由第二式，$x_k-x_{k-1}=-\eta\,v_k$。代入第一式：

$$
x_{k+1}=x_k-\eta(\beta v_k+\nabla f(x_k))=x_k-\eta\nabla f(x_k)+\beta\,(x_k-x_{k-1})
$$

就是 Polyak 形式。差別只在記錄的是「速度」還是「上一步位移」。

</details>

**物理圖像：** 有摩擦的球在地形上滾動，$m\ddot x+c\,\dot x=-\nabla f(x)$。把它離散化就得到 heavy ball，$\beta$ 越接近 1 代表摩擦越小、慣性越大。

**在二次函數上的最佳參數與收斂率：** 若 $\mu\le\lambda(A)\le L$，取

$$
\eta=\frac{4}{(\sqrt L+\sqrt\mu)^2},\qquad \beta=\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2
$$

收斂率為 $\dfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$。和 GD 的 $\dfrac{\kappa-1}{\kappa+1}$ 相比，**$\kappa$ 換成了 $\sqrt\kappa$**，迭代數從 $O(\kappa)$ 降到 $O(\sqrt\kappa)$。

<details markdown="1">
<summary>推導：把遞迴寫成 2×2 矩陣</summary>

沿 $A$ 的某個特徵值 $\lambda$ 的方向，誤差 $e_k$ 滿足二階線性遞迴：

$$
e_{k+1}=(1+\beta-\eta\lambda)\,e_k-\beta\,e_{k-1}
$$

特徵方程為 $r^2-(1+\beta-\eta\lambda)\,r+\beta=0$。當判別式為負（兩根共軛複數），兩根的模都是 $\sqrt\beta$，與 $\lambda$ 無關。上面那組 $\eta,\beta$ 恰好讓 $\lambda=\mu$ 與 $\lambda=L$ 兩端都落在判別式為零的邊界，所有特徵方向都以 $\sqrt\beta=(\sqrt\kappa-1)/(\sqrt\kappa+1)$ 的速率衰減。

這就是為什麼動量的軌跡會呈**螺旋狀**：複數根代表旋轉加衰減。

</details>

**實測（本站 `optlab`，二次碗 $\kappa=100$，起點 $(-4.5,2)$，到 $f<10^{-8}$）：** 最佳步長的 GD 需要 595 步，最佳參數的 heavy ball 需要 85 步。

## 應用場景

- **深度學習：** SGD + momentum（$\beta=0.9$ 是常見預設）至今仍是訓練影像模型的主流選擇之一。
- **病態的最小平方問題：** 例如參數尺度差異很大的迴歸或反演問題，動量可以明顯減少迭代數。
- **作為其他方法的零件：** Adam 的一階動差本質上就是動量的指數移動平均。

## 解決什麼問題

梯度下降在窄長峽谷中，梯度幾乎垂直於谷的走向：在陡峭方向來回彈跳，在平緩方向只前進一點點。

動量把連續幾步的梯度累加起來：

- **陡峭方向**的梯度一正一負，在 $v$ 裡互相**抵銷** → 震盪被抑制。
- **平緩方向**的梯度一直同號，在 $v$ 裡**累積** → 等效步長放大約 $1/(1-\beta)$ 倍。

結果是用幾乎相同的計算成本（每步仍只算一次梯度），把條件數的影響從 $\kappa$ 降到 $\sqrt\kappa$。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 預設的 $\kappa=50$ 下比較 GD 與 Momentum：注意 GD 的鋸齒和動量的螺旋。GD 的 $\eta$ 已經接近上限 $2/L=0.04$，再調大就發散。
2. 把 $\beta$ 調到 0.97：慣性太大，會衝過頭並繞很多圈才停下。
3. 把 $\beta$ 調到 0，並把 $\eta$ 調成和 GD 一樣：兩條軌跡完全重合。
4. 打開 Nesterov：它在轉彎前「先看一步」，過衝通常比較小。

## 程式與練習

<!-- practice -->

## 失效情境

- **$\beta$ 太大：** 阻尼不足，過衝後長時間振盪，甚至發散。
- **一般強凸函數（非二次）：** 上面那組最佳參數只對二次函數推導出來；對一般函數，heavy ball 不保證有同樣的加速，要有一般性的加速保證需改用 Nesterov 法。
- **非凸地形：** 慣性可能讓它衝過想要的谷。反過來說，這也能幫助它越過淺的局部最小值。

## 面試追問

<details markdown="1">
<summary>Q1. 動量為什麼能減少震盪？用一句話說。</summary>

在高曲率方向，連續幾步的梯度正負交替，在速度裡互相抵銷；在低曲率方向，梯度同號而累積，所以震盪變小、前進變快。

</details>

<details markdown="1">
<summary>Q2. β = 0.9 時，平緩方向的等效步長大約放大多少倍？</summary>

梯度近似固定為 $g$ 時，$v$ 收斂到 $g/(1-\beta)$，所以放大約 $1/(1-0.9)=10$ 倍。

</details>

<details markdown="1">
<summary>Q3. 為什麼動量的軌跡會是螺旋形？</summary>

誤差滿足二階線性遞迴，參數合適時特徵根是共軛複數。複數根的模代表衰減，輻角代表旋轉，合起來就是螺旋。

</details>

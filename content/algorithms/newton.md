---
review: draft
lab:
  problem: rosenbrock
  algos: [gd, newton]
  compare: [bfgs, trust_region]
  params:
    gd: {lr: 0.002}
    newton: {alpha: 1.0}
  n_iter: 100
practice: newton
---

## 定義

**想法：** 在目前位置用二階 Taylor 展開建立二次模型，然後直接跳到這個模型的最低點。

$$
m_k(d)=f(x_k)+\nabla f(x_k)^\top d+\tfrac12\,d^\top \nabla^2 f(x_k)\,d
$$

令 $\nabla m_k(d)=\nabla f(x_k)+\nabla^2 f(x_k)\,d=0$，得到**牛頓方向**與更新規則：

$$
\nabla^2 f(x_k)\,d_k=-\nabla f(x_k),\qquad x_{k+1}=x_k+\alpha\,d_k
$$

$\alpha=1$ 是純牛頓法；$\alpha<1$ 稱為阻尼牛頓法。

**收斂性（局部）：** 若 $x^*$ 處的 Hessian 正定、且 Hessian 在 $x^*$ 附近 Lipschitz 連續，則存在常數 $C$，起點夠接近 $x^*$ 時

$$
\lVert x_{k+1}-x^*\rVert\le C\,\lVert x_k-x^*\rVert^2
$$

這是**二次收斂**：誤差 $10^{-2}\to10^{-4}\to10^{-8}$，正確的位數每步大約加倍。

**成本：** 每步要算 Hessian（$n^2$ 個二階導數）並解一個 $n\times n$ 線性系統（$O(n^3)$）。

<details markdown="1">
<summary>推導：為什麼二次函數一步就到？</summary>

若 $f(x)=\tfrac12x^\top Ax-b^\top x$，則 $\nabla f=Ax-b$、$\nabla^2f=A$。牛頓方向 $d=-A^{-1}(Ax_k-b)$，所以

$$
x_{k+1}=x_k-A^{-1}(Ax_k-b)=A^{-1}b=x^*
$$

和條件數 $\kappa$、旋轉角度都無關。這叫做**仿射不變性**：把變數做任何可逆的線性變換 $x=Ty$，牛頓法產生的點列完全對應。梯度下降就沒有這個性質，所以會被條件數拖慢。

</details>

<details markdown="1">
<summary>推導：二次收斂從哪裡來？</summary>

令 $e_k=x_k-x^*$，$H_k=\nabla^2 f(x_k)$。因為 $\nabla f(x^*)=0$，

$$
e_{k+1}=e_k-H_k^{-1}\nabla f(x_k)=H_k^{-1}\big(H_k e_k-(\nabla f(x_k)-\nabla f(x^*))\big)
$$

由積分形式的均值定理，$\nabla f(x_k)-\nabla f(x^*)=\int_0^1\nabla^2 f(x^*+te_k)\,e_k\,dt$。括號內是「用 $H_k$ 取代沿途 Hessian 的平均」所造成的誤差；Hessian Lipschitz 常數為 $M$ 時，它不超過 $\tfrac M2\lVert e_k\rVert^2$。再乘上有界的 $\lVert H_k^{-1}\rVert$，就得到 $\lVert e_{k+1}\rVert\le C\lVert e_k\rVert^2$。

</details>

## 應用場景

- **非線性方程求解（牛頓-拉夫森法）：** 解 $F(x)=0$ 時用 Jacobian 代替 Hessian，是數值模式中處理非線性項的標準方法，例如地下水模式中非受壓含水層的非線性方程。
- **統計估計：** 邏輯斯迴歸常用的 IRLS（迭代重新加權最小平方）就是牛頓法；最大概似估計也常用牛頓法或其變形。
- **其他演算法的內層：** 內點法每一步都要做一次牛頓步；信賴域法與 SQP 也以牛頓模型為核心。

## 解決什麼問題

一階方法只看斜率，不看曲率：在窄長的谷裡，梯度方向幾乎垂直於谷的走向，所以需要 $O(\kappa)$ 步。

牛頓法用 Hessian 把「每個方向的曲率」一起考慮進來，等於自動為每個方向配上正確的步長：

- 二次函數一步到位，**和條件數無關**。
- 接近解時二次收斂，幾步就能把誤差降到機器精度。

代價是每步要算並分解 Hessian；而且它只保證局部收斂，離解太遠時可能亂跳，甚至收斂到最大值。這些缺點分別導向 [BFGS](bfgs.html)（避免算 Hessian）與[信賴域法](trust-region.html)（保證全域收斂）。

## 動畫演示

<!-- lab -->

**建議的實驗：**

1. 預設的 Rosenbrock：牛頓法第 6 步就達到 $f<10^{-10}$；GD（$\eta=0.002$）100 步後仍有 $f=1.3\times10^{-3}$。注意牛頓法中途有一步跳到圖外（$y\approx-2.8$，$f=793$），**純牛頓法不保證每步都下降**。
2. 把 $\alpha$ 調到 0.5：第 54 步才達到 $f<10^{-10}$。阻尼讓誤差每步只減半，二次收斂變成線性收斂。
3. 換到二次碗，隨意調 $\kappa$ 和旋轉 $\theta$：牛頓法永遠一步到底。
4. 換到 Himmelblau：從預設起點 $(-0.5,-0.5)$，牛頓法第 5 步就因梯度範數 $<10^{-10}$ 而停止，停在 $(-0.27,-0.92)$，那裡 $f=182$，是**局部最大值**。點擊圖面把起點改到 $(3.5,1.5)$ 附近，就會收斂到最小值 $(3,2)$（從 $(3.5,1.5)$ 實測：第 4 步達到 $f<10^{-10}$）。

## 程式與練習

<!-- practice -->

## 失效情境

- **Hessian 不正定：** 牛頓方向可能不是下降方向，會被吸到鞍點或最大值（實驗 4）。牛頓法只找 $\nabla f=0$ 的點，不分極大極小。
- **離解太遠：** 二次模型不準，可能大幅跳躍甚至發散。實務上要搭配線搜尋或信賴域。
- **Hessian 奇異：** 線性系統無解，方向不存在。
- **高維問題：** $n=10^6$ 時，Hessian 有 $10^{12}$ 個元素，存不下也解不動，這就是深度學習不用牛頓法的原因。

## 面試追問

<details markdown="1">
<summary>Q1. 牛頓法為什麼收斂這麼快？什麼條件下才成立？</summary>

它用二次模型逼近 $f$，模型誤差是 $O(\lVert e\rVert^3)$，所以每步誤差變成前一步的平方（二次收斂）。條件是起點夠接近解、解處 Hessian 正定、而且 Hessian 夠平滑（Lipschitz 連續）。

</details>

<details markdown="1">
<summary>Q2. 為什麼牛頓法可能收斂到最大值？怎麼避免？</summary>

牛頓步只是解 $\nabla m=0$，在 Hessian 負定的地方，二次模型的「駐點」是最大值，牛頓法照樣會跳過去。可以檢查 $d^\top\nabla f<0$，不是下降方向時改用負梯度；或修正 Hessian 使其正定；或改用信賴域法。

</details>

<details markdown="1">
<summary>Q3. 為什麼深度學習不用牛頓法？</summary>

參數量 $n$ 動輒數百萬以上，Hessian 需要 $O(n^2)$ 記憶體、解線性系統需要 $O(n^3)$ 時間，完全不可行；而且隨機小批次的 Hessian 估計雜訊很大。實務上改用一階方法，或 L-BFGS、Hessian-free 等近似方法。

</details>

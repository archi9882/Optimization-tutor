# Optimization Tutor：最佳化演算法圖鑑

一個互動式的最佳化演算法學習網站。目錄收錄 59 種跨領域常用的演算法，每一種都依同一個結構整理：

1. **定義**：數學形式、假設、收斂性，推導分層收合
2. **應用場景**：跨領域的實際例子，包括機器學習、農業、水文與地球科學反演
3. **解決什麼問題**：它補了前一個方法的哪個缺點
4. **動畫演示**：可調參數、可換測試問題、可多個演算法同場比較
5. **程式與練習**：兩個折疊區塊。「虛擬碼與示範程式」顯示的 Python 就是動畫實際執行的程式；「挖空練習」把示範程式的關鍵運算式挖掉讓你填，也可以切換成從零寫，分三層自動檢查，並把你的軌跡畫在參考解上

## 目前進度

| 項目 | 狀態 |
|---|---|
| 目錄（59 種，含篩選與搜尋） | ✅ |
| 渲染器 ①：2D 等高線 + 路徑 + 收斂曲線（含信賴域半徑） | ✅ |
| 完整頁面 | 15 頁（AI 草稿，待作者審閱）<br>一階方法 8 頁：GD、SGD、動量法、Nesterov、AdaGrad、RMSProp、Adam、座標下降<br>二階與擬牛頓 7 頁：牛頓法、高斯-牛頓、Levenberg-Marquardt、BFGS、L-BFGS、非線性 CG、信賴域法 |
| 程式與練習 | 15 種演算法各有虛擬碼、示範程式與挖空練習（另可切換成從零寫） |
| 渲染器 ②～⑤（多面體、族群、Pareto、圖/表格） | 尚未開始 |

## 快速開始

```bash
pip install jinja2 markdown pyyaml pytest
python -m pytest -q              # 參考實作、評分器與建置的測試
python scripts/build.py --serve  # 建置到 _site/ 並開在 http://localhost:8000
```

Windows 可將 `python` 換成 `py`。網站必須透過 `http://` 開啟，直接雙擊 HTML 檔時 Pyodide 無法載入。

預設從 CDN 載入 Pyodide 與 KaTeX。若要離線或在不允許外部請求的環境使用，先用 npm 下載 `pyodide@0.27.7` 與 `katex@0.16.9` 兩個套件，解壓到 `DIR/pyodide`、`DIR/katex`，再執行 `python scripts/build.py --vendor DIR --out _site_offline`。

## 架構

```
optlab/            參考實作（純 Python、無 numpy，CPython 與 Pyodide 共用）
  problems.py      測試問題：二次碗（可調 κ 與旋轉）、Rosenbrock、Himmelblau（含殘差形式）
  demos/           每種演算法一個檔案：虛擬碼 + 示範函式（頁面顯示的程式＝動畫執行的程式）
  optimizers.py    動畫用的優化器清單與參數範圍，step 函式直接取自 demos/
  trace.py         執行器：產生軌跡 {xs, fs, status}
  grading.py       三層評分：單步數值 → 收斂行為 → 軌跡重合
  exercises/       每題的骨架、提示、解答與測資
  api.py           JSON 介面：建置時與瀏覽器中呼叫同一份程式
content/
  catalog.yaml     所有演算法的 metadata（首頁卡片由此產生）
  algorithms/*.md  完整頁面（frontmatter 設定動畫與練習）
site_src/          Jinja2 模板、CSS、JS（渲染器、Pyodide worker）
scripts/build.py   產生靜態網站到 _site/
tests/             pytest
```

**單一來源：** 動畫的預設畫面在建置時由 CPython 執行 `optlab` 預先算好，所以頁面打開就能看到；使用者調整參數時，改由瀏覽器中的 Pyodide 執行同一份 `optlab` 即時重算。練習題的評分器也在瀏覽器中執行，與 `pytest` 用的是同一份程式碼。

**執行安全：** Python 跑在 Web Worker 裡，學習者的程式若陷入無窮迴圈，超過 8 秒會自動終止並重啟 worker。

## 新增一種演算法

1. 在 `content/catalog.yaml` 加一筆 metadata，首頁就會出現它的卡片與骨架頁。
2. 若屬於連續二維問題，在 `optlab/` 加 step 函式並登記到 `OPTIMIZERS`，就能直接使用渲染器 ①。
3. 寫 `content/algorithms/<id>.md`，用 `<!-- lab -->` 和 `<!-- practice -->` 標記插入動畫與練習。
4. 示範程式與虛擬碼放在 `optlab/demos/<id>.py`；練習題（骨架、提示、測資、挖空位置）放在 `optlab/exercises/<id>.py`，並登記到 `EXERCISES`。
5. 執行 `python -m pytest -q`，建置測試會檢查 id、標籤與嵌入資料是否一致。

## 部署

`.github/workflows/pages.yml` 會在推送到 `main` 時跑測試、建置，並部署到 GitHub Pages。第一次使用前，需要在 repo 的 Settings → Pages 把 Source 設為「GitHub Actions」。

## 資料來源與 AI 使用聲明

- **資料：** 全部是合成的解析測試函數，沒有使用外部資料集。
- **AI 協作：** 網站架構、渲染器、Pyodide 整合、目錄 metadata 與各演算法頁面的初稿由 AI（Claude）撰寫。標示「AI 草稿」的頁面尚未經作者逐段審閱。頁面中的數值都是本 repo 程式實際執行的結果。
- **引用：** 目前頁面沒有引用特定文獻。加入引用前，會先查證原始出處。

### 待查證的陳述

以下說法是領域內常見的結論，但頁面上尚未附上已查證的文獻，審閱時需要補上出處或刪除：

| 頁面 | 陳述 |
|---|---|
| 動量法 | 固定參數的 heavy ball 對一般強凸函數不保證加速 |
| Nesterov | 一階方法在凸、L-smooth 函數上的 Ω(1/k²) 下界 |
| AdaGrad | 線上凸最佳化的 O(√T) regret 保證 |
| Adam | 存在讓 Adam 不收斂的簡單凸問題（AMSGrad） |
| Adam | 部分影像任務上 SGD + 動量泛化較好的實證報告 |
| SGD | Robbins–Monro 學習率條件 |
| 座標下降 | SOR 對對稱正定矩陣在 0 < ω < 2 時收斂 |
| 牛頓法 | 局部二次收斂的條件（Hessian 正定且 Lipschitz 連續） |
| 高斯-牛頓 | 地震定位的 Geiger 法屬於高斯-牛頓迭代 |
| Levenberg-Marquardt | 水文模式參數校準軟體普遍採用 LM 類方法 |
| BFGS | 強凸與 Wolfe 線搜尋下的超線性收斂 |
| 信賴域法 | 全域收斂保證與接近解時的二次收斂 |

## 第三方資源

[Pyodide](https://pyodide.org) 0.27.7（jsDelivr CDN）、[KaTeX](https://katex.org) 0.16.9（cdnjs）。

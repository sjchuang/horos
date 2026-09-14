# SAM-T4 — 標註頁「SAM」工具(點/框互動式輔助,SAM 2.1)

工具列第 5 個按鈕「✦」(快捷鍵 5)。第一次點某張圖會跑一次 image encoder(桌機 GPU 約 0.2 s,
首次使用會先下載約 150 MB 的權重,狀態列會說明),之後每一次點擊只跑 prompt decoder(數十 ms)。
候選形狀只在畫面上預覽,按 Enter 才寫入,走的是一般的儲存路徑(版本鎖、多人衝突邏輯不變)。

## 啟動

```
horos ui <project>
瀏覽器開 http://localhost:5000/annotate,點任一張圖進入編輯器
```

## 測試步驟

### A. 點一下就有 mask

1. 按 5 或點「✦」。側欄出現 SAM 面板,狀態列顯示「preparing the image…」,數秒內變成「ready (N ms)」
2. 在 Object Class 輸入類別(例如 `forklift`),面板的「Enter accepts it as forklift」跟著變
3. 左鍵點物體:畫布出現半透明的類別色 mask 加白色虛線邊,點位是綠色「+」;
   狀態列顯示「mask N px² · confidence 0.9x · NN ms」
4. 再點同一物體的另一處:mask 擴大修正,第二次起狀態列不再有「(encoder)」字樣
5. 右鍵點(不拖曳)被誤含進來的區域:紅色「−」點,mask 收回;Shift+左鍵效果相同。
   右鍵按住拖曳仍是平移畫布,不會落點
6. Backspace 移除最後一個點並重算;Esc 清空全部

### B. 粗框

7. 按住左鍵拖一個大概的框放開:藍色虛線框為 prompt,mask 貼合框內物體;可再加正/負點修正

### C. 接受與輸出型別

8. Enter(或按「Accept」):跳出「Which class?」選單,輸入框已帶入目前類別(全選狀態)、
   對應 chip 高亮:再按 Enter 沿用,直接打字可換類別,點 chip 也可;Esc/Discard 丟棄這個形狀。
   形狀以選定類別加入(polygon),工具維持在 SAM,狀態清空,可立刻點下一個物體;
   Auto Save 開著會在 0.5 s 後儲存,Ctrl+Z 可還原
8a. 影像上有 pseudo label(虛線、pending)時,SAM 畫在同一物件上按 Enter:選單預填**該 pseudo label 的類別**
   (以 IoU × 信心度挑最佳、IoU ≥ 0.3 才算),說明列顯示「Suggested from the pseudo label under it (IoU 0.xx, confidence 0.xx)」;
   照片上沒有 pseudo label 時,改問 loop 目前的模型(最新完成的 run,沒有就 OWLv2 依類別名稱 zero-shot)
   這張看到什麼(GET /images/<id>/predictions,每張只問一次並快取),同樣以 IoU × 信心度挑,說明列寫「from the model」;
   模型也沒看到相關物件時回到目前(上次指定)的類別。選單裡的「Suggest the class from the model」可關閉整個建議(預設開)
8b. 選單裡勾「Keep this class for the next objects」(記在 localStorage `horos_keep_class`,預設不勾):
   之後 Enter 直接以 Object Class 的類別寫入、不再跳選單;Object Class 清空時仍會問。
   取消勾選要再開一次選單(清空 Object Class 後 Enter)
9. Output 切成 Box:預覽變成 mask 的外接矩形,Enter 加入的是 rectangle
9b. 「Points ≤」步進器(預設空白 = 完整輪廓):填 16 後正在預覽的 mask 立刻重算成最多 16 個控制點,
    右側顯示「n points」;數值會記住(localStorage),也套用到 ⬠ / P 與「Boxes to polygons」的轉換;
    3 是下限,填低於 3 會被清空
10. 右上 model 下拉可切 SAM 2.1 Tiny / Small / SAM ViT-B;切換後重新 prefetch,選擇會記住

### D. 多物體批次接受(SAM-T5)

14. 點第一個物體得到 mask 後按 <b>Space</b> 或 <b>Enter</b>:兩鍵效果相同,mask 立刻寫入(2026-09-13 起
    Space 不再是「下一個物體」,Next 按鈕已移除),留在 Draw 工具可直接點下一個
15. 多物體批次:得到 mask 後不按鍵、直接拖一個新框(Object Class 需有值),目前的 mask 自動入列成該類別色的
    實線薄邊,狀態列「1 object queued — prompt the next one, Enter / Space accepts all」
16. 再拖第三個框:「Accept」按鈕顯示「Accept 3 (Enter)」
17. Enter 或 Space:所有入列的物體加上目前的候選一次寫入,各自保留當時的類別;toast「Added 3 objects」;
    Ctrl+Z 一次還原整批
18. Esc(或「Clear (Esc)」按鈕)第一下清掉目前物體的點與框(入列的保留);再按一次丟掉整個隊列(toast 提示數量)
19. 切工具或切圖:隊列一併清空(未寫入的不會偷跟到下一張)

### E. 既有 box 當作 hint(SAM-T6)

20. 用 ▭ 工具畫一個 box(或開一張已有 box 標註的圖)。Shapes 列表中 rectangle 的那一列多了「⬠」按鈕;
    點它:該 box 原地變成 polygon(類別、pending 狀態、分數、id 都不變),toast「1 box → polygon」
21. 用 select 工具點選另一個 box,按 <b>P</b>:效果相同;Ctrl+Z 可還原成 box
22. SAM 面板底部「Boxes → polygons (this image)」:這張圖所有 box 一次轉換(embedding 只算一次,
    每個 box 一次 decoder);沒有 box 時按鈕停用。SAM 找不到 mask 的 box 維持 box,toast 註明「kept as box」
23. 尚未儲存的 box 也能轉:結果走一般的儲存流程(Auto Save / Ctrl+S)

### F. 邊界

11. 只放負點:狀態列黃字「no mask for this prompt — add a point inside the object」,Accept 停用
12. 切到別張圖再切回:prompt 清空;若工具仍是 SAM,新圖自動 prefetch
13. macOS 上能力清單 `assist_interactive` 為 limited(較慢),功能不被封鎖

## 預期結果

一張圖的第一次點擊付一次 encoder 成本,之後每次點擊都是即時回饋;所有寫入都經由既有的儲存流程。

## 已知限制

- 一次一張圖;沒有跨影格追蹤(SAM 2 的 video 功能未使用)
- 入列的物體不能再修 prompt;要改就 Esc 丟掉隊列重來,或接受後用 select 工具編輯
- box → polygon 只用 box 當 prompt,沒有正負點;結果不滿意就 Ctrl+Z 後改用 SAM 工具點選修正
- 候選 polygon 由 mask 邊界簡化而來,細碎孔洞不保留
- 觸控裝置沒有右鍵與 Shift,負點需以滑鼠操作

### F. 一個物件分成好幾塊（2026-09-14）

20. 物件被遮擋成幾段：點第一段得到 mask 後按 <b>+</b>（或「+ Part」）：mask 保留、prompt 清空，狀態列「1 part kept — click the object's next piece; Enter / Space finishes it」
21. 點第二段、第三段，每段按 +；Accept 按鈕顯示「Accept · 3 parts (Enter)」
22. Enter 或 Space：寫入「一個」標註，segmentation 有三個多邊形，bbox 涵蓋全部；Shapes 清單顯示「(polygon ×3)」
23. Esc 第一下清目前 prompt、第二下丟掉已保留的 parts、第三下才丟隊列；Output 為 Box 時 + 無效（提示需要 Polygon）
24. Edit 工具：多段物件整體可拖曳、可選取（點任一段）；控制點顯示在「最後點到的那一段」上——點另一段就換到那一段，
    可拖點、點邊線插點、拖點合併，每一段都能微調（多邊形順序會跟著調換，儲存內容等價）
25. 按 + 之後點下一段：新的 mask 只包含被點到的那一段（伺服器改取「點所在的連通區塊」而不是最大的區塊），
    已保留的段以實線淡色顯示在畫布上；prompt 完全重新開始，不會沿用前一段的點或框

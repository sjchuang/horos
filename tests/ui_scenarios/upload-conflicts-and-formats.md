# 上傳擴充 — VOC/Darknet 匯入、檔名衝突確認、純照片與後補標註（E1-T11）

## 怎麼啟動

```bash
horos init ./demo_project
horos ui ./demo_project
# 瀏覽器開 http://localhost:5000
```

## 測試步驟

### A. VOC 與 Darknet 上傳

1. 準備 Roboflow 匯出的 Pascal VOC zip（影像旁有同名 `.xml`），拖進虛線框
2. 應直接匯入成功，摘要列顯示 `(VOC)`；split 依 train/valid/test 目錄歸類
3. 準備 Roboflow 匯出的 Darknet zip（影像旁有同名 `.txt` 與 `_darknet.labels`），拖入
4. 應直接匯入成功，摘要列顯示 `(DARKNET)`，類別名取自 `_darknet.labels`

### B. Darknet 缺類別名清單

1. 準備一包 Darknet zip，但**移除所有 `_darknet.labels`**，拖入
2. 應彈出「Class names needed」對話框，逐類一個輸入框，預設值為類別索引（0、1、…）
3. 修改名稱（如 `helmet`、`vest`）後按「Import with these names」
4. 匯入成功，「Instances per class」表格顯示的是輸入的名稱
5. 重做步驟 1 但按 Cancel：狀態列顯示「Import cancelled — nothing was changed.」，摘要數字不變

### B2. LabelMe 上傳

1. 準備 LabelMe 標註的 zip（每張影像旁有同名 `.json`，含 `shapes` 與 `imagePath`；
   可以是單一資料夾，也可以分 train/valid/test 子目錄），拖進虛線框
2. 應直接匯入成功，摘要列顯示 `(LABELME)`，類別名取自 shape 的 `label`
3. 若 zip 裡有沒有 `.json` 的影像，摘要的 warnings 會列出「N image(s) have no LabelMe JSON —
   imported as unannotated images」，這些影像可直接到 Annotate 頁標註
4. 若 shape 含 circle / line / point 等，warnings 會分別列出轉外接框與跳過的數量，匯入不中斷

### C. 檔名衝突確認

1. 上傳任一資料集 zip，成功後**再上傳同一包**
2. 不應彈窗：狀態列顯示 `Imported 0 images, … — N identical duplicate(s) skipped`（內容相同自動跳過）
3. 修改 zip 裡任一張影像的內容（檔名不變）後重新上傳
4. 應彈出「Some file names already exist」對話框，列出衝突檔名
5. 按「Overwrite」：狀態列顯示 `1 overwritten`，該影像與其標註被新版取代
   （確認後不會重新上傳：zip 已留在伺服器，只送出決定並重跑匯入 job，進度列從 Extracting 重新開始）
6. 重做步驟 3–4 改按「Skip them」：顯示 `1 conflict(s) skipped`，專案維持原樣
7. 重做步驟 3–4 改按「Import renamed」：顯示 `1 renamed`，出現 `xxx_1.jpg` 新檔
8. 重做步驟 3–4 改按「Cancel」：狀態列顯示 cancelled，任何東西都沒被改動，伺服器上暫存的 zip 一併刪除

### D. 純照片匯入（E1-T11）

1. 直接把幾張照片（jpg / png / webp）拖進虛線框，不需要打包
2. 進度列先顯示 `Uploading … · N photos`，接著 `Reading annotations — images…` → `Checking for duplicates` →
   `Copying images`（沒有 Saving annotations 的內容，Assigning splits 一閃而過）
3. 狀態列顯示 `Imported N photo(s) with no labels — label them on the Annotate page, or let the Loop pick a batch`，
   warnings 計數 ≥ 1（內容為「No annotation file found — imported N photo(s) with no labels」）
4. 「Dataset」卡片：images 數增加 N，annotations 不變，「unlabeled · in no set」增加 N；train/valid/test 計數不變
5. 一包只有照片的 zip（任意子目錄）拖進去結果相同，摘要列尾端 `(IMAGES)` 不出現，而是步驟 3 的句子
6. 同一批照片再拖一次：不彈窗，狀態列 `Imported 0 photo(s) … — N identical duplicate(s) skipped`
7. 拖進一個既不是 zip 也不是照片的檔案（如 .txt）：狀態列紅字 `Import failed: Not photos: …`，頁面不卡死

### E. 為已在專案裡的照片後補標註

1. 先依 D 上傳一批純照片，再到 Annotate 頁替其中一張畫一個框並存檔
2. 準備同一批照片的 COCO zip（照片 + `_annotations.coco.json`）拖入
3. 應彈出「These photos already have labels」對話框，只列出步驟 1 標過的那一張；其餘照片不在清單裡
4. 按「Replace」：狀態列 `Imported 0 images, M annotations … — labels applied to N photo(s) already here, 1 relabeled`；
   到 Annotate 頁該張只剩 zip 裡的標註，其他照片各自得到 zip 的標註並進入某個 set
5. 重做 2–3 改按「Keep both」：`1 with both label sets`，該張同時有手畫的框與 zip 的框
6. 重做 2–3 改按「Keep existing」：`1 kept as they were`，該張維持手畫的框，其他照片照常收到標註
7. 重做 2–3 改按「Cancel」：狀態列 cancelled，任何照片都沒被改動
8. 只上傳 `_annotations.coco.json`（zip 裡沒有照片）：照片以檔名對上專案裡的照片並套用標註；
   json 內寬高與專案照片不同的檔名不套用，warnings 會寫出兩邊尺寸
9. 同一包 COCO zip 上傳兩次：第二次不彈窗（標註完全相同視為重複，`identical duplicate(s) skipped`）

## 預期結果

- 六種格式（COCO / YOLO / VOC / Darknet / VIA / LabelMe）拖進去都直接匯入，格式自動偵測；
  純照片（散檔或 zip）也能匯入，成為未標記、不屬於任何 set 的照片
- 內容相同的重複上傳永不彈窗；真正的衝突（檔名不同內容、既有標註會被改）一定先問、未確認前不寫入任何資料

## 已知限制

- VOC 只讀 bbox（VOC 無標準 polygon 表示法）；Darknet 只讀 bbox
- VOC / Darknet / VIA 為僅匯入格式；匯出支援 COCO / YOLO / LabelMe（`horos export --format labelme`
  或 Web API `/api/v1/dataset/export`），UI 目前沒有匯出格式選單
- LabelMe 的 line / linestrip / point / mask 形狀不匯入（有警告計數）；circle 轉為外接框
- 衝突對話框一次套用同一策略到整批衝突，不支援逐張選擇（檔名衝突與標註衝突兩個對話框皆然）
- 散檔照片一次上傳為單一 multipart 請求，數千張請改用 zip

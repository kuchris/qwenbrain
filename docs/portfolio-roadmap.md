# QwenBrain：履歷定位與交付路線

目前已經可以當作「LLM 記憶與持續學習的研究工程專案」介紹。它的價值是建立可追查的
實驗、拆開失敗原因，並依據結果調整設計。若要讓面試官下載後能操作、驗證主張，仍需
補上下面的交付項目。這份文件是路線圖；未完成的項目不代表已實作。

2026-09-12 進度：[GitHub repo](https://github.com/kuchris/qwenbrain) 已建立，
目前設為 private。明確指令的記憶 demo 已完成，Windows 入口通過 13/13 個示範步驟
及互動寫入後重啟查詢。CPU 測試共 26 個，workflow 涵蓋 Python 3.11 的 Linux 與
Windows。四組記憶比較已有[重現入口](reproduction.md)；實際執行證據見
[GPU 重現紀錄](experiments/reproduction-results.md)。公開展示仍待完成。

## 1. 專案定位

建議名稱：**QwenBrain — Local LLM Memory & Continual-Learning Evaluation**。

主線可以用四句話講清楚：

1. 最初假設：每條訊息做一次 LoRA 更新，應該能保留新的個人事實。
2. 實驗拆解：分別測量權重變化、句子 completion、清空上下文後的問答、舊事實保留、
   無關答案，以及結束符行為。Loss 下降不足以代表可用的記憶。
3. 工程比較：加入 frozen、oracle、SQLite retrieval 和 QA/replay 對照，再擴充生效時間、
   觀測時間、歷史更正與 UNKNOWN。
4. 新發現：正確來源已選中時，模型仍可能拒答；嚴格驗證器也可能拒絕內容正確但格式
   不符的回答。因此儲存、讀取、生成、驗證要分別評估。

避免把結果寫成「證明權重不能記憶」、「仿造海馬迴」或「解決災難性遺忘」。這些主張
超出本地小型測試。也不要把 full-name lookup 稱為 semantic search 或把三個資料種子
說成三次獨立模型訓練。

## 2. 現在能拿出來的工程證據

| 能力 | 已實作內容 | 可以如何驗證 |
| --- | --- | --- |
| 訓練系統 | BF16 凍結基座、LoRA、明確 loss mask、候選接受／回滾與持久化 | [更新機制紀錄](history/milestone-3.md)、原始 JSON 與 checksum |
| 實驗設計 | acquisition、replay、stopping、四組記憶對照、時間與 UNKNOWN 測試 | [實驗索引](../reports/README.md)，含完整失敗答案 |
| 資料系統 | SQLite append-only 版本、生效／觀測時間、晚到更正、來源 ID | [memory.py](../qwenbrain/memory.py)、CPU 測試與子程序重開結果 |
| 可靠性 | typed value 檢查、拒答原因、raw/checked 配對比較 | [grounding.py](../qwenbrain/grounding.py)、[測試](../tests/test_grounding.py) |
| 可追查性 | 模型／來源／設定雜湊、分離實驗與 live adapter、逐題輸出 | [fresh-data 結果](experiments/grounded-memory-results.md) |
| 可重現性 | 新建 acquisition checkpoint，依序跑 completion、retention 與四組比較，再從原始答案計分 | [重現指南](reproduction.md)、[執行紀錄](experiments/reproduction-results.md) |

目前有 26 個 CPU 測試方法，其中六個涵蓋 demo 的實際程序重啟、版本更正、錯誤輸入、
唯讀查詢與並行寫入；另六個涵蓋重現流程的來源雜湊、輸出保護、搬動、混用報告拒絕與
重新計分。GPU 重現只涵蓋四組比較的必要流程；stopping、temporal 與 grounded-memory
仍是獨立實驗，不能把這次重現寫成全部 GPU 實驗都重新驗證。

## 3. 還缺什麼：依交付順序

| 優先序 | 缺口 | 最小交付 | 完成條件 |
| --- | --- | --- | --- |
| 完成 | 明確指令的端到端記憶操作 | `memory-demo.bat`：寫入、查詢、修改、歷史查詢、重啟與一鍵示範 | 13/13 個示範步驟通過；實際跨程序讀回版本與 source ID，未知內容回 UNKNOWN |
| 完成 | 四組比較依賴本機舊訓練檔案與人工彙整 | `scripts/reproduce-memory.bat`：明確輸入、checkpoint 雜湊與自動重新計分 | 獨立 source export、新 venv、新訓練 checkpoint 跑完四階段；12 個預訓練模型檔複製後逐一驗證；CPU 重算與原摘要逐 byte 相同 |
| P1 | 新資料仍沿用問句與情境模板 | 先固定未用過的問句／情境，再做一輪獨立評估 | 分開報 extraction、retrieval、answer、abstention、錯誤接受和錯誤拒絕；不只報總分 |
| P1 | 無關能力與工程成本測得太少 | 擴大互不重複的控制題，記錄查詢延遲、寫入成本、儲存量及樣本數 | 資料與輸出可稽核；時間比較控制生成長度，或明確分報各階段 |
| P1 | 公開展示包仍待完成 | 公開可見性、授權文件與示範錄影；repo、CPU workflow 和 CLI demo 已建立 | 外部讀者能找到程式、執行測試、看證據與限制 |

接著做基本發布包與獨立評估。更多模型、向量資料庫、MoE 和
consolidation 訓練可以等到現有瓶頸被定位後再決定。現有歷史答案失敗常發生在來源已正確取出的情況；
擴大儲存容量並不直接處理這個問題。

## 4. 已完成：可展示的記憶 demo

獨立入口 [memory-demo.bat](../memory-demo.bat) 支援 `person -> project_name`，
沿用現有 versioned store，以精確姓名與日期取值。原本 `chat.bat` 保持原有實驗行為。
執行 `.\memory-demo.bat walkthrough` 可重播全部步驟；
[使用指南](memory-demo.md) 說明日期語意、來源和限制。

以下是新資料庫中的互動操作範例；實際輸出為含有來源欄位的 JSON：

```text
remember "Aster Rook" project_name Cobalt --valid-from 2026-01-01
ask "What is Aster Rook's project called?"
  Cobalt | source_id=1 | mode=source_value

remember "Aster Rook" project_name Vesper --valid-from 2026-02-01
ask "What was Aster Rook's project called on 2026-01-15?"
  Cobalt | source_id=1 | mode=source_value

quit
[restart]
ask "What is Aster Rook's project called?"
  Vesper | source_id=2 | mode=source_value

ask "What is Aster Rook's favorite color?"
  UNKNOWN | reason=unsupported_query
```

第一版編碼器使用明確指令與型別檢查，儲存原始輸入及來源。這讓示範先驗證持久化與
版本語意；它不等於「從每句自然語言自動抽取記憶」。自然語言抽取應是下一個獨立評估。

精確欄位查詢可以直接回傳資料庫的 canonical value。若保留 LLM 生成供比較，就分別
標記 `source_value` 與 `model_generated`，各自計分。不能把直接讀欄位的正確率說成
模型學會記憶，也不應把不同路徑混成一個改善百分比。

寫入／重新啟動／修改／晚到歷史更正／UNKNOWN 已有可重播的整合案例；每個回答附來源
或拒答原因。實際 Windows 啟動檔的[驗證紀錄](../reports/memory-demo-validation.json)
包含逐步輸出，以及查詢前後的資料庫雜湊。

## 5. 履歷文字草稿：僅使用現在已有的能力

**QwenBrain — Local LLM Memory & Continual-Learning Evaluation**

Python · PyTorch · Transformers · PEFT/LoRA · SQLite · uv

- Built a local Qwen3.5-0.8B evaluation framework comparing LoRA adaptation,
  oracle evidence and SQLite retrieval, with fresh-context queries, checkpoint
  checksums and unrelated-task regression probes.
- Implemented versioned memory with effective/observation timestamps and
  source-based answer checks; evaluated 24 synthetic identities across 336
  staged queries and added 26 CPU test methods covering persistence, temporal
  lookup, grounding and experiment artifact integrity.
- Built a four-stage GPU reproduction pipeline with explicit artifact inputs,
  checkpoint checksums and automatic raw-output scoring; verified it in a fresh
  source export and locked environment using newly trained adapters.

336 包含 answerable、UNKNOWN 與控制題，也包含重複模板和階段；獨立已知人物數是 24。
這兩點適合放在專案頁或面試解說，避免把題數包裝成資料多樣性。以上是草稿，尚未寫入
正式 CV；公開展示仍待完成。重現流程的範圍與實際驗證條件見
[GPU 重現紀錄](experiments/reproduction-results.md)。

若面試官追問結果，可以引用：四組小型對照中 SQLite 為 32/32、指定 LoRA/replay 設定
為 25/32；更進一步的 temporal 測試顯示取對來源仍不保證生成正確。配合
[完整限制](experiments/memory-results.md) 解釋，比單寫「100% accuracy」更能展示判斷力。

## 6. 整理後的檔案規則

- 根目錄：README、setup/chat/memory-demo 入口、環境與 dependency lock。
- `scripts/`：實驗與 GPU 檢查入口；執行條件見該資料夾 README。
- `docs/experiments/`：protocol、結果解讀；`docs/research/`：文獻；
  `docs/history/`：原始計畫與早期里程碑。
- `reports/`：保留原始 JSON 路徑；新增索引，避免混淆原始量測與敘事。
- `qwenbrain/`、`configs/`、`tests/`：保留既有模組與 fixture 路徑。
- `runs/`、`models/`、`adapters/`、`data/`、`.cache/`：本機產物，不因整理刪除。

檔案整理階段只調整文件、導航與 batch 入口路徑。搬動前的檔案副本與 SHA-256 清單保存在本機
`runs/organization-20260912-132238/`。舊量測 JSON 裡的路徑與 source hash 保留原樣；
它們是歷史 provenance，不因移動啟動檔而重新編寫。

# FindForYou - 日常物品搜尋助手

使用 YOLO-World 開放詞彙偵測技術，幫助你快速找到家中日常生活用品。

## 🎯 功能特色

- **智慧偵測** - 使用 YOLO-World 自動偵測並追蹤日常物品
- **開放詞彙** - 可動態新增任何想偵測的物品類別（英文+中文對照）
- **多攝影機支援** - 支援多個攝影機，可設定名稱和所在位置
- **自動定時偵測** - 可開關的背景定時偵測，透過 WebSocket 即時推送
- **設定頁面** - 完整的設定介面管理攝影機、類別和自動偵測
- **常用物品快捷** - 可自訂首頁的常用物品快捷搜尋按鈕
- **即時預覽** - 設定頁面可預覽各攝影機畫面
- **語音輸入** - 支援語音搜尋（需瀏覽器支援）
- **離線使用** - 資料存在瀏覽器 IndexedDB，離線也能查詢

## 📁 專案結構

```
object-finder-app/
├── backend/                  # Python 後端服務
│   ├── main.py              # FastAPI 入口
│   ├── detector.py          # YOLO-World 偵測器
│   ├── scheduler.py         # 定時排程器
│   ├── custom_classes.json  # 自訂類別配置
│   ├── camera_config.json   # 攝影機配置
│   └── requirements.txt     # Python 依賴
│
└── frontend/                 # Web 前端
    ├── index.html           # 主頁面
    ├── settings.html        # 設定頁面
    ├── css/style.css        # 現代化樣式
    └── js/
        ├── app.js           # 主程式
        ├── db.js            # IndexedDB 操作
        ├── api.js           # API 通訊
        └── ui.js            # UI 互動
```

## 🚀 快速開始

### 1. 啟動後端服務

```bash
cd backend

# 安裝依賴
pip install -r requirements.txt

# 啟動服務
uvicorn main:app --host 0.0.0.0 --port 8000
```

伺服器會在 `http://localhost:8000` 啟動。

### 2. 開啟前端頁面

直接瀏覽器開啟 `http://localhost:8000`

## 🔧 類別管理 API

支援動態管理偵測類別：

```bash
# 取得目前類別
GET /api/classes

# 設定類別
POST /api/classes
{"classes": ["phone", "keys", "wallet"]}

# 新增類別
POST /api/classes/add
{"class_name": "umbrella", "class_name_zh": "雨傘"}

# 移除類別
DELETE /api/classes/{class_name}
```

## 📱 預設偵測物品

| 圖示 | 英文 | 中文 |
|------|------|------|
| 👓 | glasses | 眼鏡 |
| 📱 | cell phone | 手機 |
| 👛 | wallet | 錢包 |
| 🔑 | keys | 鑰匙 |
| 📺 | remote | 遙控器 |
| 📖 | book | 書 |
| ☕ | cup | 杯子 |
| 🍶 | bottle | 水瓶 |
| 🎧 | headphones | 耳機 |
| ⌚ | watch | 手錶 |

可透過 API 或修改 `custom_classes.json` 自訂！

## 🛠️ 技術棧

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **CV**: YOLO-World (Ultralytics), OpenCV
- **Storage**: IndexedDB (瀏覽器端)

## 📄 License

MIT License

"""
物件偵測器模組
使用 YOLO-World 開放詞彙偵測
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# 嘗試導入 ultralytics
try:
    from ultralytics import YOLOWorld
    YOLO_AVAILABLE = True
except ImportError:
    try:
        from ultralytics import YOLO as YOLOWorld
        YOLO_AVAILABLE = True
        print("⚠️ YOLOWorld 未找到，嘗試使用 YOLO")
    except ImportError:
        YOLO_AVAILABLE = False
        print("⚠️ ultralytics 未安裝，使用模擬模式")


# ========================================
# 設定
# ========================================

# 配置檔路徑
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "custom_classes.json")

# 預設類別（若配置檔不存在）
DEFAULT_CLASSES = [
    "glasses",
    "cell phone", 
    "wallet",
    "keys",
    "remote",
    "medicine bottle",
    "hearing aid",
    "book",
    "cup",
    "bottle"
]

# 預設中文對照
DEFAULT_CLASS_NAMES_ZH = {
    "glasses": "眼鏡",
    "cell phone": "手機",
    "wallet": "錢包",
    "keys": "鑰匙",
    "remote": "遙控器",
    "medicine bottle": "藥罐",
    "hearing aid": "助聽器",
    "book": "書",
    "cup": "杯子",
    "bottle": "水瓶",
    "clock": "時鐘",
    "scissors": "剪刀",
}

# 表面區域定義
DEFAULT_SURFACES = {
    "sofa": {"bbox": [0, 200, 800, 500], "name_zh": "沙發"},
    "table": {"bbox": [200, 100, 600, 300], "name_zh": "桌子"},
    "desk": {"bbox": [600, 150, 800, 400], "name_zh": "書桌"},
}


@dataclass
class Detection:
    """偵測結果資料類別"""
    object_class: str
    confidence: float
    bbox: List[float]
    surface: Optional[str] = None
    region: Optional[str] = None
    timestamp: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class ObjectDetector:
    """YOLO-World 物件偵測器類別"""
    
    def __init__(
        self, 
        model_path: str = "yolov8x-worldv2.pt",  # 最大模型，最高精準度
        camera_source: int = 0,
        config_path: str = CONFIG_PATH
    ):
        self.model_path = model_path
        self.camera_source = camera_source
        self.config_path = config_path
        self.model = None
        self.is_ready = False
        self.surfaces = DEFAULT_SURFACES
        
        # 類別管理
        self.class_definitions: List[Dict] = [] # 儲存完整的類別定義
        self.custom_classes: List[str] = [] # 僅儲存 ID 列表 (給前端用)
        self.class_names_zh: Dict[str, str] = {} # ID -> 中文名稱
        self.prompt_map: Dict[str, str] = {} # Prompt -> ID
        self.active_prompts: List[str] = [] # 給 YOLO 的所有 Prompts
        
        # 載入配置
        self._load_config()
        self._init_model()
    
    def _load_config(self):
        """載入自訂類別配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 檢查這是否是新格式 (classes 是 list of dicts)
                raw_classes = config.get("classes", [])
                if raw_classes and isinstance(raw_classes[0], dict):
                    self.class_definitions = raw_classes
                else:
                    # 舊格式轉換為新格式
                    old_classes = config.get("classes", DEFAULT_CLASSES)
                    old_names_zh = config.get("class_names_zh", DEFAULT_CLASS_NAMES_ZH)
                    self.class_definitions = []
                    for cls_id in old_classes:
                        self.class_definitions.append({
                            "id": cls_id,
                            "prompts": [cls_id],
                            "name_zh": old_names_zh.get(cls_id, cls_id)
                        })
            else:
                # 使用預設值
                self.class_definitions = []
                for cls_id in DEFAULT_CLASSES:
                    self.class_definitions.append({
                        "id": cls_id,
                        "prompts": [cls_id],
                        "name_zh": DEFAULT_CLASS_NAMES_ZH.get(cls_id, cls_id)
                    })
                self._save_config()

            # 重建索引和對照表
            self._rebuild_indices()
            print(f"✅ 載入自訂類別: {len(self.custom_classes)} 個 (共 {len(self.active_prompts)} 個提示詞)")

        except Exception as e:
            print(f"⚠️ 載入配置失敗: {e}，使用預設值")
            self.class_definitions = []
            for cls_id in DEFAULT_CLASSES:
                self.class_definitions.append({
                    "id": cls_id,
                    "prompts": [cls_id],
                    "name_zh": DEFAULT_CLASS_NAMES_ZH.get(cls_id, cls_id)
                })
            self._rebuild_indices()

    def _rebuild_indices(self):
        """從 class_definitions 重建所有輔助索引"""
        self.custom_classes = []
        self.class_names_zh = {}
        self.prompt_map = {}
        self.active_prompts = []
        
        for item in self.class_definitions:
            cls_id = item["id"]
            prompts = item.get("prompts", [cls_id])
            name_zh = item.get("name_zh", cls_id)
            
            self.custom_classes.append(cls_id)
            self.class_names_zh[cls_id] = name_zh
            
            for p in prompts:
                # 確保 prompt 是字串且不重複 (雖然 logic 上同一個 prompt 指向不同 ID 會有歧義，這裡以後加入的為準或視為無效)
                if p not in self.prompt_map:
                    self.prompt_map[p] = cls_id
                    self.active_prompts.append(p)
    
    def _save_config(self):
        """儲存自訂類別配置"""
        config = {
            "classes": self.class_definitions
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✅ 配置已儲存")
        except Exception as e:
            print(f"⚠️ 儲存配置失敗: {e}")
    
    def _init_model(self):
        """初始化 YOLO-World 模型"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO 不可用，使用模擬模式")
            self.is_ready = True
            return
        
        try:
            # 載入 YOLO-World 模型
            self.model = YOLOWorld(self.model_path)
            
            # 設定使用 GPU
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                print(f"✅ 模型已載入到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ CUDA 不可用，使用 CPU")
            
            # 設定自訂類別 (使用所有 prompts)
            self._update_model_classes()
            
            self.is_ready = True
            print(f"✅ YOLO-World 模型已載入: {self.model_path}")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            self.is_ready = True  # 使用模擬模式

    def _update_model_classes(self):
        """更新模型的類別列表"""
        if self.model and hasattr(self.model, 'set_classes'):
            # YOLO-World 需要 list of strings
            try:
                self.model.set_classes(self.active_prompts)
                print(f"✅ YOLO-World 類別已更新: {len(self.active_prompts)} 個提示詞")
            except Exception as e:
                print(f"❌ 設定模型類別失敗: {e}")

    # ========================================
    # 類別管理 API
    # ========================================
    
    def get_classes(self) -> Dict[str, Any]:
        """取得目前偵測類別列表"""
        return {
            "classes": self.custom_classes,
            "class_names_zh": self.class_names_zh,
            "class_definitions": self.class_definitions  # 新增：完整定義
        }
    
    def set_classes(self, classes: List[str]) -> bool:
        """設定要偵測的類別 (舊版 API 相容)
        注意：這裡傳入的是 ID 列表。如果 ID 存在於現有定義中，保留它；
        如果不存在，則新增一個單一 prompt 的類別。
        這會覆寫目前的 class_definitions。
        """
        try:
            new_definitions = []
            
            # 建立現有定義的 lookup
            current_def_map = {d["id"]: d for d in self.class_definitions}
            
            for cls_id in classes:
                if cls_id in current_def_map:
                    new_definitions.append(current_def_map[cls_id])
                else:
                    # 新增預設
                    new_definitions.append({
                        "id": cls_id,
                        "prompts": [cls_id],
                        "name_zh": cls_id
                    })
            
            self.class_definitions = new_definitions
            self._rebuild_indices()
            self._update_model_classes()
            self._save_config()
            
            print(f"✅ 類別已更新 (Set): {classes}")
            return True
        except Exception as e:
            print(f"❌ 設定類別失敗: {e}")
            return False
    
    def add_class(self, class_name: str, class_name_zh: Optional[str] = None) -> bool:
        """新增單一類別"""
        if class_name in self.custom_classes:
            return False
        
        new_def = {
            "id": class_name,
            "prompts": [class_name],
            "name_zh": class_name_zh if class_name_zh else class_name
        }
        
        self.class_definitions.append(new_def)
        self._rebuild_indices()
        self._update_model_classes()
        self._save_config()
        
        print(f"✅ 新增類別: {class_name}")
        return True
    
    def remove_class(self, class_name: str) -> bool:
        """移除類別"""
        if class_name not in self.custom_classes:
            return False
        
        self.class_definitions = [d for d in self.class_definitions if d["id"] != class_name]
        self._rebuild_indices()
        self._update_model_classes()
        self._save_config()
        
        print(f"✅ 移除類別: {class_name}")
        return True
    
    def get_class_name_zh(self, class_name: str) -> str:
        """取得類別的中文名稱"""
        return self.class_names_zh.get(class_name, class_name)
    
    # ========================================
    # 偵測功能
    # ========================================
    
    async def detect_snapshot(self, save_image: bool = True) -> tuple:
        """從攝影機擷取快照並進行偵測
        
        Returns:
            tuple: (detections, image_path)
        """
        
        if not YOLO_AVAILABLE or self.model is None:
            return self._get_mock_detections(), None
        
        try:
            # 開啟攝影機
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                print("⚠️ 無法開啟攝影機，使用模擬資料")
                return self._get_mock_detections(), None
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return self._get_mock_detections(), None
            
            # 執行偵測
            detections = self._detect_frame(frame)
            
            # 儲存截圖
            image_path = None
            if save_image:
                image_path = self._save_snapshot(frame, detections)
            
            return detections, image_path
            
        except Exception as e:
            print(f"❌ 偵測失敗: {e}")
            return self._get_mock_detections(), None
    
    def _save_snapshot(self, frame: np.ndarray, detections: List[Detection]) -> str:
        """儲存截圖並在圖片上畫出偵測框"""
        # 確保 static 資料夾存在
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)
        
        # 畫偵測框
        frame_with_boxes = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det.bbox]
            # 顯示 ID 和 中文名
            name_zh = self.class_names_zh.get(det.object_class, det.object_class)
            label = f"{name_zh} {det.confidence:.0%}"
            
            # 畫框
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 畫標籤背景
            # 支援中文顯示需要特殊處理 (OpenCV 不支援中文)，這裡先用英文 ID 如果無法顯示中文
            # 為了簡單起見，這裡還是主要顯示 ID，或者需要用 PIL 畫中文
            # 暫時顯示 ID + 分數
            display_text = f"{det.object_class} {det.confidence:.0%}"
            
            (w, h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_with_boxes, (x1, y1 - 25), (x1 + w + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame_with_boxes, display_text, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 儲存圖片
        filename = f"snapshot_{int(datetime.now().timestamp() * 1000)}.jpg"
        filepath = os.path.join(static_dir, filename)
        cv2.imwrite(filepath, frame_with_boxes)
        
        print(f"📸 截圖已儲存: {filename}")
        return f"/static/{filename}"
    
    def _detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """對單幀影像進行偵測"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue
            
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            
            # 取得類別名稱
            names = r.names if hasattr(r, 'names') else {}
            
            for box, conf, cls in zip(boxes, confs, clss):
                cls_id = int(cls)
                
                # 取得預測的 prompt
                if isinstance(names, dict):
                    predicted_prompt = names.get(cls_id, f"class_{cls_id}")
                elif cls_id < len(self.active_prompts):
                    predicted_prompt = self.active_prompts[cls_id]
                else:
                    predicted_prompt = f"class_{cls_id}"
                
                # 映射回 Canonical ID
                # 如果找不到 (通常不應該發生，除非 YOLO 輸出怪怪的)，就直接用 prompt
                canonical_id = self.prompt_map.get(predicted_prompt, predicted_prompt)
                
                # [Debug] 顯示觸發的 Prompt
                if predicted_prompt != canonical_id:
                     print(f"🔍 [Multi-Prompt] Detected '{predicted_prompt}' => Mapped to '{canonical_id}'")
                
                bbox = [float(x) for x in box]
                
                # 判斷所在表面
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                surface, region = self._get_surface_region(cx, cy)
                
                detections.append(Detection(
                    object_class=canonical_id, # 這裡是重點：存入統一的 ID
                    confidence=float(conf),
                    bbox=bbox,
                    surface=surface,
                    region=region,
                    timestamp=int(datetime.now().timestamp() * 1000)
                ))
        
        return detections
    
    def _get_surface_region(self, cx: float, cy: float) -> tuple:
        """判斷物品所在的表面和區域"""
        for surface_name, surface_info in self.surfaces.items():
            bbox = surface_info["bbox"]
            x1, y1, x2, y2 = bbox
            
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # 計算區域 (左/中/右)
                width = x2 - x1
                rel_x = (cx - x1) / width
                
                if rel_x < 0.33:
                    region = "left"
                elif rel_x < 0.66:
                    region = "center"
                else:
                    region = "right"
                
                return surface_name, region
        
        # 返回空值，讓 main.py 使用攝影機配置的位置覆蓋
        return "", ""
    
    def _get_mock_detections(self) -> List[Detection]:
        """產生模擬偵測資料（用於測試）"""
        import random
        
        # 使用自訂類別產生模擬資料
        mock_items = [
            ("cell phone", "sofa", "left", 0.95),
            ("remote", "table", "center", 0.88),
            ("glasses", "desk", "right", 0.92),
            ("keys", "table", "left", 0.85),
            ("wallet", "sofa", "center", 0.90),
        ]
        
        # 過濾只保留目前自訂類別中的物品
        available_items = [
            item for item in mock_items 
            if item[0] in self.custom_classes
        ]
        
        if not available_items:
            available_items = mock_items[:3]
        
        # 隨機選擇 1-3 個物品
        selected = random.sample(
            available_items, 
            k=min(random.randint(1, 3), len(available_items))
        )
        
        return [
            Detection(
                object_class=item[0],
                confidence=item[3] + random.uniform(-0.05, 0.05),
                bbox=[100.0, 100.0, 200.0, 200.0],
                surface=item[1],
                region=item[2],
                timestamp=int(datetime.now().timestamp() * 1000)
            )
            for item in selected
        ]
    
    def set_surfaces(self, surfaces: Dict[str, Any]):
        """設定表面區域定義"""
        self.surfaces = surfaces

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
        self.custom_classes: List[str] = []
        self.class_names_zh: Dict[str, str] = {}
        
        # 載入配置
        self._load_config()
        self._init_model()
    
    def _load_config(self):
        """載入自訂類別配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.custom_classes = config.get("classes", DEFAULT_CLASSES)
                self.class_names_zh = config.get("class_names_zh", DEFAULT_CLASS_NAMES_ZH)
                print(f"✅ 載入自訂類別: {len(self.custom_classes)} 個")
            except Exception as e:
                print(f"⚠️ 載入配置失敗: {e}，使用預設值")
                self.custom_classes = DEFAULT_CLASSES.copy()
                self.class_names_zh = DEFAULT_CLASS_NAMES_ZH.copy()
        else:
            self.custom_classes = DEFAULT_CLASSES.copy()
            self.class_names_zh = DEFAULT_CLASS_NAMES_ZH.copy()
            self._save_config()
    
    def _save_config(self):
        """儲存自訂類別配置"""
        config = {
            "classes": self.custom_classes,
            "class_names_zh": self.class_names_zh
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
            
            # 設定自訂類別
            if hasattr(self.model, 'set_classes'):
                self.model.set_classes(self.custom_classes)
                print(f"✅ YOLO-World 類別已設定: {self.custom_classes}")
            
            self.is_ready = True
            print(f"✅ YOLO-World 模型已載入: {self.model_path}")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            self.is_ready = True  # 使用模擬模式
    
    # ========================================
    # 類別管理 API
    # ========================================
    
    def get_classes(self) -> Dict[str, Any]:
        """取得目前偵測類別列表"""
        return {
            "classes": self.custom_classes,
            "class_names_zh": self.class_names_zh
        }
    
    def set_classes(self, classes: List[str]) -> bool:
        """設定要偵測的類別"""
        try:
            self.custom_classes = classes
            
            # 更新模型
            if self.model and hasattr(self.model, 'set_classes'):
                self.model.set_classes(classes)
            
            self._save_config()
            print(f"✅ 類別已更新: {classes}")
            return True
        except Exception as e:
            print(f"❌ 設定類別失敗: {e}")
            return False
    
    def add_class(self, class_name: str, class_name_zh: Optional[str] = None) -> bool:
        """新增單一類別"""
        if class_name in self.custom_classes:
            return False
        
        self.custom_classes.append(class_name)
        if class_name_zh:
            self.class_names_zh[class_name] = class_name_zh
        
        # 更新模型
        if self.model and hasattr(self.model, 'set_classes'):
            self.model.set_classes(self.custom_classes)
        
        self._save_config()
        print(f"✅ 新增類別: {class_name}")
        return True
    
    def remove_class(self, class_name: str) -> bool:
        """移除類別"""
        if class_name not in self.custom_classes:
            return False
        
        self.custom_classes.remove(class_name)
        self.class_names_zh.pop(class_name, None)
        
        # 更新模型
        if self.model and hasattr(self.model, 'set_classes'):
            self.model.set_classes(self.custom_classes)
        
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
            label = f"{det.object_class} {det.confidence:.0%}"
            
            # 畫框
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 畫標籤背景
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_with_boxes, (x1, y1 - 25), (x1 + w + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame_with_boxes, label, (x1 + 5, y1 - 8), 
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
                
                # 取得類別名稱
                if isinstance(names, dict):
                    class_name = names.get(cls_id, f"class_{cls_id}")
                elif cls_id < len(self.custom_classes):
                    class_name = self.custom_classes[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                bbox = [float(x) for x in box]
                
                # 判斷所在表面
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                surface, region = self._get_surface_region(cx, cy)
                
                detections.append(Detection(
                    object_class=class_name,
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
        
        return "unknown", "unknown"
    
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

"""
找東西助手 - 後端 API 服務
FastAPI 提供偵測服務和 API 端點
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from detector import ObjectDetector
from scheduler import DetectionScheduler


# ========================================
# 資料模型
# ========================================

class Detection(BaseModel):
    """單一偵測結果"""
    object_class: str
    confidence: float
    bbox: List[float]
    surface: Optional[str] = None
    region: Optional[str] = None
    timestamp: Optional[int] = None


class DetectionResponse(BaseModel):
    """偵測回應"""
    success: bool
    detections: List[Detection]
    timestamp: int
    message: Optional[str] = None
    image_path: Optional[str] = None  # 截圖路徑


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    version: str
    detector_ready: bool
    scheduler_running: bool


# ========================================
# 全域變數
# ========================================

detector: Optional[ObjectDetector] = None
scheduler: Optional[DetectionScheduler] = None
connected_websockets: List[WebSocket] = []
latest_detections: List[Detection] = []


# ========================================
# 生命週期管理
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    global detector, scheduler
    
    print("🚀 啟動找東西助手後端服務...")
    
    # 初始化偵測器
    try:
        detector = ObjectDetector()
        print("✅ 物件偵測器已載入")
    except Exception as e:
        print(f"⚠️ 偵測器載入失敗: {e}")
        detector = None
    
    # 初始化排程器
    scheduler = DetectionScheduler(
        detector=detector,
        on_detection=broadcast_detection,
        interval_seconds=30
    )
    
    yield
    
    # 清理資源
    print("🛑 關閉服務...")
    if scheduler:
        scheduler.stop()


# ========================================
# FastAPI 應用程式
# ========================================

app = FastAPI(
    title="FindForYou API",
    description="物品定位服務後端 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# API 端點
# ========================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        detector_ready=detector is not None and detector.is_ready,
        scheduler_running=scheduler is not None and scheduler.is_running
    )


# ========================================
# 攝影機管理 API
# ========================================

@app.get("/api/cameras")
async def list_cameras():
    """列出可用的攝影機"""
    import cv2
    cameras = []
    config = load_camera_config()
    
    # 測試攝影機 0-5
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # 使用用戶配置的名稱，沒有則用預設
                cam_config = config.get("cameras", {}).get(str(i), {})
                name = cam_config.get("name", f"攝影機 {i}")
                location = cam_config.get("location", "")
                
                cameras.append({
                    "id": i,
                    "name": name,
                    "location": location,
                    "display": f"{name} ({location})" if location else name
                })
            cap.release()
    
    return {
        "cameras": cameras,
        "current": detector.camera_source if detector else 0
    }


@app.get("/api/cameras/{camera_id}/preview")
async def camera_preview(camera_id: int):
    """取得攝影機預覽圖片"""
    import cv2
    import base64
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"攝影機 {camera_id} 無法開啟")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="無法擷取畫面")
    
    # 縮小圖片
    height, width = frame.shape[:2]
    scale = 640 / width
    new_size = (640, int(height * scale))
    frame = cv2.resize(frame, new_size)
    
    # 轉換為 base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "success": True,
        "camera_id": camera_id,
        "image": f"data:image/jpeg;base64,{img_base64}"
    }


@app.post("/api/cameras/{camera_id}")
async def set_camera(camera_id: int):
    """設定使用的攝影機"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    # 測試攝影機是否可用
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail=f"攝影機 {camera_id} 無法開啟")
    cap.release()
    
    detector.camera_source = camera_id
    return {
        "success": True,
        "message": f"已切換到攝影機 {camera_id}",
        "current": camera_id
    }


# 攝影機配置檔路徑
CAMERA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")


def load_camera_config():
    """載入攝影機配置"""
    if os.path.exists(CAMERA_CONFIG_PATH):
        with open(CAMERA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"cameras": {}, "default_camera": 0}


def save_camera_config(config):
    """儲存攝影機配置"""
    with open(CAMERA_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class CameraConfigRequest(BaseModel):
    """攝影機配置請求"""
    camera_id: str
    name: str
    location: str
    enabled: bool = True


@app.get("/api/cameras/config")
async def get_camera_config():
    """取得攝影機配置"""
    config = load_camera_config()
    return config


@app.post("/api/cameras/config")
async def set_camera_config(request: CameraConfigRequest):
    """設定單一攝影機配置"""
    config = load_camera_config()
    
    config["cameras"][request.camera_id] = {
        "name": request.name,
        "location": request.location,
        "enabled": request.enabled
    }
    
    save_camera_config(config)
    
    return {
        "success": True,
        "message": f"攝影機 {request.camera_id} 配置已儲存",
        "config": config
    }


@app.delete("/api/cameras/config/{camera_id}")
async def delete_camera_config(camera_id: str):
    """刪除攝影機配置"""
    config = load_camera_config()
    
    if camera_id in config["cameras"]:
        del config["cameras"][camera_id]
        save_camera_config(config)
        return {"success": True, "message": f"攝影機 {camera_id} 配置已刪除"}
    
    return {"success": False, "message": f"找不到攝影機 {camera_id}"}


@app.post("/api/snapshot", response_model=DetectionResponse)
async def trigger_snapshot():
    """手動觸發快照偵測"""
    global latest_detections
    
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    try:
        raw_detections, image_path = await detector.detect_snapshot()
        
        # 取得當前攝影機的位置配置
        camera_config = load_camera_config()
        current_camera = str(detector.camera_source)
        camera_location = "unknown"
        
        if current_camera in camera_config.get("cameras", {}):
            camera_location = camera_config["cameras"][current_camera].get("location", "unknown")
        
        # 轉換 dataclass 為 Pydantic 模型，並設定 surface 為攝影機位置
        detections = [
            Detection(
                object_class=d.object_class,
                confidence=d.confidence,
                bbox=d.bbox,
                surface=camera_location,  # 使用攝影機配置的位置
                region=d.region,
                timestamp=d.timestamp
            ) for d in raw_detections
        ]
        
        latest_detections = detections
        
        # 廣播給所有連線的 WebSocket
        await broadcast_detection(detections)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            timestamp=int(datetime.now().timestamp() * 1000),
            message=f"快照偵測完成，找到 {len(detections)} 個物品",
            image_path=image_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """上傳圖片進行偵測"""
    global latest_detections
    
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    # 檢查檔案類型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    try:
        import cv2
        import numpy as np
        
        # 讀取上傳的圖片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        # 執行偵測
        detections = detector._detect_frame(frame)
        latest_detections = detections
        
        # 廣播給所有連線的 WebSocket
        await broadcast_detection(detections)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            timestamp=int(datetime.now().timestamp() * 1000),
            message=f"偵測完成，找到 {len(detections)} 個物品"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detections/latest", response_model=DetectionResponse)
async def get_latest_detections():
    """取得最新偵測結果"""
    return DetectionResponse(
        success=True,
        detections=latest_detections,
        timestamp=int(datetime.now().timestamp() * 1000)
    )


@app.post("/api/detections", response_model=DetectionResponse)
async def save_detection(detection: Detection):
    """儲存單筆偵測資料"""
    global latest_detections
    
    try:
        # 設定時間戳記
        if detection.timestamp is None:
            detection.timestamp = int(datetime.now().timestamp() * 1000)
        
        # 更新最新偵測
        latest_detections = [detection]
        
        # 廣播給所有連線的 WebSocket
        await broadcast_detection([detection])
        
        return DetectionResponse(
            success=True,
            detections=[detection],
            timestamp=detection.timestamp,
            message="偵測資料已儲存"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detections/batch", response_model=DetectionResponse)
async def save_detections_batch(detections: List[Detection]):
    """批次儲存偵測資料"""
    global latest_detections
    
    try:
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # 為沒有時間戳記的資料設定時間
        for d in detections:
            if d.timestamp is None:
                d.timestamp = timestamp
        
        # 更新最新偵測
        latest_detections = detections
        
        # 廣播給所有連線的 WebSocket
        await broadcast_detection(detections)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            timestamp=timestamp,
            message=f"已儲存 {len(detections)} 筆偵測資料"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# 類別管理 API
# ========================================

class ClassesRequest(BaseModel):
    """類別設定請求"""
    classes: List[str]


class AddClassRequest(BaseModel):
    """新增類別請求"""
    class_name: str
    class_name_zh: Optional[str] = None


@app.get("/api/classes")
async def get_classes():
    """取得目前偵測類別列表"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    return detector.get_classes()


@app.post("/api/classes")
async def set_classes(request: ClassesRequest):
    """設定要偵測的類別"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    success = detector.set_classes(request.classes)
    if success:
        return {
            "success": True, 
            "message": f"已設定 {len(request.classes)} 個類別",
            "classes": request.classes
        }
    else:
        raise HTTPException(status_code=500, detail="設定類別失敗")


@app.post("/api/classes/add")
async def add_class(request: AddClassRequest):
    """新增單一類別"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    success = detector.add_class(request.class_name, request.class_name_zh)
    if success:
        return {
            "success": True, 
            "message": f"已新增類別: {request.class_name}",
            "classes": detector.custom_classes
        }
    else:
        return {
            "success": False, 
            "message": f"類別 {request.class_name} 已存在"
        }


@app.delete("/api/classes/{class_name}")
async def remove_class(class_name: str):
    """移除類別"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    success = detector.remove_class(class_name)
    if success:
        return {
            "success": True, 
            "message": f"已移除類別: {class_name}",
            "classes": detector.custom_classes
        }
    else:
        return {
            "success": False, 
            "message": f"類別 {class_name} 不存在"
        }


@app.post("/api/classes/reload")
async def reload_classes():
    """重新載入模型類別設定"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    try:
        # 重新設定模型類別
        if detector.model and hasattr(detector.model, 'set_classes'):
            detector.model.set_classes(detector.custom_classes)
            print(f"✅ 模型類別已重新載入: {detector.custom_classes}")
        
        return {
            "success": True, 
            "message": f"模型已重新載入 {len(detector.custom_classes)} 個類別",
            "classes": detector.custom_classes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scheduler/start")
async def start_scheduler():
    """啟動定時偵測"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="排程器未初始化")
    
    scheduler.start()
    return {"success": True, "message": "定時偵測已啟動"}


@app.post("/api/scheduler/stop")
async def stop_scheduler():
    """停止定時偵測"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="排程器未初始化")
    
    scheduler.stop()
    return {"success": True, "message": "定時偵測已停止"}


@app.get("/api/scheduler/status")
async def scheduler_status():
    """取得排程器狀態"""
    if scheduler is None:
        return {"is_running": False, "interval_seconds": 0}
    
    return {
        "is_running": scheduler.is_running,
        "interval_seconds": scheduler.interval_seconds
    }


class IntervalRequest(BaseModel):
    """間隔設定請求"""
    interval: int


@app.post("/api/scheduler/interval")
async def set_scheduler_interval(request: IntervalRequest):
    """設定偵測間隔"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="排程器未初始化")
    
    if request.interval < 5 or request.interval > 300:
        raise HTTPException(status_code=400, detail="間隔必須在 5-300 秒之間")
    
    scheduler.set_interval(request.interval)
    return {
        "success": True, 
        "message": f"偵測間隔已設為 {request.interval} 秒",
        "interval": request.interval
    }


# ========================================
# WebSocket
# ========================================

@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端點，用於即時推送偵測結果"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        while True:
            # 保持連線，等待訊息
            data = await websocket.receive_text()
            
            # 可處理客戶端訊息（如心跳）
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


async def broadcast_detection(detections_input):
    """廣播偵測結果給所有連線的 WebSocket"""
    global latest_detections
    
    # 處理 scheduler 傳入的 tuple (detections, image_path)
    image_path = None
    if isinstance(detections_input, tuple):
        detections = detections_input[0] if detections_input[0] else []
        image_path = detections_input[1] if len(detections_input) > 1 else None
    else:
        detections = detections_input if detections_input else []
    
    # 取得當前攝影機的位置配置
    camera_location = "unknown"
    if detector:
        camera_config = load_camera_config()
        current_camera = str(detector.camera_source)
        if current_camera in camera_config.get("cameras", {}):
            camera_location = camera_config["cameras"][current_camera].get("location", "unknown")
    
    latest_detections = detections
    
    # 轉換為可序列化的格式，並加上位置資訊
    def to_serializable(d):
        if hasattr(d, 'dict'):
            data = d.dict()  # Pydantic model
        elif hasattr(d, 'to_dict'):
            data = d.to_dict()  # dataclass with to_dict
        elif hasattr(d, '__dataclass_fields__'):
            from dataclasses import asdict
            data = asdict(d)  # dataclass
        else:
            data = d if isinstance(d, dict) else {}
        
        # 使用攝影機配置的位置覆蓋 surface
        if camera_location != "unknown":
            data['surface'] = camera_location
        
        # 加上圖片路徑
        if image_path:
            data['image_path'] = image_path
            
        return data
    
    message = json.dumps({
        "type": "detection",
        "data": [to_serializable(d) for d in detections],
        "timestamp": int(datetime.now().timestamp() * 1000)
    })
    
    for ws in connected_websockets.copy():
        try:
            await ws.send_text(message)
        except Exception:
            connected_websockets.remove(ws)


# ========================================
# 靜態檔案服務
# ========================================

# 掛載前端靜態檔案
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
css_path = os.path.join(frontend_path, "css")
js_path = os.path.join(frontend_path, "js")

# 分別掛載 CSS 和 JS 目錄
if os.path.exists(css_path):
    app.mount("/css", StaticFiles(directory=css_path), name="css")
if os.path.exists(js_path):
    app.mount("/js", StaticFiles(directory=js_path), name="js")

# 掛載截圖資料夾
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def serve_frontend():
    """服務前端首頁"""
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/settings")
async def serve_settings():
    """服務設定頁面"""
    return FileResponse(os.path.join(frontend_path, "settings.html"))


# ========================================
# 主程式入口
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

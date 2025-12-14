"""
定時偵測排程器模組
"""

import asyncio
import threading
from typing import Callable, Optional, List, Any
from datetime import datetime


class DetectionScheduler:
    """定時偵測排程器"""
    
    def __init__(
        self,
        detector: Any,
        on_detection: Callable,
        interval_seconds: int = 30
    ):
        self.detector = detector
        self.on_detection = on_detection
        self.interval_seconds = interval_seconds
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def start(self):
        """啟動定時偵測"""
        if self.is_running:
            print("⚠️ 排程器已在運行中")
            return
        
        self.is_running = True
        
        # 在新執行緒中運行事件迴圈
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        print(f"✅ 定時偵測已啟動 (間隔: {self.interval_seconds} 秒)")
    
    def stop(self):
        """停止定時偵測"""
        self.is_running = False
        
        if self._task:
            self._task.cancel()
        
        print("🛑 定時偵測已停止")
    
    def _run_loop(self):
        """在背景執行緒中運行事件迴圈"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._detection_loop())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.close()
    
    async def _detection_loop(self):
        """偵測迴圈"""
        while self.is_running:
            try:
                # 執行偵測
                if self.detector and self.detector.is_ready:
                    detections = await self.detector.detect_snapshot()
                    
                    if detections and self.on_detection:
                        # 呼叫回調函數
                        if asyncio.iscoroutinefunction(self.on_detection):
                            await self.on_detection(detections)
                        else:
                            self.on_detection(detections)
                    
                    print(f"📸 定時偵測完成: {len(detections)} 個物品")
                
            except Exception as e:
                print(f"❌ 定時偵測錯誤: {e}")
            
            # 等待下次偵測
            await asyncio.sleep(self.interval_seconds)
    
    def set_interval(self, seconds: int):
        """設定偵測間隔"""
        self.interval_seconds = max(5, seconds)  # 最少 5 秒
        print(f"⏱️ 偵測間隔已更新為 {self.interval_seconds} 秒")

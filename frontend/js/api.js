/**
 * API 通訊模組
 * 處理與後端偵測服務的通訊
 */

const API_CONFIG = {
    baseUrl: 'http://localhost:8000',
    endpoints: {
        snapshot: '/api/snapshot',
        detections: '/api/detections/latest',
        saveDetection: '/api/detections',
        saveDetectionsBatch: '/api/detections/batch',
        health: '/api/health',
        // 類別管理
        classes: '/api/classes',
        addClass: '/api/classes/add',
        removeClass: '/api/classes',
        // 攝影機管理
        cameras: '/api/cameras'
    },
    timeout: 10000
};

/**
 * ObjectFinderAPI 類別
 * 封裝與後端 API 的通訊
 */
class ObjectFinderAPI {
    constructor(baseUrl = API_CONFIG.baseUrl) {
        this.baseUrl = baseUrl;
        this.isConnected = false;
        this.websocket = null;
    }

    /**
     * 發送 HTTP 請求
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), API_CONFIG.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            clearTimeout(timeout);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeout);
            if (error.name === 'AbortError') {
                throw new Error('請求逾時');
            }
            throw error;
        }
    }

    /**
     * 檢查後端服務狀態
     */
    async checkHealth() {
        try {
            const result = await this.request(API_CONFIG.endpoints.health);
            this.isConnected = true;
            return result;
        } catch (error) {
            this.isConnected = false;
            console.warn('後端服務未連線:', error.message);
            return null;
        }
    }

    /**
     * 手動觸發快照偵測
     */
    async triggerSnapshot() {
        try {
            const result = await this.request(API_CONFIG.endpoints.snapshot, {
                method: 'POST'
            });
            return result;
        } catch (error) {
            console.error('快照觸發失敗:', error);
            throw error;
        }
    }

    /**
     * 取得最新偵測結果
     */
    async getLatestDetections() {
        try {
            const result = await this.request(API_CONFIG.endpoints.detections);
            return result;
        } catch (error) {
            console.error('取得偵測結果失敗:', error);
            throw error;
        }
    }

    /**
     * 儲存單筆偵測資料
     * @param {Object} detection 偵測資料
     */
    async saveDetection(detection) {
        try {
            const result = await this.request(API_CONFIG.endpoints.saveDetection, {
                method: 'POST',
                body: JSON.stringify({
                    object_class: detection.objectClass,
                    confidence: detection.confidence,
                    bbox: detection.bbox || [0, 0, 100, 100],
                    surface: detection.surface,
                    region: detection.region,
                    timestamp: detection.timestamp
                })
            });
            return result;
        } catch (error) {
            console.error('儲存偵測資料失敗:', error);
            throw error;
        }
    }

    /**
     * 批次儲存偵測資料
     * @param {Array} detections 偵測資料陣列
     */
    async saveDetectionsBatch(detections) {
        try {
            const payload = detections.map(d => ({
                object_class: d.objectClass,
                confidence: d.confidence,
                bbox: d.bbox || [0, 0, 100, 100],
                surface: d.surface,
                region: d.region,
                timestamp: d.timestamp
            }));
            
            const result = await this.request(API_CONFIG.endpoints.saveDetectionsBatch, {
                method: 'POST',
                body: JSON.stringify(payload)
            });
            return result;
        } catch (error) {
            console.error('批次儲存偵測資料失敗:', error);
            throw error;
        }
    }

    // ========================================
    // 類別管理 API
    // ========================================

    /**
     * 取得目前偵測類別列表
     */
    async getClasses() {
        try {
            const result = await this.request(API_CONFIG.endpoints.classes);
            return result;
        } catch (error) {
            console.error('取得類別失敗:', error);
            throw error;
        }
    }

    /**
     * 設定要偵測的類別
     * @param {Array} classes 類別名稱陣列
     */
    async setClasses(classes) {
        try {
            const result = await this.request(API_CONFIG.endpoints.classes, {
                method: 'POST',
                body: JSON.stringify({ classes })
            });
            return result;
        } catch (error) {
            console.error('設定類別失敗:', error);
            throw error;
        }
    }

    /**
     * 新增單一類別
     * @param {string} className 類別英文名稱
     * @param {string} classNameZh 類別中文名稱
     */
    async addClass(className, classNameZh = null) {
        try {
            const result = await this.request(API_CONFIG.endpoints.addClass, {
                method: 'POST',
                body: JSON.stringify({ 
                    class_name: className,
                    class_name_zh: classNameZh
                })
            });
            return result;
        } catch (error) {
            console.error('新增類別失敗:', error);
            throw error;
        }
    }

    /**
     * 移除類別
     * @param {string} className 類別名稱
     */
    async removeClass(className) {
        try {
            const result = await this.request(
                `${API_CONFIG.endpoints.removeClass}/${encodeURIComponent(className)}`,
                { method: 'DELETE' }
            );
            return result;
        } catch (error) {
            console.error('移除類別失敗:', error);
            throw error;
        }
    }

    // ========================================
    // 攝影機管理 API
    // ========================================

    /**
     * 取得可用攝影機清單
     */
    async getCameras() {
        try {
            const result = await this.request(API_CONFIG.endpoints.cameras);
            return result;
        } catch (error) {
            console.error('取得攝影機清單失敗:', error);
            throw error;
        }
    }

    /**
     * 設定使用的攝影機
     * @param {number} cameraId 攝影機 ID
     */
    async setCamera(cameraId) {
        try {
            const result = await this.request(
                `${API_CONFIG.endpoints.cameras}/${cameraId}`,
                { method: 'POST' }
            );
            return result;
        } catch (error) {
            console.error('設定攝影機失敗:', error);
            throw error;
        }
    }

    /**
     * 建立 WebSocket 連線以接收即時偵測結果
     */
    connectWebSocket(onMessage, onError, onClose) {
        const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws/detections';
        
        try {
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket 已連線');
                this.isConnected = true;
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (onMessage) onMessage(data);
                } catch (error) {
                    console.error('WebSocket 訊息解析失敗:', error);
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket 錯誤:', error);
                this.isConnected = false;
                if (onError) onError(error);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket 已斷線');
                this.isConnected = false;
                if (onClose) onClose();
            };

        } catch (error) {
            console.error('WebSocket 連線失敗:', error);
            this.isConnected = false;
        }
    }

    /**
     * 斷開 WebSocket 連線
     */
    disconnectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }

    /**
     * 取得連線狀態
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            baseUrl: this.baseUrl
        };
    }
}

// 匯出全域實例
window.objectFinderAPI = new ObjectFinderAPI();

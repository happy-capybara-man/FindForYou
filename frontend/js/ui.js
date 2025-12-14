/**
 * UI 互動模組
 */

class ObjectFinderUI {
    constructor() {
        this.elements = {};
        this.isLoading = false;
    }

    init() {
        this.elements = {
            searchInput: document.getElementById('searchInput'),
            searchBtn: document.getElementById('searchBtn'),
            voiceBtn: document.getElementById('voiceBtn'),
            resultSection: document.getElementById('resultSection'),
            resultIcon: document.getElementById('resultIcon'),
            resultTitle: document.getElementById('resultTitle'),
            resultLocation: document.getElementById('resultLocation'),
            resultTime: document.getElementById('resultTime'),
            confidenceFill: document.getElementById('confidenceFill'),
            confidenceValue: document.getElementById('confidenceValue'),
            quickItemsGrid: document.getElementById('quickItemsGrid'),
            recentList: document.getElementById('recentList'),
            manualScanBtn: document.getElementById('manualScanBtn'),
            historyBtn: document.getElementById('historyBtn'),
            settingsBtn: document.getElementById('settingsBtn'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            toastContainer: document.getElementById('toastContainer'),
            statusIndicator: document.getElementById('statusIndicator')
        };
        console.log('UI 初始化完成');
    }

    showLoading(text = '搜尋中...') {
        this.isLoading = true;
        const overlay = this.elements.loadingOverlay;
        const loadingText = overlay.querySelector('.loading-text');
        if (loadingText) loadingText.textContent = text;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        this.isLoading = false;
        this.elements.loadingOverlay.style.display = 'none';
    }

    showToast(message, type = 'info', duration = 3000) {
        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span class="toast-icon">${icons[type]}</span><span class="toast-message">${message}</span>`;
        this.elements.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = 'toastIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    showResult(result) {
        const section = this.elements.resultSection;
        this.elements.resultIcon.textContent = '📍';
        this.elements.resultTitle.textContent = `找到 ${result.objectClassZh}！`;
        this.elements.resultLocation.textContent = result.description || `${result.objectClassZh}在${result.surfaceZh}${result.regionZh}`;
        this.elements.resultTime.textContent = `最後看到時間：${this.formatTimeAgo(result.lastSeen)}`;
        const confidence = Math.round((result.confidence || 0.9) * 100);
        this.elements.confidenceFill.style.width = `${confidence}%`;
        this.elements.confidenceValue.textContent = `${confidence}%`;
        
        // 如果有截圖，顯示在結果區
        const existingImg = section.querySelector('.result-image');
        if (existingImg) existingImg.remove();
        
        if (result.imagePath) {
            const imgContainer = document.createElement('div');
            imgContainer.className = 'result-image';
            imgContainer.style.cssText = 'margin-top: 15px; cursor: pointer;';
            imgContainer.innerHTML = `
                <img src="${result.imagePath}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);" alt="偵測截圖">
                <p style="text-align:center; color:#aaa; font-size:12px; margin-top:8px;">點擊放大</p>
            `;
            imgContainer.addEventListener('click', () => this.showSnapshot(result.imagePath));
            section.querySelector('.result-card').appendChild(imgContainer);
        }
        
        section.style.display = 'block';
        section.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    showNotFound(query) {
        const section = this.elements.resultSection;
        this.elements.resultIcon.textContent = '🤔';
        this.elements.resultTitle.textContent = `找不到「${query}」`;
        this.elements.resultLocation.textContent = '系統目前沒有這個物品的記錄';
        this.elements.resultTime.textContent = '請確認物品名稱，或等待下次偵測';
        this.elements.confidenceFill.style.width = '0%';
        this.elements.confidenceValue.textContent = '--';
        section.style.display = 'block';
    }

    hideResult() {
        this.elements.resultSection.style.display = 'none';
    }

    updateRecentList(detections) {
        const container = this.elements.recentList;
        if (!detections || detections.length === 0) {
            container.innerHTML = `<div class="empty-state"><span class="empty-icon">📭</span><p>尚無偵測記錄</p></div>`;
            return;
        }
        container.innerHTML = detections.map((d, index) => `
            <div class="recent-item clickable" data-index="${index}" data-class="${d.objectClass}" 
                 data-class-zh="${d.objectClassZh}" data-surface="${d.surfaceZh}" 
                 data-region="${d.regionZh}" data-time="${d.timestamp}" 
                 data-confidence="${d.confidence || 0.9}" data-image="${d.imagePath || ''}"
                 style="cursor: pointer;">
                <span class="recent-item-icon">${this.getObjectIcon(d.objectClass)}</span>
                <div class="recent-item-info">
                    <div class="recent-item-name">${d.objectClassZh}</div>
                    <div class="recent-item-location">${d.surfaceZh} ${d.regionZh}</div>
                </div>
                <div class="recent-item-time">${this.formatTimeAgo(d.timestamp)}</div>
            </div>
        `).join('');
        
        // 儲存偵測資料供點擊使用
        this.recentDetections = detections;
    }

    updateStatus(isConnected, message = null) {
        const dot = this.elements.statusIndicator.querySelector('.status-dot');
        const text = this.elements.statusIndicator.querySelector('.status-text');
        dot.style.background = isConnected ? '#38ef7d' : '#f5576c';
        text.textContent = message || (isConnected ? '系統就緒' : '離線模式');
    }

    formatTimeAgo(timestamp) {
        const diff = Date.now() - timestamp;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        if (days > 0) return `${days} 天前`;
        if (hours > 0) return `${hours} 小時前`;
        if (minutes > 0) return `${minutes} 分鐘前`;
        return '剛剛';
    }

    getObjectIcon(objectClass) {
        const icons = { 'cell phone': '📱', 'phone': '📱', 'remote': '📺', 'bottle': '🍶', 'cup': '☕', 'book': '📖', 'glasses': '👓', 'keys': '🔑', 'wallet': '👛', 'headphones': '🎧', 'watch': '⌚', 'bag': '👜', 'umbrella': '🌂' };
        return icons[objectClass.toLowerCase()] || '📦';
    }

    showSnapshot(imagePath) {
        // 移除舊的截圖模態框
        const existing = document.getElementById('snapshotModal');
        if (existing) existing.remove();
        
        // 建立模態框
        const modal = document.createElement('div');
        modal.id = 'snapshotModal';
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85); z-index: 9999;
            display: flex; align-items: center; justify-content: center;
            animation: fadeIn 0.3s ease;
        `;
        
        modal.innerHTML = `
            <div style="
                background: var(--glass-bg, rgba(30,30,50,0.9));
                border-radius: 16px; padding: 20px;
                max-width: 90%; max-height: 90%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                    <h3 style="margin:0; color:#fff;">📸 掃描截圖</h3>
                    <button id="closeSnapshot" style="
                        background: #f5576c; border: none; color: #fff;
                        padding: 8px 16px; border-radius: 8px; cursor: pointer;
                        font-size: 14px;
                    ">關閉</button>
                </div>
                <img src="${imagePath}" style="
                    max-width: 100%; max-height: 70vh;
                    border-radius: 8px; display: block;
                " alt="掃描截圖">
                <p style="text-align:center; color:#aaa; margin-top:10px; font-size:14px;">
                    綠色框為偵測到的物品
                </p>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // 關閉按鈕
        document.getElementById('closeSnapshot').addEventListener('click', () => modal.remove());
        
        // 點擊背景關閉
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    setSearchValue(value) { this.elements.searchInput.value = value; }
    getSearchValue() { return this.elements.searchInput.value.trim(); }
    clearSearch() { this.elements.searchInput.value = ''; this.hideResult(); }
}

window.objectFinderUI = new ObjectFinderUI();

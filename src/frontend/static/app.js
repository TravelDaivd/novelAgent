// ============================================================
// 小说智能分析系统 - 前端交互
// ============================================================

// ---------- 配置 ----------
const API_BASE = 'http://localhost:8000/api/v1';

// ---------- DOM 引用 ----------
const messages = document.getElementById('messages');
const emptyState = document.getElementById('emptyState');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const uploadBtn = document.getElementById('uploadBtn');
const newChatBtn = document.getElementById('newChatBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

const uploadModal = document.getElementById('uploadModal');
const closeModal = document.getElementById('closeModal');
const cancelUpload = document.getElementById('cancelUpload');
const confirmUpload = document.getElementById('confirmUpload');
const chapterId = document.getElementById('chapterId');
const chapterTitle = document.getElementById('chapterTitle');
const chapterContent = document.getElementById('chapterContent');
const fileInput = document.getElementById('fileInput');

// ---------- 状态 ----------
let isProcessing = false;
let currentChapterId = null;
let messageCount = 0;
const history = [];

// ---------- 工具函数 ----------
function setStatus(loading, text) {
    statusDot.className = 'status-dot' + (loading ? ' loading' : '');
    statusText.textContent = text;
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;
}

function getCurrentTime() {
    return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

// ---------- 渲染消息 ----------
function addMessage(role, content, evidence = null) {
    emptyState.style.display = 'none';
    
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 渲染内容（支持换行）
    const textP = document.createElement('div');
    textP.innerHTML = content.replace(/\n/g, '<br>');
    contentDiv.appendChild(textP);
    
    // 如果有证据
    if (evidence && evidence.length > 0) {
        const details = document.createElement('details');
        details.className = 'evidence';
        const summary = document.createElement('summary');
        summary.textContent = `📎 查看证据 (${evidence.length} 个片段)`;
        details.appendChild(summary);
        
        evidence.forEach((item, idx) => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'evidence-item';
            itemDiv.innerHTML = `
                <div class="label">片段 ${idx + 1} · 类型: ${item.plot_type || '未知'} · 置信度: ${(item.score || 0.5).toFixed(2)}</div>
                <div>${item.text || ''}</div>
            `;
            details.appendChild(itemDiv);
        });
        
        contentDiv.appendChild(details);
    }
    
    div.appendChild(avatar);
    div.appendChild(contentDiv);
    
    messages.appendChild(div);
    messageCount++;
    scrollToBottom();
}

function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
    
    div.appendChild(avatar);
    div.appendChild(content);
    messages.appendChild(div);
    scrollToBottom();
    return div;
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

// ---------- API 调用 ----------
async function callAPI(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '请求失败');
    }
    
    return await response.json();
}

// ---------- 发送问题 ----------
async function sendQuestion() {
    const question = userInput.value.trim();
    if (!question || isProcessing) return;
    
    // 添加用户消息
    addMessage('user', question);
    userInput.value = '';
    sendBtn.disabled = true;
    isProcessing = true;
    setStatus(true, '思考中...');
    
    // 添加打字效果
    addTypingIndicator();
    
    try {
        const data = await callAPI('/chat', {
            question: question,
            chapter_id: currentChapterId,
        });
        
        removeTypingIndicator();
        addMessage('assistant', data.answer, data.evidence);
        setStatus(false, '就绪');
        
    } catch (error) {
        removeTypingIndicator();
        addMessage('assistant', `❌ 出错了：${error.message}`);
        setStatus(false, '错误');
    }
    
    sendBtn.disabled = false;
    isProcessing = false;
}

// ---------- 上传章节 ----------
async function uploadChapter() {
    const id = parseInt(chapterId.value);
    const title = chapterTitle.value.trim();
    const content = chapterContent.value.trim();
    
    if (!id || !title || !content) {
        alert('请填写完整的章节信息');
        return;
    }
    
    confirmUpload.disabled = true;
    confirmUpload.textContent = '分析中...';
    setStatus(true, '正在分析章节...');
    
    try {
        const data = await callAPI('/upload', {
            chapter_id: id,
            title: title,
            content: content,
        });
        
        currentChapterId = id;
        
        // 添加到历史
        history.push({ id, title, time: getCurrentTime() });
        renderHistory();
        
        // 关闭弹窗
        closeUploadModal();
        
        // 清空输入
        chapterId.value = '';
        chapterTitle.value = '';
        chapterContent.value = '';
        
        // 显示结果
        addMessage('assistant', `
            ✅ 章节「${title}」分析完成！
            
            📊 统计信息：
            • 章节 ID：${data.chapter_id}
            • 片段数量：${data.segment_count}
            • 识别实体：${data.entity_count} 个
            • 抽取关系：${data.relation_count} 对
            
            💡 现在你可以提问关于本章的问题了。
        `);
        
        setStatus(false, `已加载: ${title}`);
        
    } catch (error) {
        alert(`上传失败：${error.message}`);
        setStatus(false, '上传失败');
    }
    
    confirmUpload.disabled = false;
    confirmUpload.textContent = '开始分析';
}

// ---------- 弹窗控制 ----------
function openUploadModal() {
    uploadModal.classList.add('active');
}

function closeUploadModal() {
    uploadModal.classList.remove('active');
}

// ---------- 历史记录 ----------
function renderHistory() {
    const list = document.getElementById('historyList');
    list.innerHTML = history.map((item, idx) => `
        <li onclick="loadHistory(${idx})">
            📄 ${item.title}
            <span style="color: var(--text-muted); font-size: 12px; margin-left: 8px;">${item.time}</span>
        </li>
    `).join('');
}

function loadHistory(idx) {
    const item = history[idx];
    currentChapterId = item.id;
    document.getElementById('currentTitle').textContent = item.title;
    setStatus(false, `已加载: ${item.title}`);
    addMessage('assistant', `已切换到「${item.title}」，你可以继续提问。`);
}

// ---------- 文件读取 ----------
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        chapterContent.value = event.target.result;
    };
    reader.readAsText(file, 'UTF-8');
});

// ---------- 快捷键 ----------
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

// ---------- 事件绑定 ----------
sendBtn.addEventListener('click', sendQuestion);
uploadBtn.addEventListener('click', openUploadModal);
newChatBtn.addEventListener('click', () => {
    messages.innerHTML = '';
    emptyState.style.display = 'flex';
    currentChapterId = null;
    document.getElementById('currentTitle').textContent = '小说智能分析';
    setStatus(false, '系统就绪');
});

closeModal.addEventListener('click', closeUploadModal);
cancelUpload.addEventListener('click', closeUploadModal);
uploadModal.addEventListener('click', (e) => {
    if (e.target === uploadModal) closeUploadModal();
});

confirmUpload.addEventListener('click', uploadChapter);

// ---------- 启动就绪 ----------
setStatus(false, '系统就绪');
console.log('📖 小说智能分析系统已加载');
console.log(`📍 API 地址: ${API_BASE}`);
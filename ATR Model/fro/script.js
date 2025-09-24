(function () {
    // Central API base (match the running backend URL)
    const API_BASE = 'https://atr-backend-production-1c4f.up.railway.app';
    const apiStatusBadge = document.getElementById('apiStatusBadge');
    const themeToggle = document.getElementById('themeToggle');
    const ttsToggle = document.getElementById('ttsToggle');
    const toastContainer = document.getElementById('toastContainer');
    // Cross-browser fetch with timeout helper
    async function fetchWithTimeout(url, options, timeoutMs) {
        try {
            if (AbortSignal && typeof AbortSignal.timeout === 'function') {
                return await fetch(url, { ...(options||{}), signal: AbortSignal.timeout(timeoutMs) });
            }
        } catch (_) {}
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeoutMs);
        try {
            return await fetch(url, { ...(options||{}), signal: controller.signal });
        } finally {
            clearTimeout(id);
        }
    }

    // Theme persistence
    try {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') document.documentElement.classList.add('theme-light');
        themeToggle && (themeToggle.textContent = savedTheme === 'light' ? '🌞' : '🌓');
        themeToggle && themeToggle.addEventListener('click', () => {
            const isLight = document.documentElement.classList.toggle('theme-light');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            themeToggle.textContent = isLight ? '🌞' : '🌓';
        });
    } catch (_) {}

    // API health polling
    async function pollHealth() {
        try {
            const res = await fetchWithTimeout(API_BASE + '/progress', { method: 'GET' }, 6000);
            if (res && res.ok) {
                apiStatusBadge && (apiStatusBadge.textContent = 'Online', apiStatusBadge.className = 'badge badge--ok');
            } else {
                apiStatusBadge && (apiStatusBadge.textContent = 'Degraded', apiStatusBadge.className = 'badge badge--warn');
            }
        } catch (e) {
            apiStatusBadge && (apiStatusBadge.textContent = 'Offline', apiStatusBadge.className = 'badge badge--err');
        }
        setTimeout(pollHealth, 8000);
    }
    apiStatusBadge && pollHealth();
    // Modal removed - interface is now direct

    const tabs = Array.from(document.querySelectorAll('.tab')); // legacy tabs (index.html)
    const panels = Array.from(document.querySelectorAll('.tabpanel'));
    const sideItems = Array.from(document.querySelectorAll('.side-item'));
    const sidebarFoldToggle = document.getElementById('sidebarFoldToggle');
    const appLayout = document.getElementById('appLayout');
    const sidebarFoldFloating = document.getElementById('sidebarFoldFloating');

    const fileInput = document.getElementById('audioFile');
    const dropArea = document.getElementById('dropArea');
    const fileInfo = document.getElementById('fileInfo');
    const uploadAudio = document.getElementById('uploadAudio');

    // Toasts
    function showToast(message, kind) {
        if (!toastContainer) return;
        try {
            const el = document.createElement('div');
            el.className = 'toast' + (kind === 'err' ? ' toast--err' : ' toast--ok');
            el.textContent = message;
            toastContainer.appendChild(el);
            setTimeout(() => { el.style.opacity = '0'; }, 3000);
            setTimeout(() => { el.remove(); }, 3600);
        } catch (_) {}
    }

    // Phase 3 elements
    const recStartBtn = document.getElementById('recStartBtn');
    const recStopBtn = document.getElementById('recStopBtn');
    const recStatus = document.getElementById('recStatus');
    const recordedPreview = document.getElementById('recordedPreview');
    const textPrompt = document.getElementById('textPrompt');
    const submitInteractBtn = document.getElementById('submitInteractBtn');
    const multiAskBtn = document.getElementById('multiAskBtn');
    const multiStopBtn = null;
    const regenerateBtn = null;
    const composerCopyBtn = null;
    const multiAskFromAudioBtn = document.getElementById('multiAskFromAudioBtn');
    const fileSelect = document.getElementById('fileSelect');
    const fileSelectStatus = document.getElementById('fileSelectStatus');
    const fileSelectQa = document.getElementById('fileSelectQa');
    const fileSelectStatusQa = document.getElementById('fileSelectStatusQa');
    const multiPrompt = document.getElementById('multiPrompt');
    const qaMessages = document.getElementById('qaMessages');
    const multiResults = document.getElementById('multiResults');
    const multiCombinedAnswer = document.getElementById('multiCombinedAnswer');
    const multiChatAnswer = document.getElementById('multiChatAnswer');
    const summaryBox = document.getElementById('summaryBox');
    const summarizeBtn = document.getElementById('summarizeBtn');
    const summaryLength = document.getElementById('summaryLength');
    const summaryPlayBtn = document.getElementById('summaryPlayBtn');
    const summaryDownloadWavBtn = document.getElementById('summaryDownloadWavBtn');
    const summaryDownloadTxtBtn = document.getElementById('summaryDownloadTxtBtn');
    // Multi answers controls
    const multiAnswerPlayBtn = document.getElementById('multiAnswerPlayBtn');
    const multiAnswerDownloadWavBtn = document.getElementById('multiAnswerDownloadWavBtn');
    const multiAnswerDownloadTxtBtn = document.getElementById('multiAnswerDownloadTxtBtn');
    const multiAnswerAudio = document.getElementById('multiAnswerAudio');
    let lastMultiItems = [];
    const summaryCopyBtn = document.getElementById('summaryCopyBtn');
    const summaryClearBtn = document.getElementById('summaryClearBtn');
    const summaryCount = document.getElementById('summaryCount');
    const summaryAudio = document.getElementById('summaryAudio');
    const submitStatus = document.getElementById('submitStatus');
    const responseText = null;
    const responseAudio = null;

    // Training elements
    const startRealTrainBtn = document.getElementById('startRealTrainBtn');
    const trainStatus = document.getElementById('trainStatus');
    const trainProgressBar = document.getElementById('trainProgressBar');
    const trainProgressText = document.getElementById('trainProgressText');
    const trainArtifact = document.getElementById('trainArtifact');
    const trainStages = document.getElementById('trainStages');
    let trainPollTimer = null;
    // Toggle empty vs conversation state for Q&A
    (function setupQaEmptyState(){
        const qaPanel = (qaMessages && qaMessages.closest('.qa-chat')) || null;
        function updateEmptyState(){
            if (!qaPanel || !qaMessages) return;
            const hasMsg = qaMessages && qaMessages.children && qaMessages.children.length > 0;
            if (hasMsg) qaPanel.classList.remove('chat-empty');
            else qaPanel.classList.add('chat-empty');
        }
        // Initial
        updateEmptyState();
        // Observe mutations in message list
        try {
            const mo = new MutationObserver(updateEmptyState);
            qaMessages && mo.observe(qaMessages, { childList: true });
        } catch(_) {}
        // Also update when user switches tabs to Q&A
        sideItems.forEach(function(btn){
            btn.addEventListener('click', function(){
                if (this.getAttribute('data-panel') === 'qa') updateEmptyState();
            });
        });

        // In empty state, auto-focus composer so user can type immediately
        try { if (multiPrompt) multiPrompt.focus(); } catch(_) {}
    })();

    // ChatGPT-like composer: Enter to send, Shift+Enter for newline, autoresize
    (function setupComposerBehaviors(){
        if (!multiPrompt) return;
        const maxRows = 6;
        function autoResize(){
            try {
                multiPrompt.style.height = 'auto';
                const lineHeight = 24; // approx
                const lines = Math.min(maxRows, Math.ceil(multiPrompt.scrollHeight / lineHeight));
                multiPrompt.style.height = Math.min(multiPrompt.scrollHeight, maxRows * lineHeight + 6) + 'px';
            } catch(_) {}
        }
        multiPrompt.addEventListener('input', autoResize);
        autoResize();

        multiPrompt.addEventListener('keydown', function(e){
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (multiAskBtn && !multiAskBtn.disabled) {
                    multiAskBtn.click();
                }
            }
        });
    })();

    // Keep messages scrolled to bottom as new content appears
    (function keepChatScrolled(){
        if (!qaMessages) return;
        function toBottom(){ try { qaMessages.scrollTop = qaMessages.scrollHeight; } catch(_) {} }
        try {
            const mo = new MutationObserver(toBottom);
            mo.observe(qaMessages, { childList: true, subtree: true });
        } catch(_) {}
    })();


    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedBlob = null;

    // Modal functions removed - interface is now direct

    // Tabs
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            const target = this.getAttribute('data-tab');
            tabs.forEach(function (t) { t.classList.remove('tab--active'); t.setAttribute('aria-selected', 'false'); });
            panels.forEach(function (p) { p.classList.remove('tabpanel--active'); });
            this.classList.add('tab--active');
            this.setAttribute('aria-selected', 'true');
            const panel = document.querySelector('.tabpanel[data-panel="' + target + '"]');
            if (panel) panel.classList.add('tabpanel--active');
        });
    });

    // Sidebar switcher
    sideItems.forEach(function (btn) {
        btn.addEventListener('click', function(){
            const target = this.getAttribute('data-panel');
            sideItems.forEach(b => b.classList.remove('side-item--active'));
            this.classList.add('side-item--active');
            panels.forEach(function (p) {
                if (p.getAttribute('data-panel') === target) p.classList.add('tabpanel--active');
                else p.classList.remove('tabpanel--active');
            });
            if (target === 'summary' || target === 'qa') { loadAudioFiles && loadAudioFiles(); }
        });
    });

    // Sidebar fold toggle
    function updateSidebarFabVisibility() {
        if (!sidebarFoldFloating || !appLayout) return;
        const folded = appLayout.classList.contains('layout--sidebar-folded');
        sidebarFoldFloating.style.display = folded ? '' : 'none';
    }

    sidebarFoldToggle && sidebarFoldToggle.addEventListener('click', function(){
        if (!appLayout) return;
        appLayout.classList.toggle('layout--sidebar-folded');
        updateSidebarFabVisibility();
    });
    sidebarFoldFloating && sidebarFoldFloating.addEventListener('click', function(){
        if (!appLayout) return;
        appLayout.classList.remove('layout--sidebar-folded');
        updateSidebarFabVisibility();
    });

    // Upload logic
    // Drag & Drop
    if (dropArea) {
        ['dragenter','dragover'].forEach(evt => dropArea.addEventListener(evt, (e) => {
            e.preventDefault(); e.stopPropagation(); dropArea.classList.add('drop-area--over');
        }));
        ['dragleave','drop'].forEach(evt => dropArea.addEventListener(evt, (e) => {
            e.preventDefault(); e.stopPropagation(); dropArea.classList.remove('drop-area--over');
        }));
        dropArea.addEventListener('click', () => fileInput && fileInput.click());
        dropArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer; if (!dt || !dt.files || !dt.files.length) return;
            const file = dt.files[0];
            const list = new DataTransfer(); list.items.add(file); fileInput.files = list.files;
            fileInput.dispatchEvent(new Event('change'));
        });
    }
    if (fileInput) fileInput.addEventListener('change', function () {
        const file = this.files && this.files[0];
        
        // Reset upload button state when new file is selected
        (function(){
            const uploadBtn = document.getElementById('uploadBtn');
            if (uploadBtn) {
        uploadBtn.textContent = 'Upload';
        uploadBtn.style.backgroundColor = '';
        uploadBtn.disabled = false;
            }
        })();
        
        if (!file) {
            fileInfo.textContent = 'No file chosen.';
            uploadAudio.removeAttribute('src');
            uploadAudio.load();
            return;
        }

        const valid = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/m4a', 'audio/x-m4a'];
        if (file.type && !valid.includes(file.type)) {
            fileInfo.textContent = 'Unsupported file type: ' + file.type;
            uploadAudio.removeAttribute('src');
            uploadAudio.load();
            return;
        }

        fileInfo.textContent = 'Selected: ' + file.name;
        const objectUrl = URL.createObjectURL(file);
        uploadAudio.src = objectUrl;
        uploadAudio.load();
    });

    // Upload button functionality with retry logic
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) uploadBtn.addEventListener('click', async function () {
        const file = fileInput.files && fileInput.files[0];
        if (!file) {
            fileInfo.textContent = 'Please select a file first.';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Retry logic
        let retryCount = 0;
        const maxRetries = 3;
        
        const attemptUpload = async () => {
            try {
                fileInfo.textContent = `Uploading... (Attempt ${retryCount + 1}/${maxRetries})`;
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';
                
                // First check if backend is available
                if (retryCount === 0) {
                    try {
                        const healthCheck = await fetchWithTimeout(API_BASE + '/progress', {
                            method: 'GET'
                        }, 8000); // 8s timeout for health check
                        if (!healthCheck.ok) {
                            throw new Error('Backend not responding');
                        }
                    } catch (healthError) {
                        throw new Error('Backend server not available. Please ensure the backend is running.');
                    }
                }
                
                const response = await fetchWithTimeout(API_BASE + '/upload', {
                    method: 'POST',
                    body: formData,
                }, 120000); // 120s timeout

                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.status === 'already_trained') {
                        fileInfo.textContent = data.message || 'File already trained!';
                        uploadBtn.textContent = 'Already Trained ✓';
                        uploadBtn.style.backgroundColor = '#17a2b8'; // Blue color for already trained
                    } else {
                        fileInfo.textContent = data.message || 'Upload successful!';
                        uploadBtn.textContent = 'Uploaded ✓';
                        uploadBtn.style.backgroundColor = '#28a745'; // Green color for new upload
                    }
                    showToast('Upload successful', 'ok');
                    return true; // Success
                } else {
                    const errorData = await response.json().catch(() => ({ error: response.statusText }));
                    throw new Error(errorData.error || response.statusText);
                }
            } catch (error) {
                retryCount++;
                console.log(`Upload attempt ${retryCount} failed:`, error.message);
                
                if (retryCount < maxRetries) {
                    // Wait before retry (exponential backoff)
                    const delay = Math.pow(2, retryCount) * 1000; // 2s, 4s, 8s
                    fileInfo.textContent = `Upload failed, retrying in ${delay/1000}s... (${retryCount}/${maxRetries})`;
                    await new Promise(resolve => setTimeout(resolve, delay));
                    return attemptUpload(); // Retry
                } else {
                    // All retries failed
                    fileInfo.textContent = 'Upload failed after ' + maxRetries + ' attempts: ' + error.message;
                    uploadBtn.textContent = 'Upload Failed';
                    uploadBtn.style.backgroundColor = '#dc3545';
                    showToast('Upload failed: ' + error.message, 'err');
                    return false;
                }
            } finally {
                if (retryCount >= maxRetries) {
                    uploadBtn.disabled = false;
                    // Reset button after 5 seconds
                    setTimeout(() => {
                        uploadBtn.textContent = 'Upload';
                        uploadBtn.style.backgroundColor = '';
                    }, 5000);
                }
            }
        };

        // Start upload attempt
        await attemptUpload();
    });

    // Recording logic
    async function startRecording() {
        recordedChunks = [];
        recordedBlob = null;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = function (e) {
                if (e.data && e.data.size > 0) recordedChunks.push(e.data);
            };
            mediaRecorder.onstop = function () {
                recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
                const url = URL.createObjectURL(recordedBlob);
                recordedPreview.src = url;
                recordedPreview.load();
                recStatus.textContent = 'Recorded';
            };
            mediaRecorder.start();
            recStatus.textContent = 'Recording...';
            recStartBtn.disabled = true;
            recStopBtn.disabled = false;
        } catch (err) {
            recStatus.textContent = 'Mic access denied or unavailable';
            console.error(err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            recStartBtn.disabled = false;
            recStopBtn.disabled = true;
        }
    }

    recStartBtn && recStartBtn.addEventListener('click', startRecording);
    recStopBtn && recStopBtn.addEventListener('click', stopRecording);

    // Submit to backend /interact
    submitInteractBtn && submitInteractBtn.addEventListener('click', async function () {
        submitStatus.textContent = 'Sending...';
            // Response tab removed; no-op

        const form = new FormData();
        const text = (textPrompt && textPrompt.value || '').trim();
        if (text) form.append('text', text);
        if (!text && recordedBlob) {
            // File name with extension expected by backend, use webm; backend accepts any for demo
            form.append('audio', recordedBlob, 'recording.webm');
        }

        if (!text && !recordedBlob) {
            submitStatus.textContent = 'Provide text or record audio first.';
            return;
        }

        try {
            const res = await fetch(API_BASE + '/interact', {
                method: 'POST',
                body: form
            });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            // Response tab removed; optionally show toast only
            submitStatus.textContent = 'Done';
            showToast('Response ready', 'ok');
        } catch (e) {
            console.error(e);
            submitStatus.textContent = 'Failed: ' + e.message;
            showToast('Interact failed: ' + e.message, 'err');
        }
    });

    // Summarize latest transcript
    summarizeBtn && summarizeBtn.addEventListener('click', async function () {
        summarizeBtn.disabled = true;
        try {
            const form = new FormData();
            const tid = (fileSelectQa && fileSelectQa.value) || (fileSelect && fileSelect.value) || '';
            if (tid) form.append('transcript_id', tid);
            const res = await fetch(API_BASE + '/summarize-latest', { method: 'POST', body: form });
            const data = await res.json();
            if (res.ok) {
                summaryBox.value = data.summary || '';
                showToast(data.cached ? 'Loaded cached summary' : 'Summary generated', 'ok');
            } else {
                showToast('Summarize failed: ' + (data.error || ('HTTP ' + res.status)), 'err');
            }
        } catch (e) {
            showToast('Summarize failed: ' + e.message, 'err');
        } finally {
            summarizeBtn.disabled = false;
        }
    });

    // Summary helpers
    function updateSummaryCount() {
        if (!summaryBox || !summaryCount) return;
        const text = (summaryBox.value || '').trim();
        summaryCount.textContent = text ? `(${text.length} chars)` : '';
    }
    summaryBox && summaryBox.addEventListener('input', updateSummaryCount);
    updateSummaryCount();
    summaryCopyBtn && summaryCopyBtn.addEventListener('click', async function(){
        try { await navigator.clipboard.writeText(summaryBox.value || ''); showToast('Summary copied', 'ok'); }
        catch(e){ showToast('Copy failed', 'err'); }
    });
    summaryClearBtn && summaryClearBtn.addEventListener('click', function(){
        summaryBox.value = ''; updateSummaryCount(); showToast('Cleared', 'ok');
    });

    // Client-side TTS helper (SpeechSynthesis)
    function isTtsEnabled(){
        try { return localStorage.getItem('ttsEnabled') !== '0'; } catch(_) { return true; }
    }

    function speakText(text){
        try {
            if (!isTtsEnabled()) return false;
            if (!text || !window.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') return false;
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 1.0; // default speed
            u.pitch = 1.0;
            u.volume = 1.0;
            window.speechSynthesis.speak(u);
            return true;
        } catch(_) { return false; }
    }

    // TTS toggle persistence
    (function setupTtsToggle(){
        if (!ttsToggle) return;
        try {
            const enabled = isTtsEnabled();
            ttsToggle.checked = !!enabled;
            ttsToggle.addEventListener('change', function(){
                try { localStorage.setItem('ttsEnabled', this.checked ? '1' : '0'); } catch(_) {}
                try { if (!this.checked && window.speechSynthesis) window.speechSynthesis.cancel(); } catch(_) {}
            });
        } catch(_) {}
    })();

    // Summary: Play via client-side TTS (fallback to backend if unavailable)
    summaryPlayBtn && summaryPlayBtn.addEventListener('click', async function(){
        const text = (summaryBox && summaryBox.value || '').trim();
        if (!text) return;
        if (speakText(text)) return;
        try {
            const form = new FormData();
            form.append('text', text);
            const res = await fetch(API_BASE + '/tts', { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
            if (data.audio_b64_wav) {
                const wavBlob = b64ToBlob(data.audio_b64_wav, 'audio/wav');
                const url = URL.createObjectURL(wavBlob);
                if (summaryAudio) {
                    summaryAudio.src = url;
                    summaryAudio.load();
                    summaryAudio.play();
                }
            }
        } catch (e) { }
    });

    // Multi Answer: Play via client-side TTS (fallback to backend)
    multiAnswerPlayBtn && multiAnswerPlayBtn.addEventListener('click', async function(){
        const text = (multiCombinedAnswer && multiCombinedAnswer.value || '').trim();
        if (!text) return;
        if (speakText(text)) return;
        try {
            const form = new FormData();
            form.append('text', text);
            const res = await fetch(API_BASE + '/tts', { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
            if (data.audio_b64_wav) {
                const wavBlob = b64ToBlob(data.audio_b64_wav, 'audio/wav');
                const url = URL.createObjectURL(wavBlob);
                if (multiAnswerAudio) {
                    multiAnswerAudio.src = url;
                    multiAnswerAudio.load();
                    multiAnswerAudio.play();
                }
            }
        } catch (e) { }
    });

    // Multi Answer: Download WAV
    multiAnswerDownloadWavBtn && multiAnswerDownloadWavBtn.addEventListener('click', async function(){
        const text = (multiCombinedAnswer && multiCombinedAnswer.value || '').trim();
        if (!text) return;
        try {
            const form = new FormData();
            form.append('text', text);
            const res = await fetch(API_BASE + '/tts', { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
            if (data.audio_b64_wav) {
                const wavBlob = b64ToBlob(data.audio_b64_wav, 'audio/wav');
                const url = URL.createObjectURL(wavBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'answers.wav';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        } catch (e) { }
    });

    // Multi Answer: Download TXT
    multiAnswerDownloadTxtBtn && multiAnswerDownloadTxtBtn.addEventListener('click', function(){
        const items = Array.isArray(lastMultiItems) ? lastMultiItems : [];
        let payload = '';
        if (items.length) {
            try {
                payload = items
                    .map((it, i) => {
                        const q = it && it.q ? String(it.q).trim() : '';
                        const a = it && it.a ? String(it.a).trim() : '';
                        return (q ? `Q${i+1}: ${q}` : '') + (a ? `\nA${i+1}: ${a}` : '');
                    })
                    .filter(Boolean)
                    .join('\n\n');
            } catch(_) {}
        }
        if (!payload) {
            const text = (multiCombinedAnswer && multiCombinedAnswer.value || '').trim();
        if (!text) return;
            payload = text;
        }
        const url = URL.createObjectURL(new Blob([payload], { type: 'text/plain;charset=utf-8' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = 'qa_answers.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // Q&A message-level actions (icons)
    qaMessages && qaMessages.addEventListener('click', async function(e){
        const target = e.target;
        if (!target || !target.getAttribute) return;
        const action = target.getAttribute('data-action');
        if (!action) return;
        const text = (target.getAttribute('data-text') || '').trim();
        if (!text) return;
        if (action === 'mini-play') {
            if (speakText(text)) return;
            try {
                const form = new FormData();
                form.append('text', text);
                const res = await fetch(API_BASE + '/tts', { method: 'POST', body: form });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
                if (data.audio_b64_wav) {
                    const wavBlob = b64ToBlob(data.audio_b64_wav, 'audio/wav');
                    const url = URL.createObjectURL(wavBlob);
                    const audio = new Audio(url);
                    audio.play();
                }
            } catch (_) {}
        } else if (action === 'copy-pair') {
            // Find previous user message to build Q&A pair
            const msgEl = target.closest('.msg');
            let userText = '';
            if (msgEl) {
                let prev = msgEl.previousElementSibling;
                while (prev && !prev.classList.contains('msg')) prev = prev.previousElementSibling;
                if (prev && prev.classList.contains('msg--user')) {
                    const bubble = prev.querySelector('.msg__bubble');
                    userText = bubble ? (bubble.textContent || '').trim() : '';
                }
            }
            const qaText = (userText ? ('Q: ' + userText + '\n\n') : '') + 'A: ' + text;
            try { await navigator.clipboard.writeText(qaText); showToast && showToast('Copied', 'ok'); } catch(_) {}
        }
    });

    // Summary: Download WAV
    summaryDownloadWavBtn && summaryDownloadWavBtn.addEventListener('click', async function(){
        const text = (summaryBox && summaryBox.value || '').trim();
        if (!text) return;
        try {
                        const form = new FormData();
            form.append('text', text);
                        const res = await fetch(API_BASE + '/tts', { method: 'POST', body: form });
                        const data = await res.json();
                        if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
                        if (data.audio_b64_wav) {
                            const wavBlob = b64ToBlob(data.audio_b64_wav, 'audio/wav');
                            const url = URL.createObjectURL(wavBlob);
                            const a = document.createElement('a');
                            a.href = url;
                a.download = 'summary.wav';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                        }
                    } catch (e) { }
                });

    // Summary: Download TXT
    summaryDownloadTxtBtn && summaryDownloadTxtBtn.addEventListener('click', function(){
        const text = (summaryBox && summaryBox.value || '').trim();
        if (!text) return;
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
        a.download = 'summary.txt';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
    });

    // Chat UI removed

    // Stop functionality removed per user request
    let currentAbort = null;

    // Multi-question ask
    multiAskBtn && multiAskBtn.addEventListener('click', async function () {
        const prompt = (multiPrompt && multiPrompt.value || '').trim();
        if (!prompt) return;
        // Require a transcript selection if backend expects it
        const tidCheck = (fileSelectQa && fileSelectQa.value) || (fileSelect && fileSelect.value) || '';
        if (!tidCheck) {
            if (qaMessages) {
                const warn = document.createElement('div');
                warn.className = 'msg msg--assistant';
                warn.innerHTML = '<div class="msg__bubble"></div>';
                warn.querySelector('.msg__bubble').textContent = 'Please select an audio file first.';
                qaMessages.appendChild(warn);
                qaMessages.scrollTop = qaMessages.scrollHeight;
            }
            return;
        }
        multiAskBtn.disabled = true;
        
        console.log('[multiqa] sending prompt', { prompt, transcript_id: tidCheck });
        showToast && showToast('Sending question…', 'ok');
        if (multiResults) {
            multiResults.innerHTML = '';
            try { multiResults.style.display = 'none'; } catch(_) {}
        }
        // Append user bubble
        if (qaMessages) {
            const userMsg = document.createElement('div');
            userMsg.className = 'msg msg--user';
            userMsg.innerHTML = '<div class="msg__bubble"></div>';
            userMsg.querySelector('.msg__bubble').textContent = prompt;
            qaMessages.appendChild(userMsg);
            qaMessages.scrollTop = qaMessages.scrollHeight;
            if (multiPrompt) { try { multiPrompt.value = ''; } catch(_) {} }
        }
        // Typing placeholder for assistant
        let typingEl = null;
        if (qaMessages) {
            typingEl = document.createElement('div');
            typingEl.className = 'msg msg--assistant';
            typingEl.innerHTML = '<div class="msg__bubble">...</div>';
            qaMessages.appendChild(typingEl);
            qaMessages.scrollTop = qaMessages.scrollHeight;
        }
        let stopIndicator = null;
        try {
            // Prepare abort controller to allow stopping generation
            let controller = undefined;
            currentAbort = null;
            const form = new FormData();
            form.append('prompt', prompt);
            const tid = tidCheck;
            if (tid) form.append('transcript_id', tid);
            const url = API_BASE + '/multiqa';
            console.log('[multiqa] POST', url);
            const res = await fetch(url, { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
            const items = data.items || [];
            lastMultiItems = items;
            // Build combined paragraph only
            try {
                const combined = items
                    .map(it => (it && it.a ? String(it.a).trim() : ''))
                    .filter(Boolean)
                    .join('\n\n');
                if (multiCombinedAnswer) multiCombinedAnswer.value = combined || '';
                if (stopIndicator) stopIndicator();
                // Replace typing bubble with assistant bubble + actions
                if (qaMessages && typingEl) {
                    typingEl.innerHTML = '<div class="msg__bubble"></div>';
                    const bubble = typingEl.querySelector('.msg__bubble');
                    await typeOutText(bubble, combined || 'No answer found.');
                    const actions = document.createElement('div');
                    actions.className = 'msg__actions';
                    const safeText = (combined || '').replace(/"/g, '&quot;');
                    actions.innerHTML = '<button data-action="mini-play" data-text="'+safeText+'" title="Play">🎵</button>'+
                                         '<button data-action="copy-pair" data-text="'+safeText+'" title="Copy Q&A">📋</button>';
                    typingEl.appendChild(actions);
                    qaMessages.scrollTop = qaMessages.scrollHeight;
                } else if (multiChatAnswer) {
                    await typeOutText(multiChatAnswer, combined || 'No answer found.');
                }
            } catch (_) {}
        } catch (e) {
            console.error('[multiqa] error', e);
            // Show error in chat window
            if (qaMessages) {
                if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
                const errMsg = document.createElement('div');
                errMsg.className = 'msg msg--assistant';
                errMsg.innerHTML = '<div class="msg__bubble"></div>';
                errMsg.querySelector('.msg__bubble').textContent = 'Error: ' + e.message;
                qaMessages.appendChild(errMsg);
                qaMessages.scrollTop = qaMessages.scrollHeight;
            } else {
            const block = document.createElement('div');
            block.textContent = 'Error: ' + e.message;
                try { multiResults.appendChild(block); multiResults.style.display = ''; } catch(_) {}
            }
        } finally {
            currentAbort = null;
            if (stopIndicator) stopIndicator();
            multiAskBtn.disabled = false;
            
            try { showToast && showToast('Done', 'ok'); } catch(_){}
        }
    });


    // Multi-question from recorded audio
    multiAskFromAudioBtn && multiAskFromAudioBtn.addEventListener('click', async function () {
        if (!recordedBlob) return;
        multiResults && (multiResults.innerHTML = '');
        try { if (multiResults) multiResults.style.display = 'none'; } catch(_) {}
        multiAskFromAudioBtn.disabled = true;
        console.log('[multiqa] sending recorded audio');
        showToast && showToast('Sending recorded audio…', 'ok');
        // Typing placeholder for assistant
        let typingEl = null;
        if (qaMessages) {
            typingEl = document.createElement('div');
            typingEl.className = 'msg msg--assistant';
            typingEl.innerHTML = '<div class="msg__bubble">...</div>';
            qaMessages.appendChild(typingEl);
            qaMessages.scrollTop = qaMessages.scrollHeight;
        }
        let stopIndicator = null;
        try {
            const form = new FormData();
            form.append('audio', recordedBlob, 'multiqa.webm');
            const tid = fileSelect && fileSelect.value || '';
            if (tid) form.append('transcript_id', tid);
            const url = API_BASE + '/multiqa';
            console.log('[multiqa] POST', url);
            const res = await fetch(url, { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
            const items = data.items || [];
            lastMultiItems = items;
            // Build combined paragraph only
            try {
                const combined = items
                    .map(it => (it && it.a ? String(it.a).trim() : ''))
                    .filter(Boolean)
                    .join('\n\n');
                if (multiCombinedAnswer) multiCombinedAnswer.value = combined || '';
                if (stopIndicator) stopIndicator();
                if (qaMessages && typingEl) {
                    typingEl.innerHTML = '<div class="msg__bubble"></div>';
                    const bubble = typingEl.querySelector('.msg__bubble');
                    await typeOutText(bubble, combined || 'No answer found.');
                    const actions = document.createElement('div');
                    actions.className = 'msg__actions';
                    actions.innerHTML = '<button aria-label="thumb up">👍</button><button aria-label="thumb down">👎</button><button aria-label="regenerate">↻</button>';
                    typingEl.appendChild(actions);
                    qaMessages.scrollTop = qaMessages.scrollHeight;
                } else if (multiChatAnswer) {
                    await typeOutText(multiChatAnswer, combined || '');
                }
            } catch (_) {}
        } catch (e) {
            console.error('[multiqa] error', e);
            if (qaMessages) {
                if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
                const errMsg = document.createElement('div');
                errMsg.className = 'msg msg--assistant';
                errMsg.innerHTML = '<div class="msg__bubble"></div>';
                errMsg.querySelector('.msg__bubble').textContent = 'Error: ' + e.message;
                qaMessages.appendChild(errMsg);
                qaMessages.scrollTop = qaMessages.scrollHeight;
            } else {
            const block = document.createElement('div');
            block.textContent = 'Error: ' + e.message;
                try { multiResults.appendChild(block); multiResults.style.display = ''; } catch(_) {}
            }
        } finally {
            if (stopIndicator) stopIndicator();
            multiAskFromAudioBtn.disabled = false;
            try { showToast && showToast('Done', 'ok'); } catch(_){}
        }
    });

    // Typewriter animation helper
    async function typeOutText(targetEl, fullText) {
        try { targetEl.textContent = ''; } catch(_) {}
        const delay = 14;
        for (let i = 0; i < fullText.length; i++) {
            targetEl.textContent = fullText.slice(0, i + 1);
            await new Promise(r => setTimeout(r, delay));
        }
    }

    // Typing indicator helper (animated dots ...)
    function startTypingIndicator(targetEl) {
        try { targetEl.textContent = '...'; } catch(_) {}
        let i = 0;
        const frames = ['.', '..', '...'];
        const id = setInterval(() => {
            try { targetEl.textContent = frames[i % frames.length]; } catch(_) {}
            i++;
        }, 350);
        return function stop() { try { clearInterval(id); } catch(_) {} };
    }

    function b64ToBlob(b64, contentType) {
        const byteChars = atob(b64);
        const byteNumbers = new Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) byteNumbers[i] = byteChars.charCodeAt(i);
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: contentType });
    }

    // Training logic
    async function startTraining() {
        trainStatus.textContent = 'Starting AI training...';
        try {
            const res = await fetch(API_BASE + '/train-real', { method: 'POST' });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();
            trainStatus.textContent = data.status || 'AI training started';
            showToast('Training started', 'ok');
            pollTraining();
        } catch (e) {
            trainStatus.textContent = 'Failed: ' + e.message;
            showToast('Training failed to start: ' + e.message, 'err');
        }
    }

    async function pollTraining() {
        if (trainPollTimer) clearInterval(trainPollTimer);
        trainPollTimer = setInterval(async function () {
            try {
                const res = await fetch(API_BASE + '/progress');
                if (!res.ok) return;
                const data = await res.json();
                const p = Math.max(0, Math.min(100, Number(data.progress || 0)));
                trainProgressBar.style.width = p + '%';
                trainProgressText.textContent = p + '%';
                
                // Show more detailed status
                if (data.is_training) {
                    trainStatus.textContent = `Training... ${data.current_stage || ''}`;
                } else if (data.trained) {
                    trainStatus.textContent = 'Training Complete!';
                } else {
                    trainStatus.textContent = 'Ready to train';
                }
                
                trainArtifact.textContent = data.artifact || '—';
                // Update stages stepper
                if (trainStages) {
                    const order = ['prepare','transcribe','chunk','embed','qa_setup','summary','finalize'];
                    const current = String(data.current_stage || '').toLowerCase();
                    const items = Array.from(trainStages.querySelectorAll('.stage'));
                    items.forEach((el) => {
                        el.classList.remove('stage--active','stage--done');
                        const stage = el.getAttribute('data-stage');
                        const idx = order.indexOf(stage);
                        const curIdx = order.indexOf(current);
                        if (curIdx === -1) return;
                        if (idx < curIdx) el.classList.add('stage--done');
                        if (idx === curIdx) el.classList.add('stage--active');
                    });
                    if (!data.is_training && data.trained) {
                        items.forEach((el) => el.classList.add('stage--done'));
                    }
                    if (!data.is_training && !data.trained) {
                        items.forEach((el) => el.classList.remove('stage--active','stage--done'));
                    }
                }
                if (!data.is_training) clearInterval(trainPollTimer);
            } catch (e) {
                // ignore intermittent errors
            }
        }, 300); // Reduced polling interval for more responsive updates
    }

    startRealTrainBtn && startRealTrainBtn.addEventListener('click', startTraining);

    // Load audio file list for context selection
    async function loadAudioFiles(){
        const selects = [fileSelect, fileSelectQa].filter(Boolean);
        if (!selects.length) return;
        // Show loading placeholder
        selects.forEach(sel => {
            sel.innerHTML = '';
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Loading…';
            sel.appendChild(opt);
            sel.disabled = true;
        });
        try {
            const res = await fetch(API_BASE + '/audio-files');
            const data = await res.json();
            const rawItems = (data && data.items) || [];
            // Normalize items to { id, label }
            const items = rawItems.map((it) => {
                try {
                    const id = it.id || it.transcript_id || it.uuid || '';
                    const label = it.filename || it.name || it.title || id || '';
                    return { id, label };
                } catch(_) { return { id: '', label: '' }; }
            }).filter(it => it.id || it.label);
            const populate = (sel, statusEl) => {
                sel.innerHTML = '';
            if (!items.length) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'No audio files';
                    sel.appendChild(opt);
                    sel.disabled = true;
                    if (statusEl) statusEl.textContent = '—';
                return;
            }
            const anyOpt = document.createElement('option');
            anyOpt.value = '';
            anyOpt.textContent = 'All files';
                sel.appendChild(anyOpt);
            items.forEach(it => {
                const opt = document.createElement('option');
                    opt.value = it.id || '';
                    opt.textContent = it.label || it.id || '';
                    sel.appendChild(opt);
                });
                sel.disabled = false;
                if (statusEl) statusEl.textContent = '';
            };
            populate(fileSelect, fileSelectStatus);
            populate(fileSelectQa, fileSelectStatusQa);
        } catch (e) {
            [
                [fileSelect, fileSelectStatus],
                [fileSelectQa, fileSelectStatusQa]
            ].forEach(([sel, statusEl]) => {
                if (!sel) return;
                sel.innerHTML = '';
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Error loading files';
                sel.appendChild(opt);
                sel.disabled = true;
                if (statusEl) statusEl.textContent = 'Failed to load list';
            });
        }
    }

    // Initial load
    loadAudioFiles();

    // Re-load when user switches to Summary or Q&A tabs
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            const target = this.getAttribute('data-tab');
            if (target === 'summary' || target === 'qa') {
                loadAudioFiles();
            }
        });
    });
})();



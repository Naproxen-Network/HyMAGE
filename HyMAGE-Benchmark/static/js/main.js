/* ═══════════════════════════════════════════════════════
   Hypergraph Benchmarks — Frontend Logic
   ═══════════════════════════════════════════════════════ */

// ── i18n Translations ──
const T = {
    en: {
        app_title: "HyperBench",
        nav_home: "Home",
        nav_structural: "Structural Metrics",
        home_title: "Hypergraph Benchmark Platform",
        home_subtitle: "Structural analysis and annotation tools for HyMAGE-generated hypergraphs",
        global_dataset: "Global Dataset",
        dataset_name: "Dataset Name",
        drop_hint: "Drag & drop a folder (e.g. CS) or files (.txt / .json) here",
        btn_upload: "Upload",
        structural_desc: "Evaluate 8 structural patterns using JS divergence and Pearson correlation",
        configuration: "Configuration",
        select_dataset: "Select Dataset",
        select_files: "Select Files",
        or_upload: "Or Upload New Files",
        btn_run: "Run Analysis",
        btn_train: "Train",
        comparison: "Comparison Results",
        visualizations: "Visualizations",
        loading: "Processing...",
        no_dataset: "No dataset loaded",
        drop_txt: "Drop .txt files",
        struct_placeholder: "Select files and run analysis",
        structural_page_desc: "Compare structural properties of hypergraphs across 8 patterns",
        nav_annotate: "Annotation",
        annotate_desc: "Manually annotate hypergraphs with rich node profiles and labeled hyperedges",
    },
    zh: {
        app_title: "超图基准",
        nav_home: "首页",
        nav_structural: "结构指标",
        home_title: "超图基准测试平台",
        home_subtitle: "面向 HyMAGE 生成超图的结构分析与标注工具",
        global_dataset: "全局数据集",
        dataset_name: "数据集名称",
        drop_hint: "拖拽文件夹（如 CS）或文件（.txt / .json）到此处",
        btn_upload: "上传",
        structural_desc: "使用JS散度和Pearson相关系数评估8种结构模式",
        configuration: "配置",
        select_dataset: "选择数据集",
        select_files: "选择文件",
        or_upload: "或上传新文件",
        btn_run: "运行分析",
        btn_train: "训练",
        comparison: "对比结果",
        visualizations: "可视化",
        loading: "处理中...",
        no_dataset: "未加载数据集",
        drop_txt: "拖拽 .txt 文件",
        struct_placeholder: "选择文件并运行分析",
        structural_page_desc: "对比超图的8种结构特性",
        nav_annotate: "标注",
        annotate_desc: "手动标注超图，为每个节点添加丰富的属性信息和标签",
    },
    ko: {
        app_title: "하이퍼벤치",
        nav_home: "홈",
        nav_structural: "구조 지표",
        home_title: "하이퍼그래프 벤치마크 플랫폼",
        home_subtitle: "HyMAGE 생성 하이퍼그래프를 위한 구조 분석 및 주석 도구",
        global_dataset: "글로벌 데이터셋",
        dataset_name: "데이터셋 이름",
        drop_hint: "폴더(예: CS) 또는 파일(.txt / .json)을 여기에 드래그하세요",
        btn_upload: "업로드",
        structural_desc: "JS 발산과 Pearson 상관계수를 사용하여 8가지 구조 패턴 평가",
        configuration: "설정",
        select_dataset: "데이터셋 선택",
        select_files: "파일 선택",
        or_upload: "또는 새 파일 업로드",
        btn_run: "분석 실행",
        btn_train: "훈련",
        comparison: "비교 결과",
        visualizations: "시각화",
        loading: "처리 중...",
        no_dataset: "데이터셋 없음",
        drop_txt: ".txt 파일 드래그",
        struct_placeholder: "파일을 선택하고 분석을 실행하세요",
        structural_page_desc: "8가지 패턴에 걸쳐 하이퍼그래프의 구조적 특성 비교",
        nav_annotate: "주석",
        annotate_desc: "풍부한 노드 프로필과 레이블이 있는 하이퍼엣지를 수동으로 주석 달기",
    }
};

let currentLang = localStorage.getItem('lang') || 'en';

function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('lang', lang);
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (T[lang] && T[lang][key]) {
            el.textContent = T[lang][key];
        }
    });
    document.querySelectorAll('.btn-lang').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });
}

// ── Sidebar Toggle (Mobile) ──
function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('show');
}

// ── Folder / File reading utilities ──

/**
 * Recursively read all files from a dropped directory entry.
 * Returns a Promise that resolves to an array of File objects.
 */
function readDirectoryEntryRecursive(directoryEntry) {
    return new Promise((resolve) => {
        const reader = directoryEntry.createReader();
        const allFiles = [];

        function readBatch() {
            reader.readEntries(entries => {
                if (entries.length === 0) {
                    resolve(allFiles);
                    return;
                }
                const promises = [];
                for (const entry of entries) {
                    if (entry.isFile) {
                        promises.push(new Promise(res => {
                            entry.file(f => { allFiles.push(f); res(); });
                        }));
                    } else if (entry.isDirectory) {
                        promises.push(
                            readDirectoryEntryRecursive(entry).then(files => {
                                allFiles.push(...files);
                            })
                        );
                    }
                }
                Promise.all(promises).then(() => readBatch());
            });
        }
        readBatch();
    });
}

/**
 * Process a drop event. If a folder is dropped, recursively read all
 * .txt / .json files from it and auto-fill the dataset name.
 * Returns {folderName: string|null, files: File[]}.
 */
async function processDropItems(dataTransfer) {
    const items = dataTransfer.items;
    let folderName = null;
    let collectedFiles = [];

    if (items && items.length > 0) {
        const entries = [];
        for (let i = 0; i < items.length; i++) {
            const entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
            if (entry) entries.push(entry);
        }

        for (const entry of entries) {
            if (entry.isDirectory) {
                // Use folder name as dataset name
                folderName = entry.name;
                const files = await readDirectoryEntryRecursive(entry);
                // Only keep .txt and .json files
                const filtered = files.filter(f =>
                    f.name.endsWith('.txt') || f.name.endsWith('.json')
                );
                collectedFiles.push(...filtered);
            } else if (entry.isFile) {
                const file = await new Promise(res => entry.file(f => res(f)));
                collectedFiles.push(file);
            }
        }
    }

    // Fallback: if no webkitGetAsEntry support, use plain files
    if (collectedFiles.length === 0 && dataTransfer.files.length > 0) {
        collectedFiles = Array.from(dataTransfer.files);
    }

    return { folderName, files: collectedFiles };
}

// ── Pending files store (per drop zone) ──
const pendingFiles = {};

// ── Drop Zone Setup ──
function setupDropZone(zoneId, inputId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', async e => {
        e.preventDefault();
        zone.classList.remove('drag-over');

        const { folderName, files } = await processDropItems(e.dataTransfer);

        if (files.length === 0) {
            showToast('No .txt or .json files found', 'warning');
            return;
        }

        // Store files for later upload
        pendingFiles[zoneId] = files;

        // If this is the global drop zone, auto-fill dataset name from folder
        if (folderName) {
            const nameInput = document.getElementById('datasetName');
            if (nameInput && (zoneId === 'globalDropZone')) {
                nameInput.value = folderName;
            }
        }

        // Show count
        const names = files.map(f => f.name);
        const txtCount = names.filter(n => n.endsWith('.txt')).length;
        const jsonCount = names.filter(n => n.endsWith('.json')).length;
        const label = folderName
            ? `📁 Folder "${folderName}": ${txtCount} txt, ${jsonCount} json`
            : `Files ready: ${names.join(', ')}`;
        showToast(label, 'info');

        // Update zone visual
        zone.innerHTML = `
            <i class="fas fa-folder-open" style="color:var(--accent-blue);font-size:1.5rem"></i>
            <p style="margin:6px 0 0;font-size:0.85rem"><strong>${folderName || 'Files'}</strong></p>
            <small class="text-muted">${txtCount} txt, ${jsonCount} json files ready</small>
        `;

        // Dispatch custom event so page scripts can auto-upload on folder drop
        if (folderName) {
            zone.dispatchEvent(new CustomEvent('folderDropped', {
                detail: { folderName, files }
            }));
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            // Store files from input as well
            pendingFiles[zoneId] = Array.from(input.files);
            const names = Array.from(input.files).map(f => f.name).join(', ');
            showToast(`Files selected: ${names}`, 'info');
        }
    });
}

// ── File Upload ──
function uploadGlobalFiles() {
    const dataset = document.getElementById('datasetName').value.trim() || 'default';
    const input = document.getElementById('globalFileInput');

    // Gather files: prefer pendingFiles (from folder drop), then input.files
    let files = pendingFiles['globalDropZone'] || [];
    if (files.length === 0 && input.files && input.files.length > 0) {
        files = Array.from(input.files);
    }

    if (files.length === 0) {
        showToast('No files selected — drop a folder or select files', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('dataset', dataset);
    for (const f of files) {
        formData.append('files', f);
    }

    showLoading(`Uploading ${files.length} files...`);
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { showToast(data.error, 'danger'); return; }
        showToast(`Uploaded ${data.files.length} files to "${data.dataset}"`, 'success');
        // Reset
        input.value = '';
        delete pendingFiles['globalDropZone'];
        // Restore drop zone visual
        const zone = document.getElementById('globalDropZone');
        if (zone) {
            zone.innerHTML = `
                <i class="fas fa-cloud-upload-alt fa-2x mb-2"></i>
                <p data-i18n="drop_hint">Drag & drop a folder or files (.txt / .json) here</p>
                <input type="file" id="globalFileInput" multiple accept=".txt,.json" style="display:none">
            `;
            setupDropZone('globalDropZone', 'globalFileInput');
        }
        loadDatasets();
        // Refresh dataset selectors on other pages if present
        ['structDataset'].forEach(id => {
            if (document.getElementById(id)) loadDatasetOptions(id);
        });
    })
    .catch(err => { hideLoading(); showToast('Upload error: ' + err, 'danger'); });
}

// ── Load Datasets ──
function loadDatasets() {
    fetch('/api/datasets')
    .then(r => r.json())
    .then(datasets => {
        const container = document.getElementById('datasetList');
        if (!container) return;
        if (datasets.length === 0) {
            container.innerHTML = '<span class="text-muted small">No datasets uploaded yet</span>';
            return;
        }
        container.innerHTML = datasets.map(d => `
            <div class="dataset-badge">
                <i class="fas fa-database"></i>
                <strong>${d.name}</strong>
                <span class="file-count">${d.txt_count} txt, ${d.json_count} json</span>
            </div>
        `).join('');
    });
}

// ── Load Dataset Options for Select ──
function loadDatasetOptions(selectId) {
    fetch('/api/datasets')
    .then(r => r.json())
    .then(datasets => {
        const sel = document.getElementById(selectId);
        if (!sel) return;
        const current = sel.value;
        sel.innerHTML = '<option value="">-- Select --</option>';
        datasets.forEach(d => {
            sel.innerHTML += `<option value="${d.name}" ${d.name === current ? 'selected' : ''}>${d.name} (${d.txt_count} txt, ${d.json_count} json)</option>`;
        });
    });
}

// ── Load Dataset Files into Checklist ──
function loadDatasetFiles(selectId, containerId, filter) {
    const dataset = document.getElementById(selectId).value;
    const container = document.getElementById(containerId);
    if (!dataset || !container) return;

    fetch(`/api/dataset/${dataset}/files`)
    .then(r => r.json())
    .then(data => {
        container.innerHTML = '';
        let files = data.files;
        if (filter === '.txt') {
            files = files.filter(f => f.type === 'txt');
        }
        files.forEach(f => {
            const id = `ck_${containerId}_${f.name.replace(/[^a-zA-Z0-9]/g, '_')}`;
            container.innerHTML += `
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" value="${f.name}" id="${id}">
                    <label class="form-check-label small" for="${id}">
                        ${f.name}
                        <span class="text-muted">(${(f.size/1024).toFixed(1)}KB)</span>
                    </label>
                </div>`;
        });
    });
}

// ── Auto Upload & Select (for sub-page folder drops) ──
async function autoUploadAndSelect(folderName, files, selectId, checklistId, filter) {
    showLoading(`Uploading folder "${folderName}"...`);
    const formData = new FormData();
    formData.append('dataset', folderName);
    for (const f of files) formData.append('files', f);

    try {
        const r = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await r.json();
        hideLoading();
        if (data.error) { showToast(data.error, 'danger'); return; }
        showToast(`Uploaded ${data.files.length} files to "${folderName}"`, 'success');

        // Refresh dataset dropdown and auto-select the new dataset
        await loadDatasetOptions(selectId);
        setTimeout(() => {
            const sel = document.getElementById(selectId);
            if (sel) {
                sel.value = folderName;
                // Trigger file list load
                loadDatasetFiles(selectId, checklistId, filter);
            }
        }, 200);
    } catch(err) {
        hideLoading();
        showToast('Upload error: ' + err, 'danger');
    }
}

// ── Loading Overlay ──
function showLoading(text) {
    const overlay = document.getElementById('loadingOverlay');
    const textEl = document.getElementById('loadingText');
    if (textEl) textEl.textContent = text || T[currentLang]?.loading || 'Processing...';
    if (overlay) overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.style.display = 'none';
}

// ── Toast Notification ──
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const icons = {
        success: 'fa-check-circle',
        danger: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };

    const id = 'toast_' + Date.now();
    const html = `
        <div id="${id}" class="toast show" role="alert">
            <div class="toast-header">
                <i class="fas ${icons[type] || icons.info} text-${type} me-2"></i>
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close" onclick="document.getElementById('${id}').remove()"></button>
            </div>
            <div class="toast-body">${message}</div>
        </div>`;
    container.insertAdjacentHTML('beforeend', html);
    setTimeout(() => {
        const el = document.getElementById(id);
        if (el) el.remove();
    }, 5000);
}

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    // Language buttons
    document.querySelectorAll('.btn-lang').forEach(btn => {
        btn.addEventListener('click', () => setLanguage(btn.dataset.lang));
    });
    setLanguage(currentLang);
});

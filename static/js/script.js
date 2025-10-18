let pendingFrames = 0;
let frameCount = 0;


const app = {
    trackData: {}, // 全局对象，存储用户输入的 ID 数据

    init: function () {
        console.log("初始化应用...");
        camera.init();
        socket.init();
        ui.init();
        this.fetchTargetClasses();

    },

    // 获取目标类别并创建可视化编辑器
    fetchTargetClasses: function() {
        fetch('/get_target_classes')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById("categoriesContainer");
            container.innerHTML = "";
            data.forEach(category => {
                app.trackData[category] = []; // 初始化字典
                const categoryBox = document.createElement("div");
                categoryBox.className = "category-box";
                categoryBox.innerHTML = `
                    <div class="current-ids-section">
                        <h5>${category}</h5>
                        <div class="track-items" id="current-${category}"></div>
                    </div>
                    <div class="added-ids-section">
                        <div class="track-items" id="items-${category}"></div>
                    </div>
                    <div class="add-section">
                        <div class="bottom-buttons">
                            <input type="number" id="input-${category}" placeholder="ID">
                            <button onclick="app.addTrackId('${category}')">+</button>
                            <button class="clear-all" onclick="app.clearTrackIds('${category}')">清空所有</button>
                        </div>
                    </div>
                `;
                container.appendChild(categoryBox);
                app.updateTrackDisplay(category);
                // 添加回车事件
                const input = document.getElementById(`input-${category}`);
                input.addEventListener("keypress", function(event) {
                    if (event.key === "Enter") {
                        app.addTrackId(category);
                    }
                });
            });
        })
        .catch(error => console.error('获取类别失败:', error));
    },

    // 更新指定类别的Track显示
    updateTrackDisplay: function(category) {
        const itemsDiv = document.getElementById(`items-${category}`);
        itemsDiv.innerHTML = "";
        app.trackData[category].forEach((id, index) => {
            const itemDiv = document.createElement("div");
            itemDiv.className = "item";
            itemDiv.innerHTML = `
                <span>${id}</span>
                <button onclick="app.removeTrackId('${category}', ${index})">×</button>
            `;
            itemsDiv.appendChild(itemDiv);
        });
    },

    // 添加Track ID
    addTrackId: function(category) {
        const input = document.getElementById(`input-${category}`);
        const value = parseInt(input.value.trim());
        if (!isNaN(value) && !app.trackData[category].includes(value)) {
            app.trackData[category].push(value);
            app.updateTrackDisplay(category);
            input.value = "";
        }
    },

    // 删除Track ID
    removeTrackId: function(category, index) {
        app.trackData[category].splice(index, 1);
        app.updateTrackDisplay(category);
    },

    // 清空所有Track ID
    clearTrackIds: function(category) {
        app.trackData[category] = [];
        app.updateTrackDisplay(category);
    }
};

// **初始化**
document.addEventListener("DOMContentLoaded", () => {
    app.init(); // 保留你的 `init()` 方法，确保它仍然运行
    app.fetchTargetClasses(); // 添加表单功能
});


const camera = {
    videoElement: null,
    canvasElement: null,
    ctx: null,
    
    init: function() {
        this.videoElement = document.getElementById("video");
        this.canvasElement = document.getElementById("canvas");
        this.ctx = this.canvasElement.getContext("2d");
        
        this.enumerateCameras();
        this.startFrameCapture();
    },
    
    enumerateCameras: function() {
        navigator.mediaDevices.enumerateDevices().then(devices => {
            let cameraSelect = document.getElementById("cameraSelect");
            
            devices.forEach(device => {
                if (device.kind === "videoinput") {
                    let option = document.createElement("option");
                    option.value = device.deviceId;
                    option.text = device.label || `摄像头 ${cameraSelect.length + 1}`;
                    cameraSelect.appendChild(option);
                }
            });
            
            cameraSelect.onchange = () => {
                let selectedId = cameraSelect.value;
                socket.emit("switch_camera", selectedId);
                this.startCamera(selectedId);
            };
            
            if (cameraSelect.options.length > 0) {
                this.startCamera(cameraSelect.options[0].value);
            }
        });
    },
    
    startCamera: function(deviceId) {
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                deviceId: { exact: deviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        })
        .then(stream => {
            this.videoElement.srcObject = stream;
            this.videoElement.onloadedmetadata = () => {
                this.videoElement.play();
            };
        })
        .catch(err => console.error("无法访问摄像头:", err));
    },
    
    startFrameCapture: function() {
        const MAX_FRAMES = 10;
        setInterval(() => {
            if (pendingFrames >= MAX_FRAMES) return;
            if (this.videoElement.videoWidth > 0 && this.videoElement.videoHeight > 0) {
                this.canvasElement.width = this.videoElement.videoWidth;
                this.canvasElement.height = this.videoElement.videoHeight;
                this.ctx.drawImage(this.videoElement, 0, 0, this.canvasElement.width, this.canvasElement.height);
                var imageData = this.canvasElement.toDataURL("image/jpeg");
                socket.emit("frame", imageData);
                pendingFrames++; // 新增4：计数增加
            }
        }, 40);
    }
};

const socket = {
    init: function () {
        this.io = io.connect("http://" + document.domain + ":5000");
        
        this.io.on("processed_frame", function (data) {

            let canvas = document.getElementById("processed_canvas");
            let ctx = canvas.getContext("2d");

            let img = new Image();

            img.onload = function () {
                // **确保 canvas1 和 canvas2 继承正确的尺寸**
                let canvas1 = document.createElement("canvas");
                let canvas2 = document.createElement("canvas");

                canvas.width = img.width;
                canvas.height = img.height;

                canvas1.width = canvas2.width = canvas.width;
                canvas1.height = canvas2.height = canvas.height;

                let ctx1 = canvas1.getContext("2d");
                let ctx2 = canvas2.getContext("2d");

                ctx.clearRect(0, 0, canvas.width, canvas.height); // 清除旧画面
                ctx.drawImage(img, 0, 0); // 绘制原始图像到 `canvas`

                let idData = {}; // 存储类别对应的 ID 数组
                let visibleBoxes = []; // 记录所有需要保留的框

                // **只有当 ui.draw_flag === 1 和 ui.focus_flag === 1 时，才填充黑色遮罩**
                if (ui.draw_flag === 1 && ui.focus_flag === 1) {
                    ctx2.fillStyle = "rgba(0, 0, 0, 0.7)";
                    ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
                }

                // 遍历所有类别（如 person, vehicle, animal）
                Object.keys(data.draw).forEach(category => {
                    console.log(`正在处理类别: ${category}`);
                    const trackingArray = data.draw[category] || [];

                    trackingArray.forEach(track => {
                        if (Array.isArray(track) && track.length >= 7) {
                            let [id, confidence, x, y, width, height, trajectory] = track;

                            // 存储 ID 数据（无论 tracking_flag 状态）
                            if (!idData[category]) {
                                idData[category] = [];
                            }
                            idData[category].push(id);

                            // 只有在 `tracking_flag === 1` 时，才限制绘制
                            if (ui.tracking_flag === 1 && (!app.trackData[category] || !app.trackData[category].includes(id))) {
                                return; // 过滤掉不在 trackData 中的 ID
                            }

                            // 记录需要保留的区域
                            visibleBoxes.push({ x, y, width, height });

                            // **只有当 draw_flag === 1 时，才绘制框、文字、轨迹**
                            if (ui.draw_flag === 1) {
                                // **在 canvas1 上绘制框、文字、轨迹**
                                ctx1.strokeStyle = category === "person" ? "red" : "green";
                                ctx1.lineWidth = 2;
                                ctx1.strokeRect(x, y, width, height);

                                let text = `${category} ID: ${id}, Score: ${(confidence * 100).toFixed(2)}%`;
                                ctx1.fillStyle = "yellow";
                                ctx1.fillText(text, x, y - 10);

                                if (Array.isArray(trajectory) && trajectory.length > 1) {
                                    ctx1.strokeStyle = "blue"; // 轨迹颜色
                                    ctx1.lineJoin = "round"; // 轨迹平滑
                                    ctx1.beginPath();
                                    trajectory.forEach((point, index) => {
                                        index === 0 ? ctx1.moveTo(point[0], point[1]) : ctx1.lineTo(point[0], point[1]);
                                    });
                                    ctx1.stroke();
                                }
                            }
                        } else {
                            console.error(`数据格式错误:`, track);
                        }
                    });
                });

                // **如果 draw_flag === 1 和 focus_flag === 1，擦除遮罩层中的框区域**
                if (ui.draw_flag === 1 && ui.focus_flag === 1) {
                    visibleBoxes.forEach(({ x, y, width, height }) => {
                        ctx2.clearRect(x, y, width, height); // 清空遮罩层中框的位置
                    });

                    // **合并所有图层**
                    ctx.drawImage(canvas2, 0, 0); // 先绘制遮罩层
                }

                ctx.drawImage(canvas1, 0, 0); // 最后绘制框、文字、轨迹

                // 获取所有类别
                const allCategories = new Set([...Object.keys(idData), ...Object.keys(app.trackData)]);
                const categories = Array.from(allCategories).sort();

                // **更新 ID 列表（始终显示所有 ID）**
                categories.forEach(category => {
                    const ids = idData[category] || [];
                    const itemsDiv = document.getElementById(`current-${category}`);
                    if (itemsDiv) {
                        itemsDiv.innerHTML = "";
                        ids.forEach(id => {
                            const itemDiv = document.createElement("div");
                            itemDiv.className = "item";
                            itemDiv.innerHTML = `<span>${id}</span>`;
                            itemsDiv.appendChild(itemDiv);
                        });
                    }
                });
            };

            img.src = data.image; // 加载图像

            frameCount++;
            pendingFrames = Math.max(0, pendingFrames - 1); // 新增5：确保不减到负数
        });

        this.io.on("stats", function (data) {
            console.log("Received stats:", data);
            document.getElementById("fps").innerText = `FPS: ${data.fps.toFixed(2)}`;
            
            // CPU usage with progress bars
            const cpuDiv = document.getElementById('cpuUsage');
            cpuDiv.innerHTML = '<h4>CPU</h4>';
            data.cpu.forEach((percent, index) => {
                const coreNum = 4 + index;
                const barDiv = document.createElement('div');
                barDiv.className = 'usage-bar';
                barDiv.innerHTML = `
                    <span>Core ${coreNum}: ${percent.toFixed(2)}%</span>
                    <progress value="${percent}" max="100"></progress>
                `;
                cpuDiv.appendChild(barDiv);
            });
            
            // NPU usage with progress bars
            const npuDiv = document.getElementById('npuUsage');
            npuDiv.innerHTML = '<h4>NPU</h4>';
            const npuStr = data.npu.replace('NPU load: ', '');
            const cores = npuStr.split(', ');
            cores.forEach(core => {
                const [label, percentStr] = core.split(': ');
                const percent = parseFloat(percentStr);
                const barDiv = document.createElement('div');
                barDiv.className = 'usage-bar';
                barDiv.innerHTML = `
                    <span>${label}: ${percent.toFixed(2)}%</span>
                    <progress value="${percent}" max="100"></progress>
                `;
                npuDiv.appendChild(barDiv);
            });
        });

    },

    emit: function (event, data) {
        this.io.emit(event, data);
    }
};




const ui = {
    draw_flag: 1,
    tracking_flag: 0,
    focus_flag: 0,
    
    init: function() {
        this.drawButton = document.getElementById("drawToggle");
        this.trackingButton = document.getElementById("trackingToggle");
        this.focusButton = document.getElementById("focusToggle");
        
        this.updateButtonStates();
        
        this.drawButton.addEventListener("click", () => this.toggleDraw());
        this.trackingButton.addEventListener("click", () => this.toggleTracking());
        this.focusButton.addEventListener("click", () => this.toggleFocus());
    },
    
    toggleDraw: function() {
        this.draw_flag = this.draw_flag === 0 ? 1 : 0;
        this.updateButtonStates();
    },
    
    toggleTracking: function() {
        this.tracking_flag = this.tracking_flag === 0 ? 1 : 0;
        this.updateButtonStates();
    },
    
    toggleFocus: function() {
        this.focus_flag = this.focus_flag === 0 ? 1 : 0;
        this.updateButtonStates();
    },
    
    updateButtonStates: function() {
        // 更新绘制按钮
        this.drawButton.textContent = this.draw_flag === 1 ? "Draw On" : "Draw Off";
        if (this.draw_flag === 1) {
            this.drawButton.classList.remove("draw-disabled");
            this.drawButton.classList.add("draw-enabled");
        } else {
            this.drawButton.classList.remove("draw-enabled");
            this.drawButton.classList.add("draw-disabled");
        }
        
        // 更新跟踪按钮
        this.trackingButton.textContent = this.tracking_flag === 1 ? "Tracking On" : "Tracking Off";
        if (this.tracking_flag === 1) {
            this.trackingButton.classList.remove("tracking-disabled");
            this.trackingButton.classList.add("tracking-enabled");
        } else {
            this.trackingButton.classList.remove("tracking-enabled");
            this.trackingButton.classList.add("tracking-disabled");
        }
        
        // 更新聚焦按钮
        this.focusButton.textContent = this.focus_flag === 1 ? "Focus On" : "Focus Off";
        if (this.focus_flag === 1) {
            this.focusButton.classList.remove("focus-disabled");
            this.focusButton.classList.add("focus-enabled");
        } else {
            this.focusButton.classList.remove("focus-enabled");
            this.focusButton.classList.add("focus-disabled");
        }
    }
};
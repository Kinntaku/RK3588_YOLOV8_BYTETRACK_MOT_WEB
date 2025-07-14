let pendingFrames = 0;
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;


const app = {
    trackData: {}, // 全局对象，存储用户输入的 ID 数据

    init: function () {
        console.log("初始化应用...");
        camera.init();
        socket.init();
        ui.init();
        this.fetchTargetClasses();

    },

    // 获取目标类别并创建表单
    fetchTargetClasses: function() {
        fetch('/get_target_classes')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById("forms-container");
            container.innerHTML = "";

            data.forEach(category => {
                app.trackData[category] = []; // 初始化字典

                const form = document.createElement("form");

                const input = document.createElement("input");
                input.type = "text";
                input.name = category;
                input.placeholder = `输入 ${category} ID`;

                const button = document.createElement("button");
                button.type = "submit";
                button.textContent = `提交 ${category}`;

                form.appendChild(input);
                form.appendChild(button);

                form.addEventListener("submit", function (event) {
                    event.preventDefault();
                    app.updateTrackIds(category, input.value);
                    input.value = ""; // 清空输入框
                });

                container.appendChild(form);
            });

            app.updateTrackDisplay(); // 初始更新显示
        })
        .catch(error => console.error('获取类别失败:', error));
    },

    // 更新 ID 数据
updateTrackIds: function(category, value) {
    const ids = value.split(" ")
        .map(id => id.trim())
        .filter(id => id) // 去掉空值
        .map(id => parseInt(id, 10)); // 转换为数字

    // **覆盖更新，而不是追加**
    app.trackData[category] = ids;

    app.updateTrackDisplay(); // 更新实时显示
},


    // 实时更新页面显示
    updateTrackDisplay: function() {
        const trackContainer = document.getElementById("trackData");
        trackContainer.innerHTML = "<h3>用户 Track 数据</h3>";

        const list = document.createElement("ul");

        Object.entries(app.trackData).forEach(([category, ids]) => {
            const listItem = document.createElement("li");
            listItem.textContent = `${category}: [${ids.join(", ")}]`;
            list.appendChild(listItem);
        });

        trackContainer.appendChild(list);
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

                // **只有当 ui.focus_flag === 1 时，填充黑色遮罩**
                if (ui.focus_flag === 1) {
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

                            // **在 canvas1 上绘制框、文字、轨迹**
                            ctx1.strokeStyle = category === "person" ? "red" : "green";
                            ctx1.lineWidth = 2;
                            ctx1.strokeRect(x, y, width, height);

                            let text = `${category} ID: ${id}, Score: ${(confidence * 100).toFixed(1)}%`;
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
                        } else {
                            console.error(`数据格式错误:`, track);
                        }
                    });
                });

                // **如果 focus_flag === 1，擦除遮罩层中的框区域**
                if (ui.focus_flag === 1) {
                    visibleBoxes.forEach(({ x, y, width, height }) => {
                        ctx2.clearRect(x, y, width, height); // 清空遮罩层中框的位置
                    });

                    // **合并所有图层**
                    ctx.drawImage(canvas2, 0, 0); // 先绘制遮罩层
                }

                ctx.drawImage(canvas1, 0, 0); // 最后绘制框、文字、轨迹

                // **更新 ID 列表（始终显示所有 ID）**
                let idList = document.getElementById("idList");
                idList.innerHTML = "";
                Object.entries(idData).forEach(([category, ids]) => {
                    let listItem = document.createElement("li");
                    listItem.textContent = `${category}: [${ids.join(", ")}]`;
                    idList.appendChild(listItem);
                });
            };

            img.src = data.image; // 加载图像

            frameCount++;
            if (frameCount % 40 === 0) {
                const now = performance.now();
                currentFps = Math.round(40000 / (now - lastFpsTime)); // 40帧/(毫秒差/1000)
                lastFpsTime = now;
                document.getElementById("fps").innerText = `FPS: ${currentFps}`;
            }
            pendingFrames = Math.max(0, pendingFrames - 1); // 新增5：确保不减到负数
        });

    },

    emit: function (event, data) {
        this.io.emit(event, data);
    }
};




const ui = {
    tracking_flag: 0,
    focus_flag: 0,
    
    init: function() {
        this.trackingButton = document.getElementById("trackingToggle");
        this.focusButton = document.getElementById("focusToggle");
        
        this.updateButtonStates();
        
        this.trackingButton.addEventListener("click", () => this.toggleTracking());
        this.focusButton.addEventListener("click", () => this.toggleFocus());
    },
    
    toggleTracking: function() {
        this.tracking_flag = this.tracking_flag === 0 ? 1 : 0;
        this.updateButtonStates();
        socket.emit("update_flags", { 
            tracking_flag: this.tracking_flag,
            focus_flag: this.focus_flag
        });
    },
    
    toggleFocus: function() {
        this.focus_flag = this.focus_flag === 0 ? 1 : 0;
        this.updateButtonStates();
        socket.emit("update_flags", { 
            tracking_flag: this.tracking_flag,
            focus_flag: this.focus_flag
        });
    },
    
    updateButtonStates: function() {
        // 更新跟踪按钮
        this.trackingButton.textContent = this.tracking_flag === 1 ? 
            "Tracking Enabled" : "Tracking Disabled";
        this.trackingButton.className = this.tracking_flag === 1 ? 
            "toggle-button tracking-enabled" : "toggle-button tracking-disabled";
        
        // 更新聚焦按钮
        this.focusButton.textContent = this.focus_flag === 1 ? 
            "Focus Enabled" : "Focus Disabled";
        this.focusButton.className = this.focus_flag === 1 ? 
            "toggle-button focus-enabled" : "toggle-button focus-disabled";
    }
};
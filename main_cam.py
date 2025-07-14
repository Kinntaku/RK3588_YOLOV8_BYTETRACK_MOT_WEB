import cv2
import numpy as np
from loguru import logger
from collections import defaultdict
import base64
import time
import traceback
from itertools import chain
from flask import Flask, jsonify, render_template,request,render_template_string,redirect
from flask_socketio import SocketIO,disconnect
from byte_tracker.tracker.byte_tracker import BYTETracker
from rknnpool import rknnPoolExecutor
from config import *

last_frame_time = 0 #用于计算帧率 


#启动web
app = Flask(__name__)
socketio = SocketIO(app)

#轨迹历史保存，每个类别独立字典
track_history = {
    cls_name: defaultdict(list) for cls_name in TARGET_CLASSES  
}

selected_device_id = None  # 记录当前选择的摄像头

frame_count = 0


def prepare_data_and_update_trace(
    classes_tlwhs,  # 按类别分组的 tlwhs 字典
    classes_ids,    # 按类别分组的 ids 字典
    classes_scores, # 按类别分组的 scores 字典
    frame_id=0, 
    max_history=30
    
):
    tracking_data = {}  # 存储最终的字典数据结构

    for class_name in TARGET_CLASSES:
        if not classes_tlwhs[class_name]:  # 跳过空类别
            continue

        cls_track_history = track_history[class_name]
        tracking_data[class_name] = []  # 为该类别创建列表

        for obj_idx in range(len(classes_tlwhs[class_name])):
            track_id = classes_ids[class_name][obj_idx]
            
            tlwh = classes_tlwhs[class_name][obj_idx]
            score = classes_scores[class_name][obj_idx]
            x1, y1, w, h = map(int, tlwh)

            # 获取该轨迹的历史点
            trajectory = cls_track_history.get(track_id, [])

            # 组合数据并添加到字典中
            tracking_data[class_name].append([
                track_id, score, x1, y1, w, h, trajectory
            ])

            # 更新轨迹
            center = (x1 + w // 2, y1 + h // 2)
            if track_id not in cls_track_history:
                cls_track_history[track_id] = [] 

            cls_track_history[track_id].append(center)
            if len(cls_track_history[track_id]) > max_history:
                cls_track_history[track_id].pop(0)

    # 定期清理轨迹
    if frame_id % 50 == 0:
        active_ids = set(chain(*classes_ids.values()))
        for cls_name in TARGET_CLASSES:
            track_history[cls_name] = {tid: track_history[cls_name][tid] 
                                       for tid in track_history[cls_name] if tid in active_ids}

    return tracking_data



def Process_Yolo(frame, class_dets, original_height, original_width, trackers, frame_id):
    tracking_data = {}

    try:
        
        # 初始化存储结构（字典）
        all_tlwhs = {cls_name: [] for cls_name in TARGET_CLASSES}
        all_ids = {cls_name: [] for cls_name in TARGET_CLASSES}
        all_scores = {cls_name: [] for cls_name in TARGET_CLASSES}
        
        if trackers is not None:
            for class_name, dets in class_dets.items():
                if not dets:
                    continue

                dets_array = np.array(dets)
                tracker = trackers[class_name]

                # 更新追踪器
                online_targets = tracker.update(
                    dets_array,
                    [original_height, original_width],
                    [original_height, original_width]
                )

                # 存储当前类别的追踪结果
                for online_target in online_targets:
                    tlwh = online_target.tlwh
                    track_id = online_target.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        all_tlwhs[class_name].append(tlwh)
                        all_ids[class_name].append(track_id)
                        all_scores[class_name].append(online_target.score)
        
        # 调用 `prepare_data_and_update_trace`
        tracking_data = prepare_data_and_update_trace(
            all_tlwhs,
            all_ids,
            all_scores,
            frame_id, 
        ) 
        
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Processing error: {e}\n{error_msg}")

    return frame, tracking_data


pool = rknnPoolExecutor(rknnModel=model, TPEs=3, func=Process_Yolo)



# **处理 WebRTC 传输的视频帧**
@socketio.on('frame')
def receive_frame(data):
    global frame_count
    try:

        # 解码 Base64 视频数据
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 目标检测与处理
        pool.put(frame)
        
        processed_frame, tracking_data,success, = pool.get()
        
        
        if not success:
            logger.debug(f"{processed_frame}{tracking_data}{success}")
            socketio.emit('processed_frame', {'image': '', 'draw': {}})
            return

        frame_count += 1
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = base64.b64encode(buffer).decode()

        socketio.emit('processed_frame', {'image': f'data:image/jpeg;base64,{frame_data}','draw':tracking_data})

    except Exception as e:
        logger.error( e)
        socketio.emit('processed_frame', {'image': '', 'draw': {}})

# **摄像头切换**
@socketio.on('switch_camera')
def switch_camera(device_id):
    global selected_device_id
    selected_device_id = device_id
    logger.info(selected_device_id)
    socketio.emit('camera_changed', {'deviceId': selected_device_id})



#获取筛选器分类
@app.route('/get_target_classes')
def get_target_classes():
    return jsonify(list(TARGET_CLASSES))

connected_client = None  # 只允许一个客户端连接

#根目录路由
@app.route('/')
def index():
    global connected_client
    if connected_client is not None:
        return redirect('/single-client')  # 如果已有连接，则跳转到限制页
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global connected_client
    if connected_client is None:
        connected_client = request.sid
        logger.info(f"Client {request.sid} connect success")
    else:
        logger.warning(f"Refused {request.sid} connect")
        socketio.emit('disconnect_notice', room=request.sid)  # 通知新客户端断开
        disconnect()

@socketio.on('disconnect')
def handle_disconnect():
    global connected_client
    if request.sid == connected_client:
        connected_client = None
        logger.info(f"Client {request.sid} disconnet")

@app.route('/single-client')
def single_client():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <body>
        <h1>一次只允许一个客户端连接</h1>
    </body>
    </html>
    ''')
    
    
def main():
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
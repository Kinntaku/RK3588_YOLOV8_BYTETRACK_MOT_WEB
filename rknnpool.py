from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor
from byte_tracker.tracker.byte_tracker import BYTETracker
import argparse
import cv2
from loguru import logger
import numpy as np
import time

from config import *
from pre_and_post_progress import *




def initRKNN(rknnModel, id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn_lite

def initRKNNs(rknnModel, TPEs):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list

def get_tracker_args():
    parser = argparse.ArgumentParser()
    # tracking args
    parser.add_argument('--track_thresh', type=float, default=0.5, help='tracking confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=30, help='the frames for keep lost tracks')
    parser.add_argument('--match_thresh', type=float, default=0.8, help='matching threshold for tracking')
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument('--mot20', dest='mot20', default=False, action='store_true', help='test mot20.')
    args = parser.parse_args()
    return args



class rknnPoolExecutor:
    def __init__(self, rknnModel, TPEs, func):
        self.TPEs = TPEs
        self.func = func
        self.rknnModel = rknnModel
        
        self.rknnPool = initRKNNs(self.rknnModel,3)
        
        # 跟踪器初始化
        self.tracker_args = get_tracker_args()
        self.trackers = {cls: BYTETracker(self.tracker_args) 
                        for cls in TARGET_CLASSES}
        # 多级流水线队列（各阶段独立）
        self.preprocess_queue = Queue(maxsize=TPEs * 3)  # 预处理 → NPU
        self.inference_queue = Queue(maxsize=TPEs * 3)   # NPU → 后处理
        self.postprocess_queue = Queue(maxsize=TPEs * 3) # 后处理 → CPU
        self.output_queue = Queue(maxsize=TPEs * 3) # 后处理 → CPU
        
        # CPU单线程队列（最终顺序控制）
        self.cpu_input_queue = Queue(maxsize=TPEs * 3)
        
        # 顺序控制
        self.next_display_id = 0
        self.frame_counter = 0
        self.pending_frames = {}  # 仅缓存错序的CPU处理结果
        
        # 线程池
        self.rknnPool = initRKNNs(self.rknnModel, TPEs)
        self.pre_pool = ThreadPoolExecutor(max_workers=TPEs)  # 预处理
        self.npu_pool = ThreadPoolExecutor(max_workers=TPEs)  # NPU推理
        self.post_pool = ThreadPoolExecutor(max_workers=TPEs) # 后处理
        self.cpu_pool = ThreadPoolExecutor(max_workers=1)     # CPU单线程

        self._start_workers()
        logger.success("RKNN Pool initialized")
    def _start_workers(self):
        # 启动各阶段工作线程
        for i in range(self.TPEs):
            self.pre_pool.submit(self._pre_worker)
            self.npu_pool.submit(self._npu_worker, self.rknnPool[i])
            self.post_pool.submit(self._post_worker)
        self.cpu_pool.submit(self._cpu_worker)
        
    def _pre_worker(self):
        """预处理线程：BGR→RGB + letterbox"""
        while True:
            frame, frame_id = self.preprocess_queue.get()
            if frame is None:
                break
            pre_time_start=time.time()
            try:
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_preprocessed, ratio, (dw, dh) = letterbox(img_rgb)
                self.inference_queue.put((img_preprocessed, ratio, (dw, dh), frame_id, frame))
            except Exception as e:
                logger.error(f"Pre-processing failed: {e}")
                # 直接提交原始帧到CPU队列（标记为失败）
                self.cpu_input_queue.put((frame, frame_id, {}, frame.shape[0], frame.shape[1], False))
            finally:
                self.preprocess_queue.task_done()
                logger.info(f"pre: {(time.time()-pre_time_start)*1000:.2f}ms")

    def _npu_worker(self, rknn_instance):
        """NPU推理线程"""
        while True:
            img_preprocessed, ratio, (dw, dh), frame_id, original_frame = self.inference_queue.get()
            if img_preprocessed is None:
                break
            
            inference_time_start=time.time()
            try:
                outputs = rknn_instance.inference(
                    inputs=[np.expand_dims(img_preprocessed, 0)],
                    data_format=['nhwc']
                )
                self.postprocess_queue.put((outputs, ratio, (dw, dh), frame_id, original_frame))
            except Exception as e:
                logger.error(f"NPU inference failed: {e}")
                self.cpu_input_queue.put((original_frame, frame_id, {}, original_frame.shape[0], original_frame.shape[1], False))
            finally:
                self.inference_queue.task_done()
                logger.info(f"inference: {(time.time()-inference_time_start)*1000:.2f}ms")

    def _post_worker(self):
        """后处理线程：解码 + scale_coords"""
        while True:
            outputs, ratio, (dw, dh), frame_id, original_frame = self.postprocess_queue.get()
            if outputs is None:
                break
            post_time_start=time.time()
            try:
                boxes, classes, scores = yolov8_post_process(outputs)
                class_dets = {cls: [] for cls in TARGET_CLASSES}
                if boxes is not None:
                    boxes = scale_coords(boxes, ratio, (dw, dh), original_frame.shape[:2])
                    for i in range(len(boxes)):
                        cls_name = CLASSES[int(classes[i])]
                        if cls_name in TARGET_CLASSES:
                            class_dets[cls_name].append([*boxes[i], scores[i]])
                
                # 提交给CPU处理（严格按顺序）
                self.cpu_input_queue.put((original_frame, frame_id, class_dets, *original_frame.shape[:2], True))
            except Exception as e:
                logger.error(f"Post-processing failed: {e}")
                self.cpu_input_queue.put((original_frame, frame_id, {}, *original_frame.shape[:2], False))
            finally:
                self.postprocess_queue.task_done()
                logger.info(f"post: {(time.time()-post_time_start)*1000:.2f}ms")

    def _cpu_worker(self):
        """CPU单线程：处理帧并保证顺序"""

                
        while True:
            # 优先检查是否有缓存中符合顺序的帧
            frame_data = self.pending_frames.pop(self.next_display_id, None)
            if frame_data:
                frame, class_dets, orig_h, orig_w, success = frame_data
                process_immediately = True
            else:
                # 获取新帧（阻塞操作，保证不会空转）
                frame, frame_id, class_dets, orig_h, orig_w, success = self.cpu_input_queue.get()
                
                # 检查帧顺序
                if frame_id == self.next_display_id:
                    process_immediately = True
                else:
                    # 乱序帧存入缓存
                    self.pending_frames[frame_id] = (frame, class_dets, orig_h, orig_w, success)
                    self.cpu_input_queue.task_done()
                    continue

            # 统一处理逻辑（唯一调用点）
            tracker_time_start = time.time()
            try:
                result = self.func(frame, class_dets, orig_h, orig_w, self.trackers, self.next_display_id) if success else (frame, {})
                self.output_queue.put((*result, success))
            except Exception as e:
                logger.error(f"Processing failed for frame {self.next_display_id}: {e}")
                self.output_queue.put((frame, {}, False))
            finally:
                if not frame_data:  # 只有从队列取的才需要task_done
                    self.cpu_input_queue.task_done()
                self.next_display_id += 1
                logger.debug(f"Frame {self.next_display_id-1} processed in {(time.time()-tracker_time_start)*1000:.2f}ms")

            # 紧急处理：防止pending_frames堆积导致内存溢出
            if len(self.pending_frames) > 20:
                oldest_id = min(self.pending_frames.keys())
                frame_data = self.pending_frames.pop(oldest_id)
                self.output_queue.put((frame_data[0], {}, False))  # 强制输出未处理的帧
                logger.warning(f"Cleared stalled frame {oldest_id} from buffer")
    def get(self):
        """获取处理后的帧（包含 success 标志）"""
        return self.output_queue.get()  # 返回 (frame, data, success)
    
    def put(self, frame):
        """提交帧到流水线"""
        self.preprocess_queue.put((frame, self.frame_counter))
        self.frame_counter += 1

    def release(self):
        """终止所有线程"""
        for _ in range(self.TPEs):
            self.preprocess_queue.put((None, -1))
            self.inference_queue.put((None, None, None, -1, None))
            self.postprocess_queue.put((None, None, None, -1, None))
        self.cpu_input_queue.put((None, -1, None, None, None, False))
        
        self.pre_pool.shutdown()
        self.npu_pool.shutdown()
        self.post_pool.shutdown()
        self.cpu_pool.shutdown()
        
        for rknn in self.rknnPool:
            rknn.release()
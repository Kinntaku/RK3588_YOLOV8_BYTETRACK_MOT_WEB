import psutil
import time

def monitor_usage():
    while True:
        # 获取CPU各个核心占用率
        cpu_percents = psutil.cpu_percent(percpu=True, interval=1)
        cpu_str = "CPU: " + ", ".join([f"Core{i}: {percent}%" for i, percent in enumerate(cpu_percents)])
        print(cpu_str)
        
        # 获取NPU占用率
        try:
            with open('/sys/kernel/debug/rknpu/load', 'r') as f:
                npu_load = f.read().strip()
            print(npu_load)
        except Exception as e:
            print(f"Error reading NPU load: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    monitor_usage()
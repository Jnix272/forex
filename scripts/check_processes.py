import psutil
import datetime

for pid in [4504, 31524]:
    try:
        p = psutil.Process(pid)
        create_time = datetime.datetime.fromtimestamp(p.create_time())
        uptime = datetime.datetime.now() - create_time
        mem = p.memory_info()
        print(f"PID {pid} ({p.name()}):")
        print(f"  Status: {p.status()}")
        print(f"  Created: {create_time} (Uptime: {uptime})")
        print(f"  CPU time: user={p.cpu_times().user:.2f}s, sys={p.cpu_times().system:.2f}s")
        print(f"  Memory: RSS={mem.rss/1024**2:.2f} MB, VMS={mem.vms/1024**2:.2f} MB")
        print(f"  Threads: {p.num_threads()}")
        print(f"  Handles: {p.num_handles()}")
    except Exception as e:
        print(f"PID {pid} error: {e}")

import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = "d:/forex-main"
with open("real_quick_test.log", "w") as f:
    subprocess.run(["d:/forex-main/.venv311/Scripts/python.exe", "-u", "-m", "training.train_gpu", "--config", "config/run.yaml", "--quick-mode"], env=env, stdout=f, stderr=subprocess.STDOUT)

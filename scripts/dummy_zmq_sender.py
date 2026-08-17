import json
import random
import time

import zmq


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    endpoint = "tcp://127.0.0.1:5555"
    print(f"Connecting to {endpoint}...")
    socket.connect(endpoint)

    batch_size = 1
    seq_len = 60
    n_features = 64
    total_elements = batch_size * seq_len * n_features

    tick = 0
    while True:
        tick += 1
        features = [random.uniform(-3.0, 3.0) for _ in range(total_elements)]

        payload = json.dumps({"features": features})
        socket.send_string(payload)

        print(f"Sent tick {tick} (size: {len(features)})")
        time.sleep(1.0)  # 1 tick per second


if __name__ == "__main__":
    main()

import pandas as pd
import time
import threading
import multiprocessing
import platform
import psutil

df = pd.read_csv("train.csv")
trip_durations = df['trip_duration']
splits = {
    "25%": trip_durations[:len(trip_durations) // 4],
    "50%": trip_durations[:len(trip_durations) // 2],
    "75%": trip_durations[:3 * len(trip_durations) // 4],
    "100%": trip_durations
}

def process_data(data):
    sorted_data = data.sort_values()
    filtered_data = sorted_data[sorted_data > 1000]
    return filtered_data

def sequential(data):
    start = time.time()
    result = process_data(data)
    end = time.time()
    return end - start

def threaded(data):
    result = []
    def worker():
        result.append(process_data(data))
    thread = threading.Thread(target=worker)
    start = time.time()
    thread.start()
    thread.join()
    end = time.time()
    return end - start

def multiprocessing_worker(data_chunk, return_dict, idx):
    return_dict[idx] = process_data(data_chunk)

def multiproc(data):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    cpu_count = multiprocessing.cpu_count()
    chunk_size = len(data) // cpu_count
    jobs = []
    start = time.time()
    for i in range(cpu_count):
        start_i = i * chunk_size
        end_i = len(data) if i == cpu_count - 1 else (i + 1) * chunk_size
        chunk = data.iloc[start_i:end_i]
        p = multiprocessing.Process(target=multiprocessing_worker, args=(chunk, return_dict, i))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    end = time.time()
    return end - start

if __name__ == "__main__":
    results = []
    for split_name, data in splits.items():
        print(f"Processing split: {split_name}")
        seq_time = sequential(data)
        th_time = threaded(data)
        mp_time = multiproc(data)
        results.append({
            "Split": split_name,
            "Sequential (s)": round(seq_time, 4),
            "Threading (s)": round(th_time, 4),
            "Multiprocessing (s)": round(mp_time, 4)
        })

    print("\n=== Performance Comparison ===")
    df_result = pd.DataFrame(results)
    print(df_result.to_string(index=False))

    print("\n=== System Info ===")
    print(f"Processor: {platform.processor()}")
    print(f"RAM: {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")

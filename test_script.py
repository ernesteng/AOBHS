import pandas as pd
from skmultiflow.data import DataStream
import time
from AOBHS_clean import AOBHSClassifier as AOBHS_clean
from scipy.io import arff
from skmultiflow.evaluation import EvaluatePrequential
import psutil
import threading

def start_memory_monitor():
    global peak_memory_usage, stop_monitoring, monitor_thread
    peak_memory_usage = 0
    stop_monitoring = threading.Event()
    
    def monitor():
        nonlocal_peak = 0
        process = psutil.Process()
        while not stop_monitoring.is_set():
            mem = process.memory_info().rss
            if mem > nonlocal_peak:
                nonlocal_peak = mem
            time.sleep(0.01)
        global peak_memory_usage
        peak_memory_usage = nonlocal_peak

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

def stop_memory_monitor():
    stop_monitoring.set()
    monitor_thread.join()
    return peak_memory_usage

filename = "data/kaggle_credit_card_1.csv"

if filename.lower().endswith('.csv'):
    df = pd.read_csv(filename)
elif filename.lower().endswith('.arff'):
    data, meta = arff.loadarff(filename)
    df = pd.DataFrame(data)

df[df.columns[-1]] = df[df.columns[-1]].astype('int64')
stream = DataStream(df)

AOBHS_clean = AOBHS_clean()

evaluator = EvaluatePrequential(max_samples=1000000,
                                n_wait=1000,
                                pretrain_size=0,
                                show_plot=True,
                                metrics=['precision','recall','gmean'],
                                )

start_memory_monitor()
time_begin = time.time()

evaluator.evaluate(stream=stream, model=[AOBHS_clean], model_names=['aobhs_clean'])

time_end = time.time()
peak_mem = stop_memory_monitor()

print(f"runtime: {time_end - time_begin}")
print(f"Peak memory usage: {peak_mem / 1024 / 1024:.2f} MB")

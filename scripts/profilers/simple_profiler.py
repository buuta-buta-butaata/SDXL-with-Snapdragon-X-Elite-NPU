import logging
import os
import psutil
import time

logger = logging.getLogger(__name__)

_profilers = {}

available = False

def register(name):
    if not available:
        return NullProfiler()
    
    if name in _profilers:
        logger.warning(f"Profiler entry '{name}' already exists. Overwriting.")
    _profilers[name] = Profiler()
    return _profilers[name]

def get(name):
    if not available:
        return NullProfiler()
    
    if not name in _profilers:
        logger.warning(f"Profiler entry '{name}' not found. Registering new entry.")
        register(name)
    return _profilers[name]

@property
def list():
    return list(_profilers);

class NullProfiler():
    def __init__(self):
        self.events = []
        self.recorded_times = {}
        pass

    def start_profile(self, name):
        pass

    def stop_profile(self, name):
        pass

    def destroy_profile_event(self):
        pass

    def exists(self, event):
        pass
        
    def event_elapsed_time(self, event):
        pass

class Profiler():
    def __init__(self):
        self.events = []
        self.recorded_times = {}
        self.start_profile("all")

    def start_profile(self, name):
        event_name = name + "-start"
        self.events.append(name)
        self._record(event_name)

    def stop_profile(self, name):
        event_name = name + "-stop"
        self._record(event_name)

    def _record(self, event_name):
        self.recorded_times[event_name] = time.perf_counter()

    def destroy_profile_event(self):
        self.stop_profile("all")

    def exists(self, event):
        return event + "-start" in self.recorded_times, event + "-stop" in self.recorded_times
        
    def event_elapsed_time(self, event):
        exists_start, exists_stop = self.exists(event)
        if exists_start:
            if not exists_stop:
                logger.warning(f"'{event}' event does not stop")
                return time.perf_counter() - self.recorded_times[event + "-start"]
        else:
            if not exists_stop:
                logger.warning(f"'{event}' event does not start")
                return 0
            else:
                logger.warning(f"'{event}' event does not exists")
                return 0
        return self.recorded_times[event + "-stop"] - self.recorded_times[event + "-start"]


def get_peak_memory_gb():
    process = psutil.Process(os.getpid())
    
    if os.name == 'nt':  # Windows環境
        info = process.memory_info()
        peak_bytes = info.peak_wset
    else:  # Linux / macOS 互換用の保険
        peak_bytes = process.memory_info().rss

    return peak_bytes / (1024 ** 3) # バイトからGBに変換

def get_current_memory_info():
    memory = psutil.virtual_memory()

    def _format(value):
        return f"{value / (1024 ** 3):.2f}"

    return [_format(x) for x in [memory.total, memory.used, memory.available]] + [memory.percent]


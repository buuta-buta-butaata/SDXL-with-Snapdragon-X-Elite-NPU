import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == lib_registration_name and ep_device.device.type == ort.OrtHardwareDeviceType.NPU]

# Configure and create session
ep_options = {'backend_path': qnn_ep.get_qnn_htp_path(),
              "enable_htp_fp16_precision" : "1",
              "htp_performance_mode": "burst",
              #"enable_htp_shared_memory_allocator": "1",
              #"htp_graph_finalization_optimization_mode": "1",
              }
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)
# session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session_options.log_severity_level = 3
session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session_options.enable_cpu_mem_arena = False
session_options.enable_mem_pattern = False
session_options.enable_mem_reuse = False

# Set run options for this specific inference
run_options = ort.RunOptions()
run_options.add_run_config_entry("qnn.perf_mode", "burst")
run_options.add_run_config_entry("qnn.rpc_control_latency", "100")

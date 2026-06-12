DEVICE_NAME = "Snapdragon X Elite CRD"
QAIRT_VERSION = "2.45"
COMPILE_OPTIONS = f"--qairt_version {QAIRT_VERSION} --qnn_options default_graph_htp_optimizations=O=3;default_graph_htp_precision=FLOAT16"
LINK_OPTIONS = f"--qairt_version {QAIRT_VERSION} --qnn_options default_graph_htp_optimizations=O=3;default_graph_htp_precision=FLOAT16"

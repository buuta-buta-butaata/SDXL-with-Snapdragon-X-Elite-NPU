import logging
import os

from utils import zip_utils, get_project_root

logger = logging.getLogger(__name__)

onnxextensions = {
    "url": r"https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Extensions/0.14.0",
    "filename": r"runtimes/win-arm64/native/ortextensions.dll",
    "target_dir": rf"{os.path.join(get_project_root(), 'lib')}"
}

def ensure_onnxextensions_lib():
    try:
        logger.info(f"Downloading from: {onnxextensions['url']}")
        zip_utils.download_file(onnxextensions["url"],
                                onnxextensions["target_dir"],
                                extensions=[".dll"],
                                filenames=[onnxextensions["filename"]])
        logger.info(f"Extracting: {onnxextensions['filename']} to {onnxextensions['target_dir']}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        
def is_onnxextensions_lib_available(path: str) -> bool:
    return os.path.isfile(path)
    
def is_onnx_tokenizer_available(path: str) -> bool:
    return os.path.isfile(path)


if __name__ == "__main__":
    print(is_onnx_tokenizer_available(r"..\compiled_models\single\dsxl\tokenizer\tokenizer.onnx"))

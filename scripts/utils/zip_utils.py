import io
import os
import urllib.request
import zipfile

def download_file(url: str, output_dir: str, extensions=None, filenames=None):
    with (
            urllib.request.urlopen(url) as res,
            io.BytesIO(res.read()) as bytes_io,
            zipfile.ZipFile(bytes_io) as zip_ref,
    ):
        extract(zip_ref, output_dir, extensions, filenames)


def extract_specific_files(zip_path, output_dir, extensions=None, filenames=None):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract(zip_ref, output_dir, extensions, filenames)

        
def extract(zip_ref, output_dir, extensions=None, filenames=None):
    for info in zip_ref.infolist():
        file_name = info.filename
        info.filename = os.path.basename(info.filename)
        
        if file_name.endswith('/'):
            continue

        if extensions and not any(file_name.lower().endswith(ext.lower()) for ext in extensions):
            continue

        if filenames and (os.path.basename(file_name) not in filenames and file_name not in filenames):
            continue

        zip_ref.extract(file_name, output_dir)

            
if __name__ == "__main__":
    zip_file_path = "onnxruntime_extensions_0.14.0.zip"
    output_folder = "lib"

    extract_specific_files(zip_file_path, output_folder,
                           extensions=[".dll"],
                           filenames=["runtimes/win-arm64/native/ortextensions.dll"])

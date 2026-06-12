import os
import zipfile
import shutil

def unzip(input_path, output_path):
    remove_target_dir = []

    with zipfile.ZipFile(input_path) as zf:
        zf.extractall()
        for file in zf.namelist():
            if not zf.getinfo(file).is_dir():
                shutil.move(file, output_path)
            else:
                remove_target_dir.append(file)

    for dir in remove_target_dir:
        os.rmdir(dir)

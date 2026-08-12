import urllib.request

def download_file(url: str, save_path: str):
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ ダウンロード完了: {save_path}")
    except Exception as e:
        print(f"❌ ダウンロード失敗: {e}")

# 使用例
# download_file(
#     "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Extensions/0.14.0",
#     "onnxruntime_extensions_0.14.0.zip"
# )

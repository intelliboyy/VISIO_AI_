# """Downloads YOLOX-Tiny and YOLOX-S ONNX models (Apache-2.0) into ./models"""
# import os, urllib.request
# BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
# os.makedirs("models", exist_ok=True)
# for n in ("yolox_tiny", "yolox_s"):
#     p = f"models/{n}.onnx"
#     if os.path.exists(p): print(n, "already present"); continue
#     print("Downloading", n, "..."); urllib.request.urlretrieve(BASE + n + ".onnx", p)
# print("Done.")


"""Downloads YOLOX (Apache-2.0) + YuNet/SFace face models (OpenCV Zoo) into ./models"""
import os, urllib.request
Y = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
Z = "https://github.com/opencv/opencv_zoo/raw/main/models/"
M = {"yolox_tiny.onnx": Y + "yolox_tiny.onnx", "yolox_s.onnx": Y + "yolox_s.onnx",
     "face_detection_yunet_2023mar.onnx": Z + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "face_recognition_sface_2021dec.onnx": Z + "face_recognition_sface/face_recognition_sface_2021dec.onnx"}
os.makedirs("models", exist_ok=True)
for n, u in M.items():
    p = "models/" + n
    if os.path.exists(p): print(n, "present"); continue
    print("Downloading", n, "..."); urllib.request.urlretrieve(u, p)
print("Done.")
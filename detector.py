# # # """YOLOX (Apache-2.0) ONNX detector. GPU (CUDA/DirectML) -> yolox_s, CPU -> yolox_tiny. Override: VISIO_MODEL=yolox_s"""
# # # import os, numpy as np, cv2, onnxruntime as ort

# # # NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}
# # # GRP = {0: "person", **{i: "vehicle" for i in (1, 2, 3, 5, 7)}, **{i: "animal" for i in (14, 15, 16, 17, 18, 19)}}
# # # MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# # # class Detector:
# # #     def __init__(s, name=None):
# # #         av = ort.get_available_providers()
# # #         pro = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in av] + ["CPUExecutionProvider"]
# # #         s.provider = pro[0]
# # #         s.name = name or os.environ.get("VISIO_MODEL") or ("yolox_s" if s.provider != "CPUExecutionProvider" else "yolox_tiny")
# # #         s.sess = ort.InferenceSession(os.path.join(MD, s.name + ".onnx"), providers=pro)
# # #         i = s.sess.get_inputs()[0]; s.inp = i.name
# # #         s.size = i.shape[2] if isinstance(i.shape[2], int) else (640 if s.name.endswith("_s") else 416)
# # #         g, st = [], []
# # #         for k in (8, 16, 32):
# # #             n = s.size // k; xv, yv = np.meshgrid(np.arange(n), np.arange(n))
# # #             g.append(np.stack((xv, yv), 2).reshape(-1, 2)); st.append(np.full((n * n, 1), k))
# # #         s.grid = np.concatenate(g).astype(np.float32); s.stride = np.concatenate(st).astype(np.float32)

# # #     def detect(s, frame, conf=0.45, nms=0.45):
# # #         h, w = frame.shape[:2]; r = min(s.size / h, s.size / w); nh, nw = int(h * r), int(w * r)
# # #         img = np.full((s.size, s.size, 3), 114, np.uint8); img[:nh, :nw] = cv2.resize(frame, (nw, nh))
# # #         o = s.sess.run(None, {s.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].copy()
# # #         o[:, :2] = (o[:, :2] + s.grid) * s.stride; o[:, 2:4] = np.exp(o[:, 2:4]) * s.stride
# # #         cls = o[:, 5:]; ids = cls.argmax(1); sc = o[:, 4] * cls[np.arange(len(o)), ids]
# # #         k = (sc > conf) & np.isin(ids, list(GRP))
# # #         if not k.any(): return []
# # #         b, sc, ids = o[k, :4], sc[k], ids[k]
# # #         xywh = np.stack([(b[:, 0] - b[:, 2] / 2) / r, (b[:, 1] - b[:, 3] / 2) / r, b[:, 2] / r, b[:, 3] / r], 1).astype(int)
# # #         keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, nms)
# # #         return [(GRP[ids[i]], NAMES[ids[i]], tuple(int(v) for v in xywh[i]), float(sc[i])) for i in np.array(keep).flatten()]


# # """YOLOX (Apache-2.0) ONNX detector. GPU (CUDA/DirectML) -> yolox_s, CPU -> yolox_tiny. Override: VISIO_MODEL=yolox_s"""
# # import os, numpy as np, cv2, onnxruntime as ort

# # NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 24: "backpack", 26: "handbag", 28: "suitcase"}
# # GRP = {0: "person", **{i: "vehicle" for i in (1, 2, 3, 5, 7)}, **{i: "animal" for i in (14, 15, 16, 17, 18, 19)}, **{i: "object" for i in (24, 26, 28)}}
# # MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# # class Detector:
# #     def __init__(s, name=None):
# #         av = ort.get_available_providers()
# #         pro = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in av] + ["CPUExecutionProvider"]
# #         s.provider = pro[0]
# #         s.name = name or os.environ.get("VISIO_MODEL") or ("yolox_s" if s.provider != "CPUExecutionProvider" else "yolox_tiny")
# #         s.sess = ort.InferenceSession(os.path.join(MD, s.name + ".onnx"), providers=pro)
# #         i = s.sess.get_inputs()[0]; s.inp = i.name
# #         s.size = i.shape[2] if isinstance(i.shape[2], int) else (640 if s.name.endswith("_s") else 416)
# #         g, st = [], []
# #         for k in (8, 16, 32):
# #             n = s.size // k; xv, yv = np.meshgrid(np.arange(n), np.arange(n))
# #             g.append(np.stack((xv, yv), 2).reshape(-1, 2)); st.append(np.full((n * n, 1), k))
# #         s.grid = np.concatenate(g).astype(np.float32); s.stride = np.concatenate(st).astype(np.float32)

# #     def detect(s, frame, conf=0.45, nms=0.45):
# #         h, w = frame.shape[:2]; r = min(s.size / h, s.size / w); nh, nw = int(h * r), int(w * r)
# #         img = np.full((s.size, s.size, 3), 114, np.uint8); img[:nh, :nw] = cv2.resize(frame, (nw, nh))
# #         o = s.sess.run(None, {s.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].copy()
# #         o[:, :2] = (o[:, :2] + s.grid) * s.stride; o[:, 2:4] = np.exp(o[:, 2:4]) * s.stride
# #         cls = o[:, 5:]; ids = cls.argmax(1); sc = o[:, 4] * cls[np.arange(len(o)), ids]
# #         k = (sc > conf) & np.isin(ids, list(GRP))
# #         if not k.any(): return []
# #         b, sc, ids = o[k, :4], sc[k], ids[k]
# #         xywh = np.stack([(b[:, 0] - b[:, 2] / 2) / r, (b[:, 1] - b[:, 3] / 2) / r, b[:, 2] / r, b[:, 3] / r], 1).astype(int)
# #         keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, nms)
# #         return [(GRP[ids[i]], NAMES[ids[i]], tuple(int(v) for v in xywh[i]), float(sc[i])) for i in np.array(keep).flatten()]


# """YOLOX (Apache-2.0) ONNX detector. GPU (CUDA/DirectML) -> yolox_s, CPU -> yolox_tiny. Override: VISIO_MODEL=yolox_s"""
# import os, numpy as np, cv2, onnxruntime as ort

# NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 24: "backpack", 26: "handbag", 28: "suitcase"}
# GRP = {0: "person", **{i: "vehicle" for i in (1, 2, 3, 5, 7)}, **{i: "animal" for i in (14, 15, 16, 17, 18, 19)}, **{i: "object" for i in (24, 26, 28)}}
# MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# class Detector:
#     def __init__(s, name=None):
#         av = ort.get_available_providers()
#         pro = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in av] + ["CPUExecutionProvider"]
#         s.provider = pro[0]
#         s.name = name or os.environ.get("VISIO_MODEL") or ("yolox_s" if s.provider != "CPUExecutionProvider" else "yolox_tiny")
#         so = ort.SessionOptions(); so.intra_op_num_threads = max(2, (os.cpu_count() or 4) // 2)
#         s.sess = ort.InferenceSession(os.path.join(MD, s.name + ".onnx"), sess_options=so, providers=pro)
#         i = s.sess.get_inputs()[0]; s.inp = i.name
#         s.size = i.shape[2] if isinstance(i.shape[2], int) else (640 if s.name.endswith("_s") else 416)
#         g, st = [], []
#         for k in (8, 16, 32):
#             n = s.size // k; xv, yv = np.meshgrid(np.arange(n), np.arange(n))
#             g.append(np.stack((xv, yv), 2).reshape(-1, 2)); st.append(np.full((n * n, 1), k))
#         s.grid = np.concatenate(g).astype(np.float32); s.stride = np.concatenate(st).astype(np.float32)

#     def detect(s, frame, conf=0.45, nms=0.45):
#         h, w = frame.shape[:2]; r = min(s.size / h, s.size / w); nh, nw = int(h * r), int(w * r)
#         img = np.full((s.size, s.size, 3), 114, np.uint8); img[:nh, :nw] = cv2.resize(frame, (nw, nh))
#         o = s.sess.run(None, {s.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].copy()
#         o[:, :2] = (o[:, :2] + s.grid) * s.stride; o[:, 2:4] = np.exp(o[:, 2:4]) * s.stride
#         cls = o[:, 5:]; ids = cls.argmax(1); sc = o[:, 4] * cls[np.arange(len(o)), ids]
#         k = (sc > conf) & np.isin(ids, list(GRP))
#         if not k.any(): return []
#         b, sc, ids = o[k, :4], sc[k], ids[k]
#         xywh = np.stack([(b[:, 0] - b[:, 2] / 2) / r, (b[:, 1] - b[:, 3] / 2) / r, b[:, 2] / r, b[:, 3] / r], 1).astype(int)
#         keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, nms)
#         return [(GRP[ids[i]], NAMES[ids[i]], tuple(int(v) for v in xywh[i]), float(sc[i])) for i in np.array(keep).flatten()]


"""YOLOX (Apache-2.0) ONNX detector. GPU (CUDA/DirectML) -> yolox_s, CPU -> yolox_tiny. Override: VISIO_MODEL=yolox_s"""
import os, numpy as np, cv2, onnxruntime as ort

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 24: "backpack", 26: "handbag", 28: "suitcase"}
GRP = {0: "person", **{i: "vehicle" for i in (1, 2, 3, 5, 7)}, **{i: "animal" for i in (14, 15, 16, 17, 18, 19)}, **{i: "object" for i in (24, 26, 28)}}
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

class Detector:
    def __init__(s, name=None):
        av = ort.get_available_providers()
        pro = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in av] + ["CPUExecutionProvider"]
        s.provider = pro[0]
        if s.provider == "CPUExecutionProvider": print("WARNING: running on CPU. For GPU: pip uninstall onnxruntime && pip install onnxruntime-gpu")
        s.name = name or os.environ.get("VISIO_MODEL") or ("yolox_s" if s.provider != "CPUExecutionProvider" else "yolox_tiny")
        so = ort.SessionOptions(); so.intra_op_num_threads = 1 if s.provider == "CUDAExecutionProvider" else max(2, (os.cpu_count() or 4) // 2)
        s.sess = ort.InferenceSession(os.path.join(MD, s.name + ".onnx"), sess_options=so, providers=pro)
        i = s.sess.get_inputs()[0]; s.inp = i.name
        s.size = i.shape[2] if isinstance(i.shape[2], int) else (640 if s.name.endswith("_s") else 416)
        g, st = [], []
        for k in (8, 16, 32):
            n = s.size // k; xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g.append(np.stack((xv, yv), 2).reshape(-1, 2)); st.append(np.full((n * n, 1), k))
        s.grid = np.concatenate(g).astype(np.float32); s.stride = np.concatenate(st).astype(np.float32)
        z = np.zeros((1, 3, s.size, s.size), np.float32)
        for _ in range(3): s.sess.run(None, {s.inp: z})  # warm-up: first CUDA run is slow

    def detect(s, frame, conf=0.45, nms=0.45):
        h, w = frame.shape[:2]; r = min(s.size / h, s.size / w); nh, nw = int(h * r), int(w * r)
        img = np.full((s.size, s.size, 3), 114, np.uint8); img[:nh, :nw] = cv2.resize(frame, (nw, nh))
        o = s.sess.run(None, {s.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].copy()
        o[:, :2] = (o[:, :2] + s.grid) * s.stride; o[:, 2:4] = np.exp(o[:, 2:4]) * s.stride
        cls = o[:, 5:]; ids = cls.argmax(1); sc = o[:, 4] * cls[np.arange(len(o)), ids]
        k = (sc > conf) & np.isin(ids, list(GRP))
        if not k.any(): return []
        b, sc, ids = o[k, :4], sc[k], ids[k]
        xywh = np.stack([(b[:, 0] - b[:, 2] / 2) / r, (b[:, 1] - b[:, 3] / 2) / r, b[:, 2] / r, b[:, 3] / r], 1).astype(int)
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, nms)
        return [(GRP[ids[i]], NAMES[ids[i]], tuple(int(v) for v in xywh[i]), float(sc[i])) for i in np.array(keep).flatten()]
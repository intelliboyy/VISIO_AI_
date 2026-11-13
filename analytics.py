# """Tracking + rule analytics: intrusion, line crossing, loitering, crowd, unattended, fall, fire, faces."""
# import os, math, threading, numpy as np, cv2
# MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# COL = {"person": (0, 255, 0), "vehicle": (255, 160, 0), "animal": (200, 80, 255), "object": (0, 200, 255)}

# def inside(poly, p): return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
# def ccw(a, b, c): return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
# def seg_hit(p, q, a, b): return ccw(p, a, b)*ccw(q, a, b) < 0 and ccw(p, q, a)*ccw(p, q, b) < 0
# def side(a, b, p): return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

# class Faces:
#     """YuNet detection + SFace recognition (OpenCV Zoo). Needs the two models in ./models"""
#     def __init__(s):
#         s.ok = False; s.db = []; s.lk = threading.Lock()
#         d, r = (os.path.join(MD, n) for n in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))
#         if os.path.exists(d) and os.path.exists(r):
#             s.det = cv2.FaceDetectorYN.create(d, "", (320, 320), 0.8, 0.3, 5000); s.rec = cv2.FaceRecognizerSF.create(r, ""); s.ok = True
#         else: print("Face models missing -> run python download_models.py (face recognition disabled)")
#     def _faces(s, img):
#         s.det.setInputSize((img.shape[1], img.shape[0])); _, f = s.det.detect(img); return [] if f is None else list(f)
#     def feat(s, img):
#         with s.lk:
#             f = s._faces(img) if s.ok else []
#             return s.rec.feature(s.rec.alignCrop(img, max(f, key=lambda r: r[2]*r[3]))) if f else None
#     def scan(s, img):
#         out = []
#         if not s.ok: return out
#         with s.lk:
#             for row in s._faces(img):
#                 ft = s.rec.feature(s.rec.alignCrop(img, row)); best = ("Unknown", "", 0.0)
#                 for n, c, e in s.db:
#                     sc = s.rec.match(ft, e, cv2.FaceRecognizerSF_FR_COSINE)
#                     if sc > 0.363 and sc > best[2]: best = (n, c, sc)
#                 out.append((tuple(int(v) for v in row[:4]),) + best)
#         return out

# class Analyzer:
#     def __init__(s): s.tr = {}; s.n = 0; s.cin = 0; s.cout = 0; s.pm = None; s.fire = 0
#     def _track(s, dets, now):
#         used = set()
#         for grp, lab, (x, y, w, h), sc in dets:
#             c = (x + w/2, y + h/2); best = None; bd = 1e9
#             for i, t in s.tr.items():
#                 d = math.hypot(c[0]-t["c"][0], c[1]-t["c"][1])
#                 if i not in used and t["g"] == grp and d < bd and d < max(w, h, 40): best, bd = i, d
#             if best is None:
#                 s.n += 1; best = s.n; s.tr[best] = dict(id=s.n, g=grp, zin={}, flag=set(), feet=None, c=c, c0=c, static=now, up=False, lying=None)
#             t = s.tr[best]; used.add(best); t.update(prev=t["feet"], c=c, feet=(x + w/2, y + h), box=(x, y, w, h), seen=now, lab=lab)
#             if math.hypot(c[0]-t["c0"][0], c[1]-t["c0"][1]) > 25: t["c0"] = c; t["static"] = now
#         for i in [i for i, t in s.tr.items() if now - t.get("seen", now) > 3]: del s.tr[i]
#         return [s.tr[i] for i in used]

#     def step(s, f, dets, modes, cfg, now, faces=None):
#         H, W = f.shape[:2]; M = set(modes); boxes, ev, cross = [], [], []; T = s._track(dets, now)
#         zones = [np.array([[p[0]*W, p[1]*H] for p in z["pts"]], np.int32) for z in cfg.get("zones", []) if len(z.get("pts", [])) > 2] \
#             or [np.array([[0, 0], [W, 0], [W, H], [0, H]], np.int32)]  # no zone drawn -> full view
#         lines = [((l["a"][0]*W, l["a"][1]*H), (l["b"][0]*W, l["b"][1]*H), l.get("flip", False)) for l in cfg.get("lines", [])]
#         ppl = [t for t in T if t["g"] == "person"]; mv = [t for t in T if t["g"] == "person" or (t["g"] == "vehicle" and "vehicle" in M)]
#         for t in T: x, y, w, h = t["box"]; boxes.append((x, y, w, h, COL[t["g"]], f'{t["lab"]} #{t["id"]}'))
#         for g in ("person", "vehicle", "animal"):
#             n = sum(t["g"] == g for t in T)
#             if g in M and n: ev.append((g, f"{n} {g}(s) detected"))
#         for t in mv:  # zone intrusion + loitering (feet point)
#             zi = next((i for i, zp in enumerate(zones) if inside(zp, t["feet"])), None)
#             if zi is None: t["zin"] = {}; t["flag"] -= {"I", "L"}; continue
#             t["zin"].setdefault(zi, now)
#             if "intrusion" in M and "I" not in t["flag"]: t["flag"].add("I"); ev.append(("intrusion", f'{t["g"]} #{t["id"]} entered restricted zone'))
#             if "loitering" in M and "L" not in t["flag"] and now - t["zin"][zi] >= cfg.get("loiter", 30):
#                 t["flag"].add("L"); ev.append(("loitering", f'{t["g"]} #{t["id"]} loitering > {cfg.get("loiter", 30)}s'))
#         if "crowd" in M:
#             for zp in zones:
#                 n = sum(inside(zp, t["feet"]) for t in ppl)
#                 if n >= cfg.get("crowd", 8): ev.append(("crowd", f"{n} people (threshold {cfg.get('crowd', 8)})")); break
#         if "line" in M:
#             for t in mv:
#                 if not t["prev"]: continue
#                 for a, b, flip in lines:
#                     if seg_hit(t["prev"], t["feet"], a, b):
#                         d = (side(a, b, t["feet"]) > 0) != flip; cross.append("in" if d else "out")
#                         if d: s.cin += 1
#                         else: s.cout += 1
#         if "fall" in M:
#             for t in ppl:
#                 x, y, w, h = t["box"]
#                 if h > w * 1.3: t["up"] = True; t["lying"] = None; t["flag"].discard("F")
#                 elif w > h * 1.2 and t["up"]:
#                     t["lying"] = t["lying"] or now
#                     if now - t["lying"] >= 2 and "F" not in t["flag"]: t["flag"].add("F"); ev.append(("fall", f'Person #{t["id"]} fell / lying down'))
#         if "unattended" in M:
#             for t in T:
#                 if t["g"] != "object": continue
#                 if any(math.hypot(t["c"][0]-p["c"][0], t["c"][1]-p["c"][1]) < 250 for p in ppl): t["static"] = now
#                 elif now - t["static"] >= cfg.get("unatt", 60) and "U" not in t["flag"]: t["flag"].add("U"); ev.append(("unattended", f'Unattended {t["lab"]}'))
#         if "fire" in M:
#             m = cv2.inRange(cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2HSV), (0, 120, 200), (35, 255, 255))
#             r = m.mean() / 255; fl = (m != s.pm).mean() if s.pm is not None else 0; s.pm = m
#             s.fire = s.fire + 1 if (0.004 < r < 0.5 and fl > 0.003) else 0  # flame colour + flicker
#             if s.fire >= 3: ev.append(("fire", f"Flame-like flicker ({r*100:.1f}% of view)"))
#         if "face" in M and faces:
#             for (x, y, w, h), name, cat, sc in faces.scan(f):
#                 wl = cat == "Watchlist"; boxes.append((x, y, w, h, (0, 0, 255) if wl else (255, 200, 0), name))
#                 if wl: ev.append(("watchlist", f"Watchlist match: {name}"))
#                 elif name == "Unknown": ev.append(("face", "Unknown face"))
#         return boxes, ev, cross, len(ppl)

# def overlay(v, cfg, an, modes):
#     H, W = v.shape[:2]
#     for z in cfg.get("zones", []):
#         if len(z.get("pts", [])) > 2: cv2.polylines(v, [np.array([[a*W, b*H] for a, b in z["pts"]], np.int32)], True, (0, 0, 255), 2)
#     for l in cfg.get("lines", []):
#         a = (int(l["a"][0]*W), int(l["a"][1]*H)); b = (int(l["b"][0]*W), int(l["b"][1]*H)); cv2.line(v, a, b, (0, 255, 255), 2)
#         dx, dy = b[0]-a[0], b[1]-a[1]; L = math.hypot(dx, dy) or 1; k = -1 if l.get("flip") else 1; m = ((a[0]+b[0])//2, (a[1]+b[1])//2)
#         cv2.arrowedLine(v, m, (int(m[0] - dy/L*40*k), int(m[1] + dx/L*40*k)), (0, 255, 0), 2)
#     if "line" in modes: cv2.putText(v, f"IN {an.cin}  OUT {an.cout}", (8, 22), 0, 0.7, (0, 255, 255), 2)

"""Tracking + rule analytics: intrusion, line crossing, loitering, crowd, unattended, fall, fire, faces."""
import os, math, threading, numpy as np, cv2
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
COL = {"person": (0, 255, 0), "vehicle": (255, 160, 0), "animal": (200, 80, 255), "object": (0, 200, 255)}

def inside(poly, p): return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0
def ccw(a, b, c): return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
def seg_hit(p, q, a, b): return ccw(p, a, b)*ccw(q, a, b) < 0 and ccw(p, q, a)*ccw(p, q, b) < 0
def side(a, b, p): return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

class Faces:
    """YuNet detection + SFace recognition (OpenCV Zoo). Needs the two models in ./models"""
    def __init__(s):
        s.ok = False; s.db = []; s.lk = threading.Lock()
        d, r = (os.path.join(MD, n) for n in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"))
        if os.path.exists(d) and os.path.exists(r):
            s.det = cv2.FaceDetectorYN.create(d, "", (320, 320), 0.6, 0.3, 5000); s.rec = cv2.FaceRecognizerSF.create(r, ""); s.ok = True
        else: print("Face models missing -> run python download_models.py (face recognition disabled)")
    def _faces(s, img):
        s.det.setInputSize((img.shape[1], img.shape[0])); _, f = s.det.detect(img); return [] if f is None else list(f)
    def feat(s, img):
        with s.lk:
            f = s._faces(img) if s.ok else []
            return s.rec.feature(s.rec.alignCrop(img, max(f, key=lambda r: r[2]*r[3]))) if f else None
    def scan(s, img, minw=0):
        """Detect + recognise. If img is narrower than minw it is upscaled (x4 max) so small/distant faces get enough pixels."""
        out = []
        if not s.ok: return out
        k = min(4.0, minw / img.shape[1]) if minw > img.shape[1] else 1.0
        if k > 1: img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        with s.lk:
            for row in s._faces(img):
                ft = s.rec.feature(s.rec.alignCrop(img, row)); best = ("Unknown", "", 0.0)
                for n, c, e in s.db:
                    sc = s.rec.match(ft, e, cv2.FaceRecognizerSF_FR_COSINE)
                    if sc > 0.363 and sc > best[2]: best = (n, c, sc)
                out.append((tuple(float(v) / k for v in row[:4]),) + best)
        return out

def nm(t):
    w = t.get("who")
    if w and w[0] != "Unknown": return w[0]
    return f'Person #{t["id"]}' if t["g"] == "person" else f'{t["lab"].capitalize()} #{t["id"]}'

class Analyzer:
    def __init__(s): s.tr = {}; s.n = 0; s.cin = 0; s.cout = 0; s.pm = None; s.fire = 0; s.k = 0; s.fb = []; s.fbt = 0
    def _track(s, dets, now):
        used = set()
        for grp, lab, (x, y, w, h), sc in dets:
            c = (x + w/2, y + h/2); best = None; bd = 1e9
            for i, t in s.tr.items():
                d = math.hypot(c[0]-t["c"][0], c[1]-t["c"][1])
                if i not in used and t["g"] == grp and d < bd and d < max(w, h, 40): best, bd = i, d
            if best is None:
                s.n += 1; best = s.n; s.tr[best] = dict(id=s.n, born=now, who=None, g=grp, zin={}, flag=set(), feet=None, c=c, c0=c, static=now, up=False, lying=None)
            t = s.tr[best]; used.add(best); t.update(prev=t["feet"], c=c, feet=(x + w/2, y + h), box=(x, y, w, h), seen=now, lab=lab)
            if math.hypot(c[0]-t["c0"][0], c[1]-t["c0"][1]) > 25: t["c0"] = c; t["static"] = now
        for i in [i for i, t in s.tr.items() if now - t.get("seen", now) > 3]: del s.tr[i]
        return [s.tr[i] for i in used]

    @staticmethod
    def wait(t, M, faces, now):  # hold an intrusion alert up to 2 s so the face has time to be identified
        return "face" in M and faces and faces.ok and t["g"] == "person" and t["who"] is None and now - t["born"] < 2.0

    def _ident(s, faces, full, W, ppl, boxes, ev, now):
        sx = full.shape[1] / W; hits = []
        for t in ppl:  # head crop from the FULL-resolution frame, upscaled -> recognises faces far from the camera
            if (t["who"] and t["who"][0] != "Unknown") or (s.k + t["id"]) % 2: continue
            x, y, w, h = t["box"]; x0, y0, x1, y1 = (max(0, int(v * sx)) for v in (x - .1*w, y - .05*h, x + 1.1*w, y + .45*h)); c = full[y0:y1, x0:x1]
            if c.size: hits += [(((x0 + b[0]) / sx, (y0 + b[1]) / sx, b[2] / sx, b[3] / sx), n, ct, sc, t) for b, n, ct, sc in faces.scan(c, 320)]
        if s.k % 3 == 0:  # whole-frame pass catches faces the person detector missed
            k = min(1.0, 1920 / full.shape[1]); im = cv2.resize(full, None, fx=k, fy=k) if k < 1 else full
            for b, n, ct, sc in faces.scan(im):
                b = [v / k / sx for v in b]; cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                t = next((t for t in ppl if t["box"][0] <= cx <= t["box"][0] + t["box"][2] and t["box"][1] <= cy <= t["box"][1] + t["box"][3]), None)
                hits.append((b, n, ct, sc, t))
        fb = []
        for b, n, ct, sc, t in hits:
            wl = ct == "Watchlist"; fb.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), (0, 0, 255) if wl else (255, 200, 0), n))
            if t is not None and (t["who"] is None or t["who"][0] == "Unknown" or sc > t["who"][2]): t["who"] = (n, ct, sc)
            if wl: ev.append(("watchlist", f"Watchlist match: {n}"))
            elif n == "Unknown": ev.append(("face", "Unknown face"))
        if fb: s.fb, s.fbt = fb, now
        if now - s.fbt < 1.0: boxes += s.fb

    def step(s, f, dets, modes, cfg, now, faces=None, full=None):
        H, W = f.shape[:2]; M = set(modes); boxes, ev, cross = [], [], []; T = s._track(dets, now)
        zones = [np.array([[p[0]*W, p[1]*H] for p in z["pts"]], np.int32) for z in cfg.get("zones", []) if len(z.get("pts", [])) > 2] \
            or [np.array([[0, 0], [W, 0], [W, H], [0, H]], np.int32)]  # no zone drawn -> full view
        zn = [z.get("name", "Zone") for z in cfg.get("zones", []) if len(z.get("pts", [])) > 2] or ["the view"]
        lines = [((l["a"][0]*W, l["a"][1]*H), (l["b"][0]*W, l["b"][1]*H), l.get("flip", False)) for l in cfg.get("lines", [])]
        ppl = [t for t in T if t["g"] == "person"]; mv = [t for t in T if t["g"] == "person" or (t["g"] == "vehicle" and "vehicle" in M)]
        s.k += 1
        if "face" in M and faces and faces.ok and full is not None: s._ident(faces, full, W, ppl, boxes, ev, now)
        for t in T: x, y, w, h = t["box"]; boxes.append((x, y, w, h, COL[t["g"]], nm(t)))
        for g in ("person", "vehicle", "animal"):
            n = sum(t["g"] == g for t in T)
            if g in M and n: ev.append((g, f"{n} {g}(s) detected"))
        for t in mv:  # zone intrusion + loitering (feet point)
            zi = next((i for i, zp in enumerate(zones) if inside(zp, t["feet"])), None)
            if zi is None: t["zin"] = {}; t["flag"] -= {"I", "L"}; continue
            t["zin"].setdefault(zi, now)
            if "intrusion" in M and "I" not in t["flag"] and not s.wait(t, M, faces, now): t["flag"].add("I"); ev.append(("intrusion", f'{nm(t)} intruded into {zn[zi]}'))
            if "loitering" in M and "L" not in t["flag"] and now - t["zin"][zi] >= cfg.get("loiter", 30):
                t["flag"].add("L"); ev.append(("loitering", f'{nm(t)} loitering in {zn[zi]} > {cfg.get("loiter", 30)}s'))
        if "crowd" in M:
            for zp in zones:
                n = sum(inside(zp, t["feet"]) for t in ppl)
                if n >= cfg.get("crowd", 8): ev.append(("crowd", f"{n} people (threshold {cfg.get('crowd', 8)})")); break
        if "line" in M:
            for t in mv:
                if not t["prev"]: continue
                for a, b, flip in lines:
                    if seg_hit(t["prev"], t["feet"], a, b):
                        d = (side(a, b, t["feet"]) > 0) != flip; cross.append("in" if d else "out")
                        if d: s.cin += 1
                        else: s.cout += 1
        if "fall" in M:
            for t in ppl:
                x, y, w, h = t["box"]
                if h > w * 1.3: t["up"] = True; t["lying"] = None; t["flag"].discard("F")
                elif w > h * 1.2 and t["up"]:
                    t["lying"] = t["lying"] or now
                    if now - t["lying"] >= 2 and "F" not in t["flag"]: t["flag"].add("F"); ev.append(("fall", f'{nm(t)} fell / lying down'))
        if "unattended" in M:
            for t in T:
                if t["g"] != "object": continue
                if any(math.hypot(t["c"][0]-p["c"][0], t["c"][1]-p["c"][1]) < 250 for p in ppl): t["static"] = now
                elif now - t["static"] >= cfg.get("unatt", 60) and "U" not in t["flag"]: t["flag"].add("U"); ev.append(("unattended", f'Unattended {t["lab"]}'))
        if "fire" in M:
            m = cv2.inRange(cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2HSV), (0, 120, 200), (35, 255, 255))
            r = m.mean() / 255; fl = (m != s.pm).mean() if s.pm is not None else 0; s.pm = m
            s.fire = s.fire + 1 if (0.004 < r < 0.5 and fl > 0.003) else 0  # flame colour + flicker
            if s.fire >= 3: ev.append(("fire", f"Flame-like flicker ({r*100:.1f}% of view)"))
        return boxes, ev, cross, len(ppl)

def overlay(v, cfg, an, modes):
    H, W = v.shape[:2]
    for z in cfg.get("zones", []):
        if len(z.get("pts", [])) > 2: cv2.polylines(v, [np.array([[a*W, b*H] for a, b in z["pts"]], np.int32)], True, (0, 0, 255), 2)
    for l in cfg.get("lines", []):
        a = (int(l["a"][0]*W), int(l["a"][1]*H)); b = (int(l["b"][0]*W), int(l["b"][1]*H)); cv2.line(v, a, b, (0, 255, 255), 2)
        dx, dy = b[0]-a[0], b[1]-a[1]; L = math.hypot(dx, dy) or 1; k = -1 if l.get("flip") else 1; m = ((a[0]+b[0])//2, (a[1]+b[1])//2)
        cv2.arrowedLine(v, m, (int(m[0] - dy/L*40*k), int(m[1] + dx/L*40*k)), (0, 255, 0), 2)
    if "line" in modes: cv2.putText(v, f"IN {an.cin}  OUT {an.cout}", (8, 22), 0, 0.7, (0, 255, 255), 2)
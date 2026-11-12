"""Visio AI VMS - prototype. Run: python server.py [--port 8090]"""
import os, sys, time, json, hashlib, secrets, sqlite3, threading
from urllib.parse import quote
import cv2, numpy as np, uvicorn
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

DATA = os.environ.get("VISIO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
SNAPS = os.path.join(DATA, "snapshots"); os.makedirs(SNAPS, exist_ok=True)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ---- brand RTSP templates (Section 7 of the guide) ----
_DH = "rtsp://{u}:{p}@{ip}:{port}/cam/realmonitor?channel={ch}&subtype={s}"
BRANDS = {
    "Hikvision / Prama / HiLook": "rtsp://{u}:{p}@{ip}:{port}/Streaming/Channels/{ch}0{s1}",
    "Dahua": _DH, "CP Plus (IP / NVR / XVR)": _DH, "Amcrest / Lorex / Imou": _DH, "Honeywell": _DH,
    "CP Plus legacy DVR (XMEye)": "rtsp://{ip}:{port}/user={u}&password={p}&channel={ch}&stream={s}.sdp?",
    "Uniview": "rtsp://{u}:{p}@{ip}:{port}/unicast/c{ch}/s{s}/live",
    "Axis": "rtsp://{u}:{p}@{ip}/axis-media/media.amp",
    "Hanwha Wisenet": "rtsp://{u}:{p}@{ip}:{port}/profile2/media.smp",
    "Bosch": "rtsp://{u}:{p}@{ip}:{port}/?h26x=4&line=1&inst={s1}",
    "Reolink": "rtsp://{u}:{p}@{ip}:{port}/h264Preview_{ch:02d}_{sn}",
    "TP-Link Tapo / VIGI": "rtsp://{u}:{p}@{ip}:{port}/stream{s1}",
    "EZVIZ": "rtsp://admin:{p}@{ip}:{port}/h264/ch{ch}/{sn}/av_stream",
}

def build_url(brand, ip, port=554, user="admin", pw="", ch=1, stream="sub"):
    sub = stream == "sub"
    return BRANDS[brand].format(u=quote(user, safe=""), p=quote(pw, safe=""), ip=ip, port=port, ch=int(ch),
                                s=1 if sub else 0, s1=2 if sub else 1, sn="sub" if sub else "main")

def source_of(c):
    t = c["source_type"]
    if t == "webcam": return int(c["ip"] or 0)
    if t == "custom": return c["url"]
    return build_url(c["brand"], c["ip"], c["port"], c["user"], c["pw"], c["ch"], c["stream"])

# ---- storage ----
db = sqlite3.connect(os.path.join(DATA, "visio.db"), check_same_thread=False); db.row_factory = sqlite3.Row
lock = threading.Lock()
db.executescript("""
CREATE TABLE IF NOT EXISTS cameras(id INTEGER PRIMARY KEY, name, source_type, brand, ip, port, user, pw, ch, stream, url, modes, fps);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, cam_id, cam_name, type, label, sev, snap, ack DEFAULT 0);
CREATE TABLE IF NOT EXISTS users(name PRIMARY KEY, pw, role);""")
def hp(p): return hashlib.sha256(("visio" + p).encode()).hexdigest()
if not db.execute("SELECT 1 FROM users").fetchone():
    db.execute("INSERT INTO users VALUES('admin',?, 'admin')", (hp("admin123"),)); db.commit()
def q(sql, a=()):
    with lock:
        cur = db.execute(sql, a); db.commit(); return [dict(r) for r in cur.fetchall()], cur.lastrowid

# ---- AI worker per camera ----
hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
SEV = {"motion": "low", "person": "medium", "tamper": "high"}

class Cam(threading.Thread):
    def __init__(s, row):
        super().__init__(daemon=True); s.row = row; s.jpg = None; s.status = "connecting"; s.run_flag = True
        s.boxes = []; s.last = {}; s.mog = cv2.createBackgroundSubtractorMOG2(300, 32, False)
        s.modes = json.loads(row["modes"] or "[]"); s.n_persons = 0
    def event(s, typ, label, frame):
        if time.time() - s.last.get(typ, 0) < 15: return  # cool-down
        s.last[typ] = time.time(); fn = f"{s.row['id']}_{int(time.time()*1000)}.jpg"
        cv2.imwrite(os.path.join(SNAPS, fn), frame)
        q("INSERT INTO events(ts,cam_id,cam_name,type,label,sev,snap) VALUES(?,?,?,?,?,?,?)",
          (time.time(), s.row["id"], s.row["name"], typ, label, SEV[typ], fn))
    def analyse(s, f):
        boxes = []; g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if "tamper" in s.modes and (g.mean() < 12 or cv2.Laplacian(g, cv2.CV_64F).var() < 4):
            s.event("tamper", "Camera covered / blinded", f)
        if "motion" in s.modes:
            m = cv2.threshold(s.mog.apply(cv2.resize(g, (320, 180))), 200, 255, cv2.THRESH_BINARY)[1]
            m = cv2.dilate(m, None, iterations=2)
            cs = [c for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 250]
            if cs:
                for c in cs: x, y, w, h = cv2.boundingRect(c); boxes.append((x*2, y*2, w*2, h*2, (0, 200, 255), "motion"))
                s.event("motion", f"Motion ({len(cs)} regions)", f)
        s.n_persons = 0
        if "person" in s.modes:
            rects, _ = hog.detectMultiScale(f, winStride=(8, 8), padding=(8, 8), scale=1.08)
            s.n_persons = len(rects)
            for x, y, w, h in rects: boxes.append((x, y, w, h, (0, 255, 0), "person"))
            if len(rects): s.event("person", f"{len(rects)} person(s) detected", f)
        s.boxes = boxes
    def run(s):
        step = 1.0 / max(1, int(s.row["fps"] or 3)); last_ai = 0
        while s.run_flag:
            try: cap = cv2.VideoCapture(source_of(s.row))
            except Exception: cap = None
            if not cap or not cap.isOpened(): s.status = "offline"; time.sleep(4); continue
            s.status = "online"
            while s.run_flag:
                ok, f = cap.read()
                if not ok: break
                f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
                if time.time() - last_ai >= step: last_ai = time.time(); s.analyse(f.copy())
                v = f.copy()
                for x, y, w, h, col, lab in s.boxes: cv2.rectangle(v, (x, y), (x+w, y+h), col, 2); cv2.putText(v, lab, (x, max(12, y-4)), 0, 0.5, col, 1)
                cv2.putText(v, time.strftime("%Y-%m-%d %H:%M:%S"), (8, v.shape[0]-8), 0, 0.5, (255, 255, 255), 1)
                s.jpg = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                if isinstance(source_of(s.row), str) and source_of(s.row).endswith((".mp4", ".avi")): time.sleep(0.03)
            cap.release(); s.status = "offline"; time.sleep(2)  # auto-reconnect

cams = {}
def start_cam(row): c = Cam(row); cams[row["id"]] = c; c.start()
for r in q("SELECT * FROM cameras")[0]: start_cam(r)

# ---- API ----
app = FastAPI(title="Visio AI VMS prototype"); tokens = {}
def auth(req: Request):
    if req.cookies.get("tok") not in tokens: raise HTTPException(401)

@app.post("/api/login")
async def login(req: Request, resp: Response):
    b = await req.json(); u = q("SELECT * FROM users WHERE name=? AND pw=?", (b.get("user"), hp(b.get("pass", ""))))[0]
    if not u: raise HTTPException(401)
    t = secrets.token_hex(16); tokens[t] = u[0]["name"]; resp.set_cookie("tok", t, httponly=True); return {"ok": True}

@app.get("/api/brands")
def brands(req: Request): auth(req); return list(BRANDS)

@app.get("/api/cameras")
def cameras(req: Request):
    auth(req); out = []
    for r in q("SELECT * FROM cameras")[0]:
        c = cams.get(r["id"]); r.pop("pw"); r["status"] = c.status if c else "?"; r["persons"] = c.n_persons if c else 0; out.append(r)
    return out

@app.post("/api/cameras")
async def add_camera(req: Request):
    auth(req); b = await req.json()
    if b["source_type"] == "rtsp" and b["brand"] not in BRANDS: raise HTTPException(400, "unknown brand")
    _, cid = q("INSERT INTO cameras(name,source_type,brand,ip,port,user,pw,ch,stream,url,modes,fps) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               (b["name"], b["source_type"], b.get("brand"), b.get("ip"), b.get("port", 554), b.get("user", "admin"), b.get("pw", ""),
                b.get("ch", 1), b.get("stream", "sub"), b.get("url"), json.dumps(b.get("modes", ["motion", "person", "tamper"])), b.get("fps", 3)))
    start_cam(q("SELECT * FROM cameras WHERE id=?", (cid,))[0][0]); return {"id": cid}

@app.delete("/api/cameras/{cid}")
def del_camera(cid: int, req: Request):
    auth(req); c = cams.pop(cid, None)
    if c: c.run_flag = False
    q("DELETE FROM cameras WHERE id=?", (cid,)); return {"ok": True}

@app.post("/api/test")
async def test(req: Request):
    auth(req); b = await req.json()
    try: src = source_of(b)
    except Exception as e: return {"ok": False, "msg": str(e)}
    cap = cv2.VideoCapture(src); ok, f = cap.read() if cap.isOpened() else (False, None); cap.release()
    return {"ok": bool(ok), "msg": f"Connected: {f.shape[1]}x{f.shape[0]}" if ok else "Could not read video (check IP, port, credentials, path)"}

@app.post("/api/webcam-demo")
def webcam_demo(req: Request):
    auth(req); _, cid = q("INSERT INTO cameras(name,source_type,ip,modes,fps) VALUES('Webcam Demo','webcam','0',?,3)", (json.dumps(["motion", "person", "tamper"]),))
    start_cam(q("SELECT * FROM cameras WHERE id=?", (cid,))[0][0]); return {"id": cid}

@app.get("/api/stream/{cid}")
def stream(cid: int, req: Request):
    auth(req)
    def gen():
        while cid in cams:
            j = cams[cid].jpg
            if j: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

@app.get("/api/events")
def events(req: Request, cam: int = 0, limit: int = 100):
    auth(req); return q("SELECT * FROM events " + ("WHERE cam_id=? " if cam else "WHERE ?=0 ") + "ORDER BY id DESC LIMIT ?", (cam, limit))[0]

@app.post("/api/events/{eid}/ack")
def ack(eid: int, req: Request): auth(req); q("UPDATE events SET ack=1 WHERE id=?", (eid,)); return {"ok": True}

@app.get("/api/snap/{fn}")
def snap(fn: str, req: Request): auth(req); return FileResponse(os.path.join(SNAPS, os.path.basename(fn)))

@app.get("/", response_class=HTMLResponse)
def index(): return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"), encoding="utf-8").read()

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8090
    print(f"Visio AI VMS prototype -> http://localhost:{port}  (admin / admin123)")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

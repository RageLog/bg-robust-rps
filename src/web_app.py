"""
web_app.py - RPS ARENA WEB (Final V2)
=====================================
"""
import os
import sys
import cv2
import time
import threading
import numpy as np
import tensorflow as tf
import urllib.request
from collections import deque
from flask import Flask, render_template, Response, jsonify, request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

app = Flask(__name__)

MODEL_FILE_EXTENSIONS = ('.keras', '.h5', '.hdf5')


def get_preprocessing_function(model_name):
    model_name = model_name.lower()
    if 'resnet' in model_name:
        return tf.keras.applications.resnet50.preprocess_input
    if 'vgg' in model_name:
        return tf.keras.applications.vgg16.preprocess_input
    if 'dense' in model_name:
        return tf.keras.applications.densenet.preprocess_input
    if 'mobilenet' in model_name:
        return tf.keras.applications.mobilenet_v3.preprocess_input
    return lambda x: x


def make_preprocess_input(preprocessor):
    @tf.keras.utils.register_keras_serializable(package="builtins", name="preprocess_input")
    def _model_preprocess_input(*args, **kwargs):
        return preprocessor(*args, **kwargs)

    return _model_preprocess_input


def _model_id(model_path):
    rel_path = os.path.relpath(model_path, config.MODELS_DIR)
    return rel_path.replace(os.sep, '/')


def _is_saved_model_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, 'saved_model.pb'))


def discover_models():
    models_dir = os.path.abspath(config.MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    found = {}
    for root, dirs, files in os.walk(models_dir):
        root = os.path.abspath(root)
        if _is_saved_model_dir(root):
            found[root] = {
                "id": _model_id(root),
                "name": os.path.basename(root),
                "type": "directory",
            }
            dirs[:] = []
            continue

        for filename in files:
            if filename.lower().endswith(MODEL_FILE_EXTENSIONS):
                model_path = os.path.abspath(os.path.join(root, filename))
                found[model_path] = {
                    "id": _model_id(model_path),
                    "name": filename,
                    "type": "file",
                }

    return sorted(found.values(), key=lambda item: item["id"].lower())


def resolve_model_id(model_id):
    if not model_id:
        return None

    normalized_id = model_id.replace('\\', '/')
    for model_info in discover_models():
        if model_info["id"] == normalized_id:
            return os.path.abspath(os.path.join(config.MODELS_DIR, *normalized_id.split('/')))
    return None


def get_default_model_path():
    models = discover_models()
    if not models:
        return None

    preferred = f"{config.RUN_NAME}_best.keras".lower()
    for model_info in models:
        if model_info["id"].lower() == preferred:
            return resolve_model_id(model_info["id"])
    return resolve_model_id(models[0]["id"])


def load_keras_model(model_path):
    preprocess_input = make_preprocess_input(
        get_preprocessing_function(os.path.basename(model_path))
    )

    load_kwargs = {
        "custom_objects": {"preprocess_input": preprocess_input},
        "safe_mode": False,
    }
    try:
        return tf.keras.models.load_model(model_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("safe_mode", None)
        return tf.keras.models.load_model(model_path, **load_kwargs)


# --- IP CAMERA READER ---
class IPCameraStream:
    def __init__(self, url):
        self.url = url
        self.stream = None
        self.bytes = b''
        self.frame = None
        self.stopped = False
        self.connect()

    def connect(self):
        try:
            self.stream = urllib.request.urlopen(self.url, timeout=5)
            print(f"    IP Camera Connected: {self.url}")
        except Exception as e:
            print(f"    IP Camera Error: {e}")
            self.stopped = True

    def read(self):
        if self.stopped or self.frame is None:
            return False, None
        return True, self.frame

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                chunk = self.stream.read(4096)
                if not chunk:
                    self.stopped = True
                    break
                self.bytes += chunk
                a = self.bytes.find(b'\xff\xd8')
                b = self.bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = self.bytes[a:b+2]
                    self.bytes = self.bytes[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None: self.frame = img
            except: time.sleep(0.1)

    def release(self):
        self.stopped = True
        try:
            if self.stream: self.stream.close()
        except: pass
    
    def isOpened(self): return not self.stopped

# --- GAME ENGINE ---
class WebGameEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.model = None
        self.selected_model_id = None
        self.selected_model_name = None
        self.model_error = None
        self.mode = "MENU"
        self.state = "WAITING"
        self.scores = {"p1": 0, "p2": 0}
        self.timer_start = 0
        self.history_p1 = deque(maxlen=3)
        self.history_p2 = deque(maxlen=3)
        self.moves = {"p1": None, "p2": None}
        self.winner = None
        self.vs1 = None
        self.vs2 = None
        self.src1_id = None
        self.src2_id = None
        self.load_model()

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = get_default_model_path()

        if model_path is None:
            message = f"No model files found in {config.MODELS_DIR}"
            print(f" Model Error: {message}")
            with self.model_lock:
                self.model = None
                self.selected_model_id = None
                self.selected_model_name = None
                self.model_error = message
            return False, message

        model_path = os.path.abspath(model_path)
        model_name = _model_id(model_path)
        print(f" Loading Model: {model_name}")
        try:
            model = load_keras_model(model_path)
            model.predict(np.zeros((1, *config.IMG_SIZE, 3), dtype=np.float32), verbose=0)
        except Exception as e:
            print(f" Model Error: {e}")
            with self.model_lock:
                self.model_error = str(e)
                if self.model is None:
                    self.selected_model_id = None
                    self.selected_model_name = None
            return False, str(e)

        with self.model_lock:
            self.model = model
            self.selected_model_id = model_name
            self.selected_model_name = os.path.basename(model_path)
            self.model_error = None

        self.history_p1.clear()
        self.history_p2.clear()
        print(f" Model Ready: {model_name}")
        return True, f"Model loaded: {model_name}"

    def select_model(self, model_id):
        model_path = resolve_model_id(model_id)
        if model_path is None:
            return False, f"Model not found: {model_id}"
        return self.load_model(model_path)

    def get_model_status(self):
        with self.model_lock:
            return {
                "selected_model": self.selected_model_id,
                "selected_model_name": self.selected_model_name,
                "model_ready": self.model is not None,
                "error": self.model_error,
            }

    def _open_source(self, source):
        if isinstance(source, str) and (source.startswith("http") or source.startswith("rtsp")):
            return IPCameraStream(source).start()
        else:
            try:
                if isinstance(source, str) and source.isdigit(): source = int(source)
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
            except: pass
        return None

    def find_best_source(self):
        print("🔄 Auto Camera Search...")
        cap = self._open_source(config.CAM_ID_MAIN)
        if cap: return cap, config.CAM_ID_MAIN
        for i in range(2):
            if str(i) == str(config.CAM_ID_MAIN): continue
            cap = self._open_source(i)
            if cap: return cap, i
        return None, None

    def init_camera(self, mode, user_cam1, user_cam2, use_default):
        with self.lock:
            # CAMERA 1
            if use_default and self.vs1 and self.vs1.isOpened():
                pass 
            else:
                target1 = config.CAM_ID_MAIN if use_default else user_cam1
                if str(self.src1_id) != str(target1) or self.vs1 is None or not self.vs1.isOpened():
                    if self.vs1: self.vs1.release()
                    self.vs1 = self._open_source(target1)
                    if self.vs1 is None and use_default:
                        self.vs1, target1 = self.find_best_source()
                    self.src1_id = target1

            # CAMERA 2 (PRO MODE) - FIXED
            if mode == "PRO":
                target2 = config.CAM_ID_SEC if use_default else user_cam2
                
                # If already open and sources match, do nothing
                if str(self.src2_id) == str(target2) and self.vs2 and self.vs2.isOpened():
                    pass
                else:
                    if self.vs2: self.vs2.release()
                    
                    # Try to open Camera 2
                    print(f"   📷 Trying Camera 2: {target2}")
                    new_vs2 = self._open_source(target2)
                    
                    if new_vs2 and new_vs2.isOpened():
                        self.vs2 = new_vs2
                        self.src2_id = target2
                    else:
                        print("   ⚠️ Camera 2 Could Not Open! (Fallback Cancelled)")
                        self.vs2 = None # NO fallback search, black screen will remain.
            else:
                if self.vs2: 
                    self.vs2.release()
                    self.vs2 = None

    def change_mode(self, new_mode, cam1=None, cam2=None, use_default=True):
        print(f"\n MODE: {new_mode}")
        self.mode = new_mode
        self.state = "WAITING"
        self.scores = {"p1":0, "p2":0}
        self.init_camera(new_mode, cam1, cam2, use_default)

    def get_prediction(self, roi, history):
        with self.model_lock:
            model = self.model
        if model is None: return 'none', 0.0
        try:
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, config.IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            preds = model.predict(img, verbose=0)[0]
            if history is not None:
                history.append(preds)
                avg = np.mean(history, axis=0)
            else: avg = preds
            idx = np.argmax(avg)
            conf = avg[idx]
            label = config.CLASS_NAMES[idx]
            if label == 'none' or conf < 0.60: return 'none', 0.0
            return label, conf
        except: return 'none', 0.0

    def determine_winner(self, p1, p2):
        if p1 == 'none' or p2 == 'none': return "INVALID"
        if p1 == p2: return "DRAW"
        wins = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        if wins.get(p1) == p2: return "P1"
        return "P2"

    def process_frame(self):
        # 1. Read Main Camera
        frame = None
        if self.vs1 and self.vs1.isOpened():
            ret, f = self.vs1.read()
            if ret: frame = f

        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            if self.mode != "MENU":
                cv2.putText(frame, "NO CAMERA CONNECTION", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


        if self.mode == "PRO":
            frame2 = None
            # Read only if Camera 2 is available and open
            if self.vs2 and self.vs2.isOpened():
                ret2, f2 = self.vs2.read()
                if ret2: frame2 = f2
            
            frame1 = cv2.resize(frame, (640, 480))
            
            if frame2 is not None:
                frame2 = cv2.resize(frame2, (640, 480))
            else:
                # If No Camera 2, Black Screen
                frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame2, "NO CAMERA 2", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            frame = np.hstack((frame1, frame2))
        
        # Resizing
        if frame.shape[0] != 720 or frame.shape[1] != 1280:
             frame = cv2.resize(frame, (1280, 720))

        h, w = frame.shape[:2]
        cx = w // 2

        # UI DRAWING
        if self.mode == "MENU":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w, h), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            cv2.putText(frame, "MAIN MENU", (cx-100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)
            cv2.putText(frame, "Please choose a game mode below", (cx-300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        elif self.mode in ["SOLO", "LOCAL", "PRO"]:
            p1_roi, p2_roi = None, None
            
            if self.mode == "SOLO":
                p1_roi = frame
            elif self.mode == "LOCAL":
                cv2.rectangle(frame, (50, 150), (450, 550), (255,255,0), 2)
                cv2.rectangle(frame, (w-450, 150), (w-50, 550), (255,0,255), 2)
                p1_roi = frame[150:550, 50:450]
                p2_roi = frame[150:550, w-450:w-50]
            elif self.mode == "PRO":
                p1_roi = frame[:, :640]
                p2_roi = frame[:, 640:]
                cv2.line(frame, (cx, 0), (cx, h), (255,255,255), 2)

            # GAME FLOW
            if self.state == "WAITING":
                msg = "START: [SPACE]"
                if int(time.time()*2)%2==0:
                    cv2.putText(frame, msg, (cx-120, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                # Debug (Solo)
                if self.mode == "SOLO" and int(time.time()*5)%3==0:
                    l, _ = self.get_prediction(p1_roi, deque(maxlen=1))
                    if l != 'none': cv2.putText(frame, f"Eye: {l}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

            elif self.state == "COUNTDOWN":
                rem = 3 - int(time.time() - self.timer_start)
                if rem > 0:
                    cv2.putText(frame, str(rem), (cx-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 8, (255,0,0), 15)
                else:
                    self.state = "SHOW"; self.timer_start = time.time()

            elif self.state == "SHOW":
                cv2.putText(frame, "SHOW!", (cx-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
                if time.time() - self.timer_start > 0.5:
                    if self.mode == "SOLO":
                        m1, _ = self.get_prediction(p1_roi, self.history_p1)
                        if m1 != 'none':
                            m2 = np.random.choice(['rock', 'paper', 'scissors'])
                            self.moves = {'p1': m1, 'p2': m2}
                            res = self.determine_winner(m1, m2)
                            self.winner = "P1" if res=="P1" else "P2" if res=="P2" else "DRAW"
                            if self.winner == "P1": self.scores['p1'] += 1
                            elif self.winner == "P2": self.scores['p2'] += 1
                            self.state = "RESULT"; self.timer_start = time.time()
                    else:
                        m1, _ = self.get_prediction(p1_roi, self.history_p1)
                        m2, _ = self.get_prediction(p2_roi, self.history_p2)
                        self.moves = {'p1': m1, 'p2': m2}
                        self.winner = self.determine_winner(m1, m2)
                        if self.winner == "P1": self.scores['p1'] += 1
                        elif self.winner == "P2": self.scores['p2'] += 1
                        self.state = "RESULT"; self.timer_start = time.time()

            elif self.state == "RESULT":
                p1_txt = str(self.moves['p1']).upper()
                p2_txt = str(self.moves['p2']).upper()
                
                if self.mode == "SOLO":
                    cv2.putText(frame, f"YOU: {p1_txt}", (100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
                    cv2.putText(frame, f"CPU: {p2_txt}", (w-400, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                else:
                    cv2.putText(frame, p1_txt, (100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
                    cv2.putText(frame, p2_txt, (w-400, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,255), 3)

                msg = "YOU WIN" if self.winner=="P1" else "YOU LOSE" if (self.winner=="P2" and self.mode=="SOLO") else "P2 WINS" if self.winner=="P2" else "DRAW"
                col = (0,255,0) if self.winner=="P1" else (0,0,255) if (self.winner=="P2" and self.mode=="SOLO") else (255,0,255) if self.winner=="P2" else (255,255,255)
                
                cv2.putText(frame, msg, (cx-150, h-100), cv2.FONT_HERSHEY_SIMPLEX, 2, col, 4)

                if time.time() - self.timer_start > 3.0: self.state = "WAITING"

            cv2.putText(frame, f"P1: {self.scores['p1']} | P2: {self.scores['p2']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return frame

game = WebGameEngine()

def generate_frames():
    while True:
        try:
            frame = game.process_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03) 
        except Exception: time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/models')
def models_route():
    status = game.get_model_status()
    return jsonify(models=discover_models(), **status)

@app.route('/model/select', methods=['POST'])
def select_model_route():
    payload = request.get_json(silent=True) or {}
    model_id = payload.get('model')
    ok, message = game.select_model(model_id)
    status = game.get_model_status()
    response = jsonify(status=message, models=discover_models(), **status)
    return response, 200 if ok else 400

@app.route('/command/<cmd>')
def command(cmd):
    cam1 = request.args.get('cam1')
    cam2 = request.args.get('cam2')
    use_default = request.args.get('use_default') == 'true'
    print(f" COMMAND: {cmd} | Default: {use_default}")
    
    if cmd == "mode_1":
        game.change_mode("SOLO", cam1, cam2, use_default)
        return jsonify(status="SOLO Mode")
    elif cmd == "mode_2":
        game.change_mode("LOCAL", cam1, cam2, use_default)
        return jsonify(status="LOCAL Mode")
    elif cmd == "mode_3":
        game.change_mode("PRO", cam1, cam2, use_default)
        return jsonify(status="PRO Mode")
    elif cmd == "quit":
        game.mode = "MENU"
        return jsonify(status="Main Menu")
    elif cmd == "space":
        if game.state == "WAITING" and game.mode != "MENU":
            game.history_p1.clear()
            game.history_p2.clear()
            game.state = "COUNTDOWN"
            game.timer_start = time.time()
            return jsonify(status="Starting!")
        
    return jsonify(status="Ready")

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)

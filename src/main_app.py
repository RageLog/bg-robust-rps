"""
main_app.py - RPS ARENA: ULTIMATE
===========================================================
"""
import os
import sys
import cv2
import time
import threading
import numpy as np
import tensorflow as tf
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config



# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)

class VideoStream:
    """Safe Threaded Video Capture"""
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stopped = True
                break
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                self.stopped = True
                break
            self.grabbed = grabbed
            self.frame = frame
            time.sleep(0.002)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.stream.release()

class GameEngine:
    def __init__(self):
        print("\n INITIALIZING RPS ARENA...")
        self.load_model()
        
    def load_model(self):
        print(" Loading AI...")
        model_path = os.path.join(config.MODELS_DIR, f"{config.RUN_NAME}_best.keras")
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Warmup
            dummy = np.zeros((1, *config.IMG_SIZE, 3), dtype=np.float32)
            self.model.predict(dummy, verbose=0)
            print(" Model and GPU Ready!")
        except Exception as e:
            print(f" Critical Error: Model could not be loaded! ({e})")
            sys.exit(1)

    def predict(self, img_roi, history=None):
        try:
            img = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, config.IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            preds = self.model.predict(img, verbose=0)[0]
            
            if history is not None:
                history.append(preds)
                avg_preds = np.mean(history, axis=0)
            else:
                avg_preds = preds
                
            idx = np.argmax(avg_preds)
            conf = avg_preds[idx]
            label = config.CLASS_NAMES[idx]
            
            if label == 'none' or conf < 0.60:
                return 'none', 0.0
            return label, conf
        except:
            return 'none', 0.0

    def determine_winner(self, p1, p2):
        if p1 == 'none' or p2 == 'none': return "INVALID"
        if p1 == p2: return "DRAW"
        wins = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        if wins.get(p1) == p2: return "P1"
        return "P2"

    def cleanup_before_menu(self, vs_list):
        """Safely closes everything before returning to menu."""
        print(" Returning to menu...")
        for vs in vs_list:
            if vs: vs.stop()
        
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # Give OS a breather (Prevents crash)
        time.sleep(0.5) 

    # --- GAME MODES ---

    def run_solo_challenge(self):
        print("Starting: SOLO CHALLENGE")
        try:
            vs = VideoStream(src=config.CAM_ID_MAIN).start()
            time.sleep(1.0)
            
            scores = {"player": 0, "cpu": 0}
            history = deque(maxlen=3)
            state = "WAITING"
            timer_start = 0
            player_move, cpu_move, winner = None, None, None
            
            while True:
                frame = vs.read()
                if frame is None: break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                center_x = w // 2
                
                # Prediction
                pred_label = 'none'
                if state == "WAITING":
                    if int(time.time()*5)%3 == 0:
                        pred_label, _ = self.predict(frame, history)
                else:
                    pred_label, _ = self.predict(frame, history)

                # UI
                cv2.rectangle(frame, (0,0), (w, 80), (20,20,20), -1)
                cv2.putText(frame, f"YOU: {scores['player']}", (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
                cv2.putText(frame, f"CPU: {scores['cpu']}", (w-250, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
                
                if state == "WAITING":
                    cv2.putText(frame, "START: [SPACE] | MENU: [Q]", (center_x-250, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                    if pred_label != 'none':
                        cv2.putText(frame, f"Detected: {pred_label.upper()}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1)
                
                elif state == "COUNTDOWN":
                    rem = 3 - int(time.time() - timer_start)
                    if rem > 0:
                        cv2.putText(frame, str(rem), (center_x-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 8, BLUE, 15)
                    else:
                        state = "SHOW"
                        timer_start = time.time()
                
                elif state == "SHOW":
                    cv2.putText(frame, "SHOW!", (center_x-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, GREEN, 5)
                    if time.time() - timer_start > 0.5:
                        if pred_label != 'none':
                            player_move = pred_label
                            cpu_move = np.random.choice(['rock', 'paper', 'scissors'])
                            res = self.determine_winner(player_move, cpu_move)
                            winner = "PLAYER" if res == "P1" else "CPU" if res == "P2" else "DRAW"
                            
                            if winner == "PLAYER": scores['player'] += 1
                            elif winner == "CPU": scores['cpu'] += 1
                            
                            state = "RESULT"
                            timer_start = time.time()
                        elif time.time() - timer_start > 2.0:
                            state = "WAITING"

                elif state == "RESULT":
                    cv2.putText(frame, f"YOU: {player_move.upper()}", (100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
                    cv2.putText(frame, f"CPU: {cpu_move.upper()}", (w-400, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
                    
                    msg = "YOU WIN!" if winner == "PLAYER" else "YOU LOSE!" if winner == "CPU" else "DRAW"
                    col = GREEN if winner == "PLAYER" else RED if winner == "CPU" else WHITE
                    cv2.putText(frame, msg, (center_x-150, h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, col, 4)
                    
                    if time.time() - timer_start > 3.0:
                        state = "WAITING"

                cv2.imshow("SOLO CHALLENGE", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'): 
                    break
                elif key == ord(' ') and state == "WAITING":
                    history.clear()
                    state = "COUNTDOWN"
                    timer_start = time.time()
            
            self.cleanup_before_menu([vs])
            
        except Exception as e:
            print(f"Error occurred: {e}")
            self.cleanup_before_menu([])

    def run_local_duel(self):
        print("Starting: LOCAL DUEL")
        try:
            vs = VideoStream(src=config.CAM_ID_MAIN).start()
            time.sleep(1.0)
            
            scores = {"p1": 0, "p2": 0}
            hist1, hist2 = deque(maxlen=3), deque(maxlen=3)
            state = "WAITING"
            timer_start = 0
            moves = {"p1": None, "p2": None}
            winner = None
            
            while True:
                frame = vs.read()
                if frame is None: break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                cx = w // 2
                
                # ROI
                p1_box = (50, 150, 450, 550)
                p2_box = (w-450, 150, w-50, 550)
                
                cv2.rectangle(frame, (p1_box[0], p1_box[1]), (p1_box[2], p1_box[3]), CYAN, 2)
                cv2.rectangle(frame, (p2_box[0], p2_box[1]), (p2_box[2], p2_box[3]), MAGENTA, 2)
                cv2.putText(frame, "P1", (p1_box[0], p1_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 2)
                cv2.putText(frame, "P2", (p2_box[0], p2_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2)
                
                cv2.putText(frame, f"{scores['p1']} - {scores['p2']}", (cx-60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)

                if state == "WAITING":
                    cv2.putText(frame, "BASLA: [BOSLUK] | MENU: [Q]", (cx-250, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                    
                elif state == "COUNTDOWN":
                    rem = 3 - int(time.time() - timer_start)
                    if rem > 0:
                        cv2.putText(frame, str(rem), (cx-30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 6, BLUE, 10)
                    else:
                        state = "SHOW"
                        timer_start = time.time()
                        
                elif state == "SHOW":
                    cv2.putText(frame, "GOSTER!", (cx-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, GREEN, 5)
                    if time.time() - timer_start > 0.5:
                        roi1 = frame[p1_box[1]:p1_box[3], p1_box[0]:p1_box[2]]
                        roi2 = frame[p2_box[1]:p2_box[3], p2_box[0]:p2_box[2]]
                        
                        m1, _ = self.predict(roi1, hist1)
                        m2, _ = self.predict(roi2, hist2)
                        moves['p1'], moves['p2'] = m1, m2
                        
                        winner = self.determine_winner(m1, m2)
                        if winner == "P1": scores['p1'] += 1
                        elif winner == "P2": scores['p2'] += 1
                        
                        state = "RESULT"
                        timer_start = time.time()
                        
                elif state == "RESULT":
                    cv2.putText(frame, moves['p1'].upper(), (p1_box[0]+50, p1_box[3]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 2)
                    cv2.putText(frame, moves['p2'].upper(), (p2_box[0]+50, p2_box[3]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2)
                    
                    msg = "P1 WINS" if winner == "P1" else "P2 WINS" if winner == "P2" else "DRAW"
                    col = CYAN if winner == "P1" else MAGENTA if winner == "P2" else WHITE
                    cv2.putText(frame, msg, (cx-150, h//2+80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)
                    
                    if time.time() - timer_start > 4.0: state = "WAITING"

                cv2.imshow("LOCAL DUEL", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord(' ') and state == "WAITING":
                    hist1.clear(); hist2.clear()
                    state = "COUNTDOWN"; timer_start = time.time()
            
            self.cleanup_before_menu([vs])
        except Exception as e:
            print(f"Error: {e}")
            self.cleanup_before_menu([])

    def run_pro_arena(self):
        print("Starting: PRO ARENA")
        try:
            vs1 = VideoStream(src=config.CAM_ID_MAIN).start()
            vs2 = VideoStream(src=config.CAM_ID_SEC).start()
            time.sleep(2.0)
            
            if vs1.stopped or vs2.stopped:
                print(" Cameras could not be opened.")
                self.cleanup_before_menu([vs1, vs2])
                return

            scores = {"p1": 0, "p2": 0}
            hist1, hist2 = deque(maxlen=3), deque(maxlen=3)
            state = "WAITING"
            timer_start = 0
            moves = {"p1": None, "p2": None}
            winner = None
            
            while True:
                f1, f2 = vs1.read(), vs2.read()
                if f1 is None or f2 is None: continue
                
                f1 = cv2.resize(f1, (640, 480)); f1 = cv2.flip(f1, 1)
                f2 = cv2.resize(f2, (640, 480)); f2 = cv2.flip(f2, 1)
                frame = np.hstack((f1, f2))
                h, w = frame.shape[:2]
                cx = w // 2
                
                cv2.line(frame, (cx, 0), (cx, h), WHITE, 2)
                cv2.putText(frame, "P1", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 2)
                cv2.putText(frame, "P2", (660, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2)
                cv2.putText(frame, f"{scores['p1']} - {scores['p2']}", (cx-60, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)

                if state == "WAITING":
                    cv2.putText(frame, "BASLA: [BOSLUK] | MENU: [Q]", (cx-250, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                    
                elif state == "COUNTDOWN":
                    rem = 3 - int(time.time() - timer_start)
                    if rem > 0:
                        cv2.putText(frame, str(rem), (cx-30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 6, BLUE, 10)
                    else:
                        state = "SHOW"; timer_start = time.time()
                        
                elif state == "SHOW":
                    cv2.putText(frame, "GOSTER!", (cx-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, GREEN, 5)
                    if time.time() - timer_start > 0.5:
                        m1, _ = self.predict(f1, hist1)
                        m2, _ = self.predict(f2, hist2)
                        moves['p1'], moves['p2'] = m1, m2
                        
                        winner = self.determine_winner(m1, m2)
                        if winner == "P1": scores['p1'] += 1
                        elif winner == "P2": scores['p2'] += 1
                        state = "RESULT"; timer_start = time.time()
                
                elif state == "RESULT":
                    cv2.putText(frame, moves['p1'].upper(), (200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, CYAN, 3)
                    cv2.putText(frame, moves['p2'].upper(), (840, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, MAGENTA, 3)
                    
                    msg = "P1 WINS" if winner == "P1" else "P2 WINS" if winner == "P2" else "DRAW"
                    col = CYAN if winner == "P1" else MAGENTA if winner == "P2" else WHITE
                    cv2.putText(frame, msg, (cx-150, h-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, col, 3)
                    
                    if time.time() - timer_start > 4.0: state = "WAITING"

                cv2.imshow("PRO ARENA", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord(' ') and state == "WAITING":
                    hist1.clear(); hist2.clear()
                    state = "COUNTDOWN"; timer_start = time.time()
            
            self.cleanup_before_menu([vs1, vs2])
        except Exception as e:
            print(f"Error: {e}")
            self.cleanup_before_menu([])

    def main_menu(self):
        while True:
            menu = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(menu, "RPS ARENA: ULTIMATE", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, YELLOW, 3)
            
            opts = ["[1] SOLO CHALLENGE", "[2] LOCAL DUEL", "[3] PRO ARENA", "[Q] EXIT"]
            for i, opt in enumerate(opts):
                col = GREEN if i < 3 else RED
                cv2.putText(menu, opt, (150, 250 + i*70), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
            
            cv2.imshow("MAIN MENU", menu)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                cv2.destroyWindow("MAIN MENU")
                self.run_solo_challenge()
            elif key == ord('2'):
                cv2.destroyWindow("MAIN MENU")
                self.run_local_duel()
            elif key == ord('3'):
                cv2.destroyWindow("MAIN MENU")
                self.run_pro_arena()
            elif key == ord('q'):
                print("👋 Exiting...")
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GameEngine()
    app.main_menu()
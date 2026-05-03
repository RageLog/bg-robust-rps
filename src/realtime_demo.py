"""
realtime_demo.py - Threaded Capture & Warmup
============================================
Optimized for DroidCam and IP Cameras.

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


class VideoStream:
    """
    Reads the image from the camera in the background (separate thread).
    This prevents the camera connection from dropping while the model predicts.
    """
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        
        # Camera settings (Only works for USB, minimal effect for HTTP)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduces latency

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
        # Connection check upon startup
        if not self.grabbed:
            print(f" Camera could not be opened: {src}")
            self.stopped = True

    def start(self):
        if self.stopped: return self
        # Start thread
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Continuously read the latest frame and clear the buffer
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
            time.sleep(0.005) # Tiny wait to prevent high CPU usage

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    print(f"\n ROCK-PAPER-SCISSORS (Threaded Demo)")
    print("-" * 50)

    # 1. Load Model
    model_path = os.path.join(config.MODELS_DIR, f"{config.RUN_NAME}_best.keras")
    if not os.path.exists(model_path):
        print(" Model not found! Run train.py first.")
        return

    print(" Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f" Error: {e}")
        return

    # 2. MODEL WARMUP - CRITICAL STEP
    # We make a prediction with dummy data before opening the camera.
    # This allows XLA compilation to finish here, preventing camera freeze.
    print(" Model XLA compilation...")
    dummy_input = np.zeros((1, *config.IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    print(" Model is ready and warmed up!")

    # 3. Start Camera (Threaded)
    print(f" Connecting: {config.CAM_ID_MAIN}")
    vs = VideoStream(src=config.CAM_ID_MAIN).start()
    
    if vs.stopped:
        print(" Camera connection failed.")
        return

    time.sleep(1.0) # Wait for camera light adjustment

    # Variables
    history = deque(maxlen=5)
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    print("\n Live Stream Started! ('q': Exit)")

    while not vs.stopped:
        # 1. Get frame (Non-blocking, gives the latest frame)
        frame = vs.read()
        if frame is None: break
        
        # Frames to process and display
        display_frame = frame.copy()
        
        # Mirroring (Only for USB, DroidCam sometimes flips)
        # display_frame = cv2.flip(display_frame, 1)

        # 2. Preprocess
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, config.IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        except Exception as e:
            continue

        # 3. Prediction
        preds = model.predict(img, verbose=0)[0]
        history.append(preds)
        
        # 4. Process Result
        avg_preds = np.mean(history, axis=0)
        class_idx = np.argmax(avg_preds)
        confidence = avg_preds[class_idx]
        class_name = config.CLASS_NAMES[class_idx].upper()

        # Calculate FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # 5. Visualization
        h, w = display_frame.shape[:2]
        
        # Color and Text
        color = (0, 255, 0)
        if class_name == 'NONE' or confidence < 0.6:
            color = (128, 128, 128)
            if class_name != 'NONE': class_name = "..."
        
        # Header Bar
        cv2.rectangle(display_frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(display_frame, f"{class_name}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.putText(display_frame, f"%{confidence*100:.0f}", (w-180, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # FPS
        cv2.putText(display_frame, f"FPS: {fps}", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show Image
        # Scale down to fit the screen (if DroidCam sends 1080p)
        if w > 1280:
            display_frame = cv2.resize(display_frame, (1280, 720))
            
        cv2.imshow("RPS AI", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
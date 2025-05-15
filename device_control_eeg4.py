import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from scipy import signal
from scipy.spatial import distance as dist
import mediapipe as mp
import time
import threading
import serial
import pandas as pd
import pickle
import pyautogui
from collections import deque
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks, landmarks, image_shape):
    h, w = image_shape[:2]
    eye_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_landmarks]
    A = dist.euclidean(eye_coords[1], eye_coords[5])
    B = dist.euclidean(eye_coords[2], eye_coords[4])
    C = dist.euclidean(eye_coords[0], eye_coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EEG processing functions
def setup_filters(sampling_rate):
    try:
        b_notch, a_notch = signal.iirnotch(50.0 / (0.5 * sampling_rate), 30.0)
        b_bandpass, a_bandpass = signal.butter(4, [0.5 / (0.5 * sampling_rate), 30.0 / (0.5 * sampling_rate)], 'band')
        print("Filters set up successfully")
        return b_notch, a_notch, b_bandpass, a_bandpass
    except Exception as e:
        print(f"Error setting up filters: {e}")
        raise

def process_eeg_data(data, b_notch, a_notch, b_bandpass, a_bandpass):
    try:
        data = signal.filtfilt(b_notch, a_notch, data)
        data = signal.filtfilt(b_bandpass, a_bandpass, data)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Processed EEG data contains NaN or Inf")
        print("EEG data processed successfully")
        return data
    except Exception as e:
        print(f"Error processing EEG data: {e}")
        raise

def calculate_psd_features(segment, sampling_rate):
    try:
        f, psd_values = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
        bands = {'alpha': (8, 13), 'beta': (14, 30), 'theta': (4, 7), 'delta': (0.5, 3)}
        features = {}
        for band, (low, high) in bands.items():
            idx = np.where((f >= low) & (f <= high))
            features[f'E_{band}'] = np.sum(psd_values[idx])
        features['alpha_beta_ratio'] = features['E_alpha'] / features['E_beta'] if features['E_beta'] > 0 else 0
        print("PSD features calculated")
        return features
    except Exception as e:
        print(f"Error calculating PSD features: {e}")
        raise

def calculate_additional_features(segment, sampling_rate):
    try:
        f, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
        peak_frequency = f[np.argmax(psd)]
        spectral_centroid = np.sum(f * psd) / np.sum(psd)
        log_f = np.log(f[1:])
        log_psd = np.log(psd[1:])
        spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
        print("Additional features calculated")
        return {'peak_frequency': peak_frequency, 'spectral_centroid': spectral_centroid, 'spectral_slope': spectral_slope}
    except Exception as e:
        print(f"Error calculating additional features: {e}")
        raise

def load_model_and_scaler():
    try:
        with open('model1.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('scaler1.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully")
        return clf, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        raise

# GUI and Blink Detection with EEG Integration
class DeviceControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Device Control with Long Eye Blinks and EEG")
        self.root.geometry("400x300")

        # Device options
        self.devices = ["Fan", "Light", "Water", "AC", "TV"]
        self.current_index = 0

        # GUI elements
        self.label = tk.Label(root, text="Select Device:", font=("Arial", 14))
        self.label.pack(pady=20)

        self.device_label = tk.Label(root, text=self.devices[self.current_index], font=("Arial", 16, "bold"))
        self.device_label.pack(pady=20)

        self.status_label = tk.Label(root, text="Long Blink: Move Right | EEG: No blink > 2s", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.toggle_button = tk.Button(root, text="Toggle Device", command=self.toggle_device, font=("Arial", 12))
        self.toggle_button.pack(pady=10)

        self.device_states = {device: False for device in self.devices}
        self.state_label = tk.Label(root, text=self.get_state_text(), font=("Arial", 10))
        self.state_label.pack(pady=10)

        # Blink detection variables
        self.EAR_THRESHOLD = 0.15
        self.CONSEC_FRAMES = 5
        self.MIN_BLINK_DURATION = 0.3
        self.NO_BLINK_THRESHOLD = 2.0
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.blink_start_time = None
        self.last_open_eyes_time = time.time()
        self.is_running = True

        # MediaPipe eye landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # EEG setup
        try:
            self.ser = serial.Serial('COM16', 115200, timeout=2)
            self.clf, self.scaler = load_model_and_scaler()
            self.b_notch, self.a_notch, self.b_bandpass, self.a_bandpass = setup_filters(512)
            self.eeg_buffer = deque(maxlen=512)
            if self.ser:
                print("ser")
            print("EEG setup completed")
        except Exception as e:
            print(f"EEG Setup Error: {e}")
            self.ser = None

        # Start webcam and detection in a separate thread
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
        self.thread = threading.Thread(target=self.detect_blinks_and_eeg)
        self.thread.daemon = True
        self.thread.start()

    def get_state_text(self):
        return "\n".join(f"{device}: {'ON' if state else 'OFF'}" for device, state in self.device_states.items())

    def toggle_device(self):
        device = self.devices[self.current_index]
        self.device_states[device] = not self.device_states[device]
        self.state_label.config(text=self.get_state_text())
        messagebox.showinfo("Device Status", f"{device} is now {'ON' if self.device_states[device] else 'OFF'}")

    def update_selection(self, direction):
        if direction == "right":
            self.current_index = (self.current_index + 1) % len(self.devices)
            self.status_label.config(text="Long Blink: Moved Right | EEG: No blink > 2s")
            print("Action: Moved Right")
        self.device_label.config(text=self.devices[self.current_index])

    def process_eeg(self):
        print(f"------------------------------------------Processing EEG, buffer size: {len(self.eeg_buffer)}")
        # if self.ser:
        #     try:
        #         raw_data = self.ser.readline().decode('latin-1').strip()
        #         print(f"Raw EEG data: {raw_data}")
        #         if raw_data:
        #             eeg_value = float(raw_data)
        #             self.eeg_buffer.append(eeg_value)
        #             print(f"EEG value appended, buffer size: {len(self.eeg_buffer)}")
        #         else:
        #             print("Empty EEG data received")
        #     except Exception as e:
        #         print(f"EEG Read Error: {e}")
        if len(self.eeg_buffer) == 512:
            try:
                buffer_array = np.array(self.eeg_buffer)
                print("Buffer converted to array")
                processed_data = process_eeg_data(buffer_array, self.b_notch, self.a_notch, self.b_bandpass, self.a_bandpass)
                psd_features = calculate_psd_features(processed_data, 512)
                additional_features = calculate_additional_features(processed_data, 512)
                features = {**psd_features, **additional_features}
                print(f"Features extracted: {features}")
                df = pd.DataFrame([features])
                X_scaled = self.scaler.transform(df)
                print("Features scaled")
                prediction = self.clf.predict(X_scaled)[0]
                print(f"EEG Predicted Class: {prediction}")
                self.eeg_buffer.clear()
                if prediction == 0:
                    #pyautogui.keyDown('r')
                    print('Relaxed')
                    time.sleep(1)
                    #pyautogui.keyUp('r')
                elif prediction == 1:
                    #pyautogui.keyDown('f')
                    self.toggle_device()
                    print('Focused')
                    time.sleep(1)
                    #pyautogui.keyUp('f')
            except Exception as e:
                print(f"Error in EEG processing: {e}")
        else:
            print("Buffer not full, skipping EEG processing")

    def detect_blinks_and_eeg(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            current_time = time.time()

            # EEG data collection
            if self.ser:
                try:
                    raw_data = self.ser.readline().decode('latin-1').strip()
                    print(f"Raw EEG data: {raw_data}")
                    if raw_data:
                        eeg_value = float(raw_data)
                        self.eeg_buffer.append(eeg_value)
                        print(f"EEG value appended, buffer size: {len(self.eeg_buffer)}")
                    else:
                        print("Empty EEG data received")
                except Exception as e:
                    print(f"EEG Read Error: {e}")

            if results.multi_face_landmarks:
                print("Face detected")
                for face_landmarks in results.multi_face_landmarks:
                    left_ear = eye_aspect_ratio(self.LEFT_EYE, face_landmarks.landmark, frame.shape)
                    right_ear = eye_aspect_ratio(self.RIGHT_EYE, face_landmarks.landmark, frame.shape)
                    ear = (left_ear + right_ear) / 2.0
                    print(f"EAR: {ear:.3f}, Blink Counter: {self.blink_counter}")

                    # Long blink detection logic
                    if ear < self.EAR_THRESHOLD:
                        if self.blink_counter == 0:
                            self.last_open_eyes_time = current_time
                            self.blink_start_time = current_time
                            print("Blink started")
                        self.blink_counter += 1
                    else:
                        #self.last_open_eyes_time = current_time
                        if self.blink_counter >= self.CONSEC_FRAMES:
                            blink_duration = current_time - self.blink_start_time
                            if blink_duration >= self.MIN_BLINK_DURATION:
                                print("Long blink detected")
                                self.root.after(0, self.update_selection, "right")
                                # self.eeg_buffer.clear()
                                self.last_blink_time = current_time
                        self.blink_counter = 0
                        self.blink_start_time = None

                    # Draw eye landmarks and text for debugging
                    for idx in self.LEFT_EYE + self.RIGHT_EYE:
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Blink: {'Yes' if ear < self.EAR_THRESHOLD else 'No'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Check for no blink > 2 seconds
                    # print("------------------------time------------------------")
                    # print(current_time - self.last_open_eyes_time)
                    if current_time - self.last_open_eyes_time >= self.NO_BLINK_THRESHOLD and self.ser:
                        print("No blink for 2 seconds, processing EEG")
                        self.process_eeg()
            else:
                print("No face detected")
                cv2.putText(frame, "No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def on_closing(self):
        self.is_running = False
        self.cap.release()
        if self.ser:
            self.ser.close()
        cv2.destroyAllWindows()
        face_mesh.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DeviceControlApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
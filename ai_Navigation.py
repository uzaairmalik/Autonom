from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import deque
import serial
import threading
import queue

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "best.pt"
CONF_THRESH = 0.3
MIN_AREA_RATIO = 0.008
MAX_AREA_RATIO = 0.6
FRAME_CONFIRM = 4

ZONES = {
    "left": (0.0, 0.33),
    "center": (0.33, 0.66),
    "right": (0.66, 1.0)
}

CLASS_WEIGHTS = {
    "person": 1.0,
    "car": 0.9,
    "truck": 0.8,
    "bus": 0.8,
    "motorcycle": 0.7,
    "bicycle": 0.6,
    "chair": 0.2,
    "table": 0.2
}

RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0
}

# =========================
# HARDWARE COMMUNICATION CLASS
# =========================
class DroneHardware:
    def __init__(self, port='COM3', baud=9600):
        self.port = port
        self.baud = baud
        self.arduino = None
        self.connected = False
        self.command_queue = queue.Queue()
        
        # Motor control parameters (in microseconds)
        self.throttle = 1300  # Hover throttle
        self.pitch = 1500     # Neutral
        self.roll = 1500      # Neutral
        self.yaw = 1500       # Neutral
        
        # Initialize connection
        self.connect()
        
        # Start command sending thread
        self.running = True
        self.thread = threading.Thread(target=self._command_sender)
        self.thread.start()
    
    def connect(self):
        """Establish connection with Arduino"""
        try:
            self.arduino = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            self.connected = True
            print(f"[HARDWARE] Connected to Arduino on {self.port}")
        except Exception as e:
            print(f"[HARDWARE] Connection failed: {e}")
            print("[HARDWARE] Running in SIMULATION mode")
            self.connected = False
    
    def _command_sender(self):
        """Background thread to send commands"""
        while self.running:
            try:
                if self.connected:
                    # Send current motor values every 50ms
                    cmd = f"T{self.throttle}P{self.pitch}R{self.roll}Y{self.yaw}\n"
                    self.arduino.write(cmd.encode())
                
                # Process queued commands
                if not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    self._process_command(cmd)
                
                time.sleep(0.05)  # 20Hz update rate
                
            except Exception as e:
                print(f"[HARDWARE ERROR] {e}")
                time.sleep(0.1)
    
    def _process_command(self, command):
        """Process AI commands and convert to motor controls"""
        # Map AI commands to motor controls
        if command == "F":  # Move forward
            self.pitch = 1600  # Lean forward
            self.roll = 1500   # Neutral roll
            print("[HARDWARE] Moving FORWARD")
            
        elif command == "L":  # Turn left
            self.roll = 1400   # Lean left
            self.yaw = 1400    # Rotate left
            print("[HARDWARE] Turning LEFT")
            
        elif command == "R":  # Turn right
            self.roll = 1600   # Lean right
            self.yaw = 1600    # Rotate right
            print("[HARDWARE] Turning RIGHT")
            
        elif command == "U":  # Ascend
            self.throttle = min(self.throttle + 50, 1800)
            print("[HARDWARE] ASCENDING")
            
        elif command == "S":  # Slow down
            self.pitch = 1450  # Slight forward
            print("[HARDWARE] SLOWING")
            
        elif command == "H":  # Hover
            self.pitch = 1500
            self.roll = 1500
            self.yaw = 1500
            print("[HARDWARE] HOVERING")
            
        elif command == "X":  # Emergency stop
            self.throttle = 1000
            self.pitch = 1500
            self.roll = 1500
            self.yaw = 1500
            print("[HARDWARE] EMERGENCY STOP")
    
    def send_command(self, command):
        """Queue a command for execution"""
        if command in ["F", "L", "R", "U", "S", "H", "X"]:
            self.command_queue.put(command)
            return True
        return False
    
    def get_status(self):
        """Get current hardware status"""
        return {
            "connected": self.connected,
            "throttle": self.throttle,
            "pitch": self.pitch,
            "roll": self.roll,
            "yaw": self.yaw
        }
    
    def shutdown(self):
        """Safely shutdown hardware"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        
        # Send emergency stop before closing
        if self.connected:
            try:
                self.arduino.write(b"X\n")
                time.sleep(0.1)
                self.arduino.close()
            except:
                pass
        
        print("[HARDWARE] Shutdown complete")

# =========================
# ENHANCED DRONE CONTROLLER
# =========================
class AutonomousDrone:
    def __init__(self):
        # AI Vision
        print("[INFO] Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)
        
        # Hardware Interface
        self.hardware = DroneHardware(port='COM3')  # Change to your port
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Camera not accessible")
            exit()
        
        # State Management
        self.state = {
            "confirmed_frames": 0,
            "last_command": "F",
            "command_history": deque(maxlen=8),
            "obstacle_grid": np.zeros((3, 3)),
            "stuck_counter": 0,
            "fps": 0,
            "frame_count": 0,
            "last_risk_time": time.time()
        }
        
        # Safety parameters
        self.safe_throttle = 1300  # Just above hover
        print("[INFO] Autonomous Drone System Ready")
    
    def get_zone(self, x, w):
        """Convert x position to zone"""
        nx = x / w
        for name, (a, b) in ZONES.items():
            if a <= nx <= b:
                return name
        return "center"
    
    def calculate_risk(self, cls, area_ratio, zone, conf):
        """Calculate risk score for detection"""
        if cls not in CLASS_WEIGHTS:
            return 0.0
        size_factor = np.log1p(area_ratio * 100)
        pos_factor = 1.5 if zone == "center" else 1.0
        return min(CLASS_WEIGHTS[cls] * size_factor * pos_factor * conf, 3.0)
    
    def update_obstacle_grid(self, detections, w, h):
        """Update spatial obstacle grid"""
        self.state["obstacle_grid"] = np.zeros((3, 3))
        for d in detections:
            gx = min(int((d["center_x"] / w) * 3), 2)
            gy = min(int((d["center_y"] / h) * 3), 2)
            self.state["obstacle_grid"][gy, gx] += d["risk"]
    
    def choose_direction(self):
        """Choose safest direction based on obstacle grid"""
        left = self.state["obstacle_grid"][:, 0].sum()
        center = self.state["obstacle_grid"][:, 1].sum()
        right = self.state["obstacle_grid"][:, 2].sum()
        
        print(f"[NAV] Obstacle densities - L:{left:.2f} C:{center:.2f} R:{right:.2f}")
        
        if left <= right and left <= center * 0.8:
            return "L"
        elif right <= left and right <= center * 0.8:
            return "R"
        else:
            return "U"
    
    def process_frame(self, frame):
        """Process single frame and make navigation decision"""
        h, w = frame.shape[:2]
        total_risk = 0.0
        center_risk = 0.0
        detections = []

        # =========================
        # OBJECT DETECTION
        # =========================
        try:
            results = self.model(frame, conf=CONF_THRESH, verbose=False)
        except Exception as e:
            print(f"[ERROR] Model inference failed: {e}")
            results = []

        for r in results:
            if not getattr(r, "boxes", None):
                continue

            for box in r.boxes:
                # Extract coordinates safely
                try:
                    coords = box.xyxy[0]
                    if hasattr(coords, 'cpu'):
                        coords = coords.cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                except Exception:
                    continue

                cls = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                # 🔍 DRAW ALL YOLO DETECTIONS (DEBUG)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                cv2.putText(frame, f"{cls} {conf:.2f}",
                            (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (180, 180, 180), 1)

                # Only navigation-relevant classes continue
                if cls not in CLASS_WEIGHTS:
                    continue

                area_ratio = ((x2 - x1) * (y2 - y1)) / float(h * w)
                if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                zone = self.get_zone(cx, w)

                risk = self.calculate_risk(cls, area_ratio, zone, conf)

                detections.append({
                    "center_x": cx,
                    "center_y": cy,
                    "risk": risk
                })

                total_risk += risk
                if zone == "center":
                    center_risk += risk

                # 🔥 RISK VISUALIZATION
                color = (0, 255, 0)
                if risk > 1.2:
                    color = (0, 0, 255)
                elif risk > 0.6:
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"R:{risk:.2f}",
                            (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)

        # =========================
        # UPDATE STATE (AFTER ALL DETECTIONS)
        # =========================
        self.update_obstacle_grid(detections, w, h)

        # Stability logic
        if total_risk > RISK_THRESHOLDS["low"]:
            self.state["confirmed_frames"] = min(self.state["confirmed_frames"] + 1, FRAME_CONFIRM * 2)
        else:
            self.state["confirmed_frames"] = max(self.state["confirmed_frames"] - 1, 0)

        environment_blocked = total_risk > RISK_THRESHOLDS["low"]

        # =========================
        # NAVIGATION DECISION
        # =========================
        if environment_blocked and self.state["confirmed_frames"] >= FRAME_CONFIRM:
            direction = self.choose_direction()
            command = direction
            action_text = f"AVOID {direction}"
            action_color = (0, 0, 255)
        else:
            command = "F"
            action_text = "MOVING FORWARD"
            action_color = (0, 255, 0)

        # Stuck prevention
        if command == self.state["last_command"] and environment_blocked:
            self.state["stuck_counter"] += 1
        else:
            self.state["stuck_counter"] = 0

        if self.state["stuck_counter"] > 3 * max(self.state["fps"], 1):
            command = "U"
            action_text = "ESCAPE ASCEND"
            action_color = (255, 0, 255)
            self.state["stuck_counter"] = 0
            print("[WARNING] Executing escape maneuver")

        # Send command to hardware
        self.hardware.send_command(command)

        # Update state
        self.state["last_command"] = command
        self.state["command_history"].append(command)

        # =========================
        # VISUALIZATION
        # =========================
        # Main action display
        cv2.putText(frame, f"CMD: {command}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, action_color, 2)

        # Status panel
        status_y = 75
        status_lines = [
            f"Risk: {total_risk:.2f}",
            f"FPS: {self.state['fps']}",
            f"Stability: {self.state['confirmed_frames']}/{FRAME_CONFIRM}",
            f"Hardware: {'CONNECTED' if self.hardware.connected else 'SIMULATION'}"
        ]

        for line in status_lines:
            cv2.putText(frame, line, (20, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            status_y += 25

        # Draw navigation zones
        for zone_name, (zone_start, zone_end) in ZONES.items():
            x1_z, x2_z = int(zone_start * w), int(zone_end * w)
            zone_color = (100, 100, 100) if zone_name != "center" else (255, 255, 0)
            cv2.rectangle(frame, (x1_z, 0), (x2_z, h), zone_color, 1)

        # Hardware status (right side)
        hw_status = self.hardware.get_status()
        hw_text = f"Thr:{hw_status['throttle']} Pit:{hw_status['pitch']}"
        cv2.putText(frame, hw_text, (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

        return frame, command, total_risk
    
    def run(self):
        """Main execution loop"""
        fps_timer = time.time()
        
        try:
            while True:
                # Calculate FPS
                self.state["frame_count"] += 1
                if time.time() - fps_timer >= 1:
                    self.state["fps"] = self.state["frame_count"]
                    self.state["frame_count"] = 0
                    fps_timer = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARNING] Frame capture failed")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame, command, risk = self.process_frame(frame)
                
                # Display
                cv2.imshow("Autonomous Drone - AI Navigation", processed_frame)
                
                # Console output (reduced frequency)
                current_time = time.time()
                if command != self.state["last_command"] or current_time - self.state["last_risk_time"] > 2:
                    print(f"[{time.strftime('%H:%M:%S')}] CMD:{command} | Risk:{risk:.2f}")
                    self.state["last_risk_time"] = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Manual shutdown requested")
                    break
                elif key == ord('h'):
                    self.hardware.send_command("H")
                    print("[INFO] Hover command sent")
                elif key == ord('x'):
                    self.hardware.send_command("X")
                    print("[INFO] Emergency stop sent")
                elif key == ord('d'):
                    print(f"[DEBUG] Grid:\n{self.state['obstacle_grid']}")
                    print(f"[DEBUG] Cmd history: {list(self.state['command_history'])}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        print("[INFO] Shutting down system...")
        
        # Emergency stop
        self.hardware.send_command("X")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.hardware.shutdown()
        
        print("[INFO] System shutdown complete")

# =========================
# ARDUINO CODE (Upload this separately)
# =========================
"""
// ARDUINO CODE: flight_controller.ino
#include <Servo.h>

Servo motor_FL, motor_FR, motor_BL, motor_BR;

// Default neutral positions (in microseconds)
int throttle = 1000;  // STOP
int pitch = 1500;     // NEUTRAL
int roll = 1500;      // NEUTRAL
int yaw = 1500;       // NEUTRAL

void setup() {
  Serial.begin(9600);
  
  // Attach motors to pins (adjust based on your wiring)
  motor_FL.attach(9);
  motor_FR.attach(10);
  motor_BL.attach(11);
  motor_BR.attach(12);
  
  // Initialize to STOP
  updateMotors();
  
  Serial.println("Arduino Flight Controller Ready");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    processCommand(input);
  }
}

void processCommand(String cmd) {
  if (cmd.length() >= 12 && cmd[0] == 'T') {
    // Format: T1300P1500R1500Y1500
    throttle = cmd.substring(1, 5).toInt();
    pitch = cmd.substring(6, 10).toInt();
    roll = cmd.substring(11, 15).toInt();
    yaw = cmd.substring(16, 20).toInt();
    
    // Safety limits
    throttle = constrain(throttle, 1000, 2000);
    pitch = constrain(pitch, 1200, 1800);
    roll = constrain(roll, 1200, 1800);
    yaw = constrain(yaw, 1200, 1800);
    
    updateMotors();
    
    // Echo back for verification
    Serial.print("OK:T");
    Serial.print(throttle);
    Serial.print(" P");
    Serial.print(pitch);
    Serial.print(" R");
    Serial.print(roll);
    Serial.print(" Y");
    Serial.println(yaw);
  }
}

void updateMotors() {
  // Basic mixing for quadcopter (simplified)
  motor_FL.writeMicroseconds(throttle - pitch + roll - yaw);
  motor_FR.writeMicroseconds(throttle - pitch - roll + yaw);
  motor_BL.writeMicroseconds(throttle + pitch + roll + yaw);
  motor_BR.writeMicroseconds(throttle + pitch - roll - yaw);
}
"""

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("=" * 50)
    print("AUTONOMOUS DRONE NAVIGATION SYSTEM")
    print("FYP - BATCH SP23-BCS")
    print("=" * 50)
    
    # Run the system
    drone = AutonomousDrone()
    drone.run()
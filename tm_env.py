import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time
import pygetwindow as gw

class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.num_rays = 20
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32)
        
        self.sct = mss.mss()
        self.window_targeted = False
        
        # New robust tracking variables
        self.prev_gray_frame = None
        self.movement_history = []  # Stores the speed of the last 15 frames
        
        self.target_window()

    def target_window(self):
        try:
            windows = gw.getWindowsWithTitle('TrackMania')
            if not windows:
                windows = gw.getWindowsWithTitle('TmForever')
                
            if windows:
                window = windows[0]
                try:
                    window.activate()  
                    time.sleep(0.5)
                except Exception:
                    pass  
                
                if window.top >= 0 and window.left >= 0:
                    self.monitor = {
                        "top": window.top + 30,  
                        "left": window.left,
                        "width": window.width,
                        "height": window.height - 30
                    }
                    self.window_targeted = True
                    print(f"[ENV SUCCESS] Locked window grid at: {self.monitor}")
                    return
            
            self.monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
        except Exception:
            self.monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

    def _get_raycast_distances(self, gray_img):
        h, w = gray_img.shape
        car_x, car_y = int(w / 2), int(h * 0.82)
        distances = []
        
        angles = np.linspace(-np.pi/2, np.pi/2, self.num_rays)
        max_ray_len = int(min(w, h) * 0.55)
        
        for angle in angles:
            edge_found = False
            for r in range(10, max_ray_len, 6):
                target_x = int(car_x + r * np.sin(angle))
                target_y = int(car_y - r * np.cos(angle))
                
                if 0 <= target_x < w and 0 <= target_y < h:
                    pixel_value = gray_img[target_y, target_x]
                    if pixel_value < 35 or pixel_value > 220:
                        distances.append(r / max_ray_len)
                        edge_found = True
                        break
                else:
                    break
            
            if not edge_found:
                distances.append(1.0)
            
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        steer = action[0]
        
        for key in ['left', 'right', 'down']:
            pydirectinput.keyUp(key)
            
        pydirectinput.keyDown('up')
            
        if steer > 0.1: 
            pydirectinput.keyDown('right')
        elif steer < -0.1: 
            pydirectinput.keyDown('left')

        time.sleep(0.04) 

        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        obs = self._get_raycast_distances(gray)

        done = False
        speed_metric = 0.0

        if self.prev_gray_frame is not None:
            h, w = gray.shape
            sample_zone_y = int(h * 0.6)
            sample_zone_x = int(w * 0.5)
            
            # Measure how fast the road is rushing underneath the car
            diff = cv2.absdiff(
                gray[sample_zone_y-100:sample_zone_y+100, sample_zone_x-100:sample_zone_x+100], 
                self.prev_gray_frame[sample_zone_y-100:sample_zone_y+100, sample_zone_x-100:sample_zone_x+100]
            )
            speed_metric = np.mean(diff)
            
            # --- THE JITTER FIX: Rolling Average ---
            self.movement_history.append(speed_metric)
            if len(self.movement_history) > 15:
                self.movement_history.pop(0) # Keep only the last 15 frames
                
            # If we have 15 frames of data, check the average speed
            if len(self.movement_history) == 15:
                avg_speed = sum(self.movement_history) / 15.0
                # If the AVERAGE speed is dead, it is a crash. Bouncing between 0 and 1 won't save it.
                if avg_speed < 4.0:
                    print("💥 [CRASH] Jittering/Stuck detected! Resetting...")
                    done = True

        self.prev_gray_frame = gray.copy()
        
        # ---------------------------------------------------------
        # NEW HIGH-SPEED RACING REWARDS
        # ---------------------------------------------------------
        wall_distance = np.min(obs)
        forward_vision = (obs[9] + obs[10]) / 2.0
        
        # 1. The faster the car goes, the more points it gets. 
        # (Grass and flipping kill momentum, so the AI will naturally avoid them).
        reward = (speed_metric * 0.5) 
        
        # 2. Huge multiplier for keeping the car in the absolute center of the track.
        reward += (wall_distance * 4.0) 
        
        # 3. Bonus for aiming the nose of the car far down the road.
        reward += (forward_vision * 2.0)
        # ---------------------------------------------------------

        return obs, reward, done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        for key in ['up', 'down', 'left', 'right', 'w', 'a', 's', 'd']:
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
        time.sleep(0.1)
        
        pydirectinput.press('delete') 
        time.sleep(4.0)                
        
        if not self.window_targeted:
            self.target_window()

        self.movement_history = []
        self.prev_gray_frame = None

        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        obs = self._get_raycast_distances(gray)
        return obs, {}

    def close(self):
        pass
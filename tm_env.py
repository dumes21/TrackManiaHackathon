import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time

class TrackmaniaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action Space: [Steering (-1 to 1), Acceleration (-1 to 1)]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation Space: 20 Raycast distances (0.0 to 1.0)
        self.num_rays = 20
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32)
        
        self.sct = mss.mss()
        # TODO: Update these 4 values with the exact numbers from your calibration script
        self.monitor = {"top": 45, "left": 10, "width": 800, "height": 600} 
        
    def _get_raycast_distances(self, gray_img):
        """Fires 20 radial rays from the center-bottom (car position) to find track edges."""
        h, w = gray_img.shape
        car_x, car_y = int(w / 2), int(h * 0.85) # Approximate position of the car on screen
        distances = []
        
        # Generate 20 angles spread across a 180-degree field of view looking forward
        angles = np.linspace(-np.pi/2, np.pi/2, self.num_rays)
        max_ray_len = int(min(w, h) * 0.6) # Maximum length a laser beam can reach
        
        for angle in angles:
            edge_found = False
            for r in range(5, max_ray_len, 5): # Scan outward in steps of 5 pixels
                # Calculate the target pixel coordinate for this ray step
                target_x = int(car_x + r * np.sin(angle))
                target_y = int(car_y - r * np.cos(angle))
                
                # Boundary check to ensure we don't look outside the window
                if 0 <= target_x < w and 0 <= target_y < h:
                    # Threshold check: Trackmania roads are typically dark grey. 
                    # If pixel intensity drops significantly or shifts to green grass, count it as a wall.
                    pixel_value = gray_img[target_y, target_x]
                    if pixel_value < 50 or pixel_value > 200: # Tune these thresholds based on the map colors
                        distances.append(r / max_ray_len)
                        edge_found = True
                        break
                else:
                    break
            
            if not edge_found:
                distances.append(1.0) # If no obstacle is hit, return max distance
                
        return np.array(distances, dtype=np.float32)

    def step(self, action):
        steer, accel = action[0], action[1]
        
        # Reset keys to avoid stuck inputs
        for key in ['w', 's', 'a', 'd']:
            pydirectinput.keyUp(key)
            
        # Execute action via DirectX
        if accel > 0.1: pydirectinput.keyDown('w')
        elif accel < -0.1: pydirectinput.keyDown('s')
        if steer > 0.1: pydirectinput.keyDown('d')
        elif steer < -0.1: pydirectinput.keyDown('a')

        # Short wait step for physics frame resolution
        time.sleep(0.03) 

        # Capture and process frame
        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # Extract our ~20 clean numbers
        obs = self._get_raycast_distances(gray)

        # Reward Function: Higher distance away from walls = better line
        # If the middle rays see long distances, the car is aiming down a straightaway.
        forward_vision = (obs[9] + obs[10]) / 2.0
        wall_proximity_penalty = np.min(obs) # Penalty if any single ray gets dangerously close to 0
        
        reward = float(forward_vision + 0.5 * wall_proximity_penalty)
        
        # Simple terminal state: if a ray reads absolute zero, we have collided
        done = bool(wall_proximity_penalty < 0.05)

        return obs, reward, done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        pydirectinput.press('delete') # Trackmania instant-restart key
        time.sleep(0.5) 
        
        # Grab a clean initial screenshot to kick off the new episode
        img = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        obs = self._get_raycast_distances(gray)
        return obs, {}

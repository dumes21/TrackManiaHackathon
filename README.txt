=== TrackMania Hackathon — Setup Instructions ===

FOLDER STRUCTURE:
  TrackManiaHackathon/
    model.py
    train_sac.py
    agent.py
    requirements.txt

  C:\Users\debbi\TmrlData\config\
    config.json   <-- copy this file here (replace existing)

STEPS:
1. Install dependencies:
      pip install tmrl stable-baselines3[extra] torch numpy gymnasium

2. Copy config.json to:
      C:\Users\debbi\TmrlData\config\config.json
      (replace the existing file)

3. Open TrackMania via ModLoader with TMInterface enabled, load the map, start Time Attack

4. Terminal 1 — start server:
      python -m tmrl --server

5. Terminal 2 — start worker (game must be open first!):
      python -m tmrl --worker

6. Terminal 3 — start training:
      python train_sac.py

7. After training, weights.pt will appear in TrackManiaHackathon/
   This is your submission file alongside agent.py and model.py.

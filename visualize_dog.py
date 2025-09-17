import time
from pathlib import Path
import mujoco as mj
from mujoco import viewer
import numpy as np

def walk_dog():
  """
  Load a MuJoCo model (default floor.xml) and visualize it.
  """
  # Resolve path relative to this file's parent (project root assumption)
  model = mj.MjModel.from_xml_path("xml/floor.xml")
  data = mj.MjData(model)

  dt = model.opt.timestep
  start = time.time()
  t = 0.0
  viewer2 = viewer.launch_passive(model, data)
  while True:
    mj.mj_step(model, data)
    print(data.qpos)
    #data.ctrl = 10 * np.sin(t + np.array([0.0, 0.1, 0.2, 0.3]))
    t += dt
    time.sleep(dt)
    viewer2.sync()



if __name__ == "__main__": 
    # Run indefinitely in real time; Ctrl+C or close window to exit.
    walk_dog()

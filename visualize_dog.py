import time
from pathlib import Path
import mujoco as mj
from mujoco import viewer
import numpy as np


def gait(t, freq=1.0, phase=0.0):
  return np.sin(2 * np.pi * freq * t + phase)
  
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
  
  # gait parameters
  freq = 3
  amp_elbow = 4
  amp_arm = -0.7

  viewer2 = viewer.launch_passive(model, data)
  while True:
    # Compute phase offsets for trot gait
    fl_phase = 0.0
    fr_phase = np.pi
    bl_phase = np.pi
    br_phase = 0.0

    mj.mj_step(model, data)

    print(data.qpos)
    
    # Compute joint commands
    data.ctrl[:] = [
      0.0,                                     # fl shoulder
      amp_elbow * gait(t, freq, fl_phase),     # fl elbow
      amp_arm * gait(t, freq, fl_phase),       # fl arm
      
      0.0,                                     # fr shoulder
      amp_elbow * gait(t, freq, fr_phase),     # fr elbow
      amp_arm * gait(t, freq, fr_phase),       # fr arm
      
      0.0,                                     # bl shoulder
      amp_elbow * gait(t, freq, bl_phase),     # bl elbow
      amp_arm * gait(t, freq, bl_phase),       # bl arm
      
      0.0,                                     # br shoulder
      amp_elbow * gait(t, freq, br_phase),     # br elbow
      amp_arm * gait(t, freq, br_phase),       # br arm
    ]

    t += dt
    time.sleep(dt)
    viewer2.sync()



if __name__ == "__main__": 
    # Run indefinitely in real time; Ctrl+C or close window to exit.
    walk_dog()

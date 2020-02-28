import pytest
import os

exp_d = ""
for f in os.listdir(os.path.join("..", "examples", "periodicAveraging")):
    if f == "MNISTtorchCNN.py":
        continue
    else:
        exp_d = f

assert exp_d != ""
assert "MNISTtorchCNN" in exp_d

assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "summary.txt"))

assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "coordinator"))
assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "coordinator", "currentAveragedState"))

assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "worker0"))
assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "worker1"))
assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "worker2"))
assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "worker3"))
assert os.path.exists(os.path.join("..", "examples", "periodicAveraging", exp_d, "worker4"))

losses_file = os.path.join("..", "examples", "periodicAveraging", exp_d, "worker0", "losses.txt")
d = open(losses_file, "r").read()
assert len(d.split("\n")) == 351

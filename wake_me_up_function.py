import subprocess
import os
from pathlib import Path

def wake_me_up():
    path_to_script = os.path.join(Path(__file__).parent, "beep_noise.cmd")
    subprocess.run(["cmd.exe", "/c", path_to_script])

    print("WAKE UP AARON!!!" * 20)

wake_me_up()
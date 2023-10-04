import os, sys
import pathlib
file_dir = pathlib.Path(__file__).parent.resolve()
file_path = os.path.join(file_dir, "run_index.sh")

cmdl = ["bash", file_path]
cmdl += sys.argv[1:]
cmd = " ".join(cmdl)
os.system(cmd)

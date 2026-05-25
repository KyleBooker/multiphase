import os
import sys
import runpy

os.add_dll_directory(r"C:\Program Files\ngsolve-v6.2.2102\bin")
runpy.run_path(sys.argv[1], run_name="__main__")

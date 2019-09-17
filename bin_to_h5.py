import numpy as np
from tkinter import filedialog
from paths import *


file_path = filedialog.askopenfilenames(initialdir=master_path, filetypes=(("preproc files", "*.csv"),))

# with open(file_path) as f:

import os
import numpy as np

path = "/home/yiminfan/projects/ctb-liyue/yiminfan/smith_clean/SMITH_codebase/output_test1/index0"

for i in [19,39,59,79,99,119,139]:
    os.system("bash eval.sh " + path + "/epoch:" + str(i) + ".txt")
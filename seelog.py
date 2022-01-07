import os
import sys

class MyResDict(object):
    def __init__(self):
        self._data = []
        print("here")

    def insert(self, model_name, gpu, mp, bs, perf):
        self._data.append(((model_name, gpu, mp, bs), perf))

    def showme(self):
        self._data = sorted(self._data, key=lambda elem: elem[0])
        print(("model_name", "gpu", "mp", "bs", "perf"))
        for elem in self._data:
            if elem[0][1] == "1":
                print(elem)

def extract_info_from_file(path, file, res_dict):
    model_name = ""
    gpu_num = 0
    bs = 0
    # if the file is not exist.
    # do not execute training
    if not os.path.isfile(path + "/" + file):
        return
    f = open(path + "/" + file)
    bad = False
    if not os.path.isdir(file):
        fn_list = file.split(".")[1].split("_")
        # log.model_6B_bs_16_gpu_8_mp_1
        for i in range(len(fn_list)):
            if "model" in fn_list[i]:
                model_name = fn_list[i + 1]
            elif "bs" == fn_list[i]:
                bs = fn_list[i + 1]
            elif "gpu" == fn_list[i]:
                gpu = fn_list[i + 1]
            elif "mp" == fn_list[i]:
                mp = fn_list[i + 1]
        iter_f = iter(f)
        best_perf = 0
        for line in iter_f:
            if "OVERFLOW!" in line:
                bad = True
            if "GPU:" in line:
                sline = line.split()
                get_index = sline.index('GPU:')
                perf = float(sline[get_index + 3])
                if bad:
                    bad = False
                    continue
                best_perf = perf
        if best_perf != 0:
                res_dict.insert(model_name, gpu, mp, bs, best_perf)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PATH = str(sys.argv[1])
    else:
        PATH = "./logs"
    files = os.listdir(PATH)
    res_dict = MyResDict()
    for f in files:
        extract_info_from_file(PATH, f, res_dict)

    res_dict.showme()



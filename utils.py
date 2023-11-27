import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_data():
    file_path = ['./new_annotation_train/', './new_annotation_test/', './new_annotation_val/']
    for path in file_path:
        list_file = []
        for f in os.listdir(path):
            list_file.append(f"{path}{f}")
        if path.__contains__("test"):
            with open(f'./new_list_test.txt', 'w') as f:
                f.write('\n'.join(list_file))
        elif path.__contains__("val"):
            with open(f'./new_list_val.txt', 'w') as f:
                f.write('\n'.join(list_file))
        else:
            with open(f'./new_list_train.txt', 'w') as f:
                f.write('\n'.join(list_file))


def create_list_file():
    path = "./annotation/"
    list_file = []
    for f in os.listdir(path):
        list_file.append(f"{path}{f}")

    with open("./list.txt",'w') as f:
        for line in list_file:
            f.write(f"{line}\n")

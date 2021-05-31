import csv
from datetime import datetime

def get_name():
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    name = "outputs/correlation-" + date_time + ".csv"
    print("Writing to: ", name)

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_num','test_acc','test_loss','train_acc','train_loss','g_gap','mqBE_L1','mqBE_L2','mqBE_L3','mqBE_L4','mqBE_L5',
        'mqAE_L1','mqAE_L2','mqAE_L3','mqAE_L4','mqAE_L5','erBE_L1','erBE_L2','erBE_L3','erBE_L4','erBE_L5','erAE_L1','erAE_L2','erAE_L3','erAE_L4','erAE_L5',
        'rAE_L1','rAE_L2','rAE_L3','rAE_L4','rAE_L5'])

    return name

def write(name, line):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)

'''
if __name__ == "__main__":
    name = get_name()
    write(name, ["test", "hello"])
    write(name, ["test2", "hell2"])
    #write(name, ["test3", "hello3"])
'''
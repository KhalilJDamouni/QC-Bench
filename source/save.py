import csv
from datetime import datetime

def get_name(bench, dataset, hp):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    name = "outputs/results-" + date_time +"-"+bench+"-"+dataset+"-"+hp+ ".csv"
    print("Writing to: ", name)

    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_id','test_acc','test_loss','train_acc','train_loss','in_S_BE','out_S_BE','in_S_AE','out_S_AE',
        'in_C_BE','out_C_BE','in_C_AE','out_C_BE','in_ER_BE','out_ER_BE','in_ER_AE','out_ER_AE'])

    return name

def write(name, line):
    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)


# from matplotlib import pyplot as plt
from prettytable import PrettyTable



def count_parameters_trainable(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        summery = table.get_string()
    with open("model_summery.txt", "w") as f:
        f.write(summery)
        f.write("\n Total Trainable Params:")
        f.write(str(total_params))
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params


def count_parameters_total(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        summery = table.get_string()
    with open("model_summary.txt", "w") as f:
        f.write(summery)
        f.write("\n Total Trainable Params:")
        f.write(str(total_params))
    # print(table)
    # print(f"Total Params: {total_params}")
    return total_params



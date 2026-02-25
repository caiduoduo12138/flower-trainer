from utils import DataList

data_list = DataList()


def add_data(exp_name: str, pid: str):
    data_list.put({exp_name: pid})
    return None


def update_data(exp_name: str, pid: str):
    for index, each in enumerate(data_list.get()):
        if list(each.keys())[0] == exp_name:
            data_list.update(index, {exp_name: pid})
    return None


def pop_data(exp_name: str, pid: str):
    _ = pid
    for each in data_list.get():
        if list(each.keys())[0] == exp_name:
            data_list.pop(each)

    return None


def get_all():
    return data_list.get()

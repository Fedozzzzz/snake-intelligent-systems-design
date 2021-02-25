import json


def get_json_info(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def set_json_info(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    print('db_module.py file')

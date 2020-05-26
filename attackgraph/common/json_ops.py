""" Jason utility operations. """
import ujson as json

import attackgraph.common.file_ops as fp


def load_json_data(path):
    '''
    Loads the data from the file as Json into a new object.
    '''
    if not fp.isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path) as data_file:
        result = json.load(data_file)
    return result


def save_json_data(path, json_obj):
    '''
    Prints the given Json object to the given file name.
    '''
    with open(path, 'w') as my_file:
        json.dump(json_obj, my_file)

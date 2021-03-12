import pickle

def parse_flatten_dict(data_dict: dict, key_prefix = '', return_dict: dict = dict()):
    for key in data_dict.keys():
        if type(data_dict[key]) == type(dict()):
            if key_prefix == '':
                new_prefix = key
            else:
                new_prefix = key_prefix + '.' + key

            return_dict = parse_flatten_dict(data_dict[key], new_prefix, return_dict)
        
        else:            
            if key_prefix == '':
                new_key = key
            else:
                new_key = key_prefix + '.' + key

            return_dict[new_key] = data_dict[key]

    return return_dict

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


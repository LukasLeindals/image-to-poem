import datetime
import json
import os
import shutil

def flatten_list(l: list) -> list:
    """
    Flatten a list of lists into a list.
    Args:
        l: a list of lists
    Returns:
        a flattened list
    """
    out = []
    for sublist in l:
        # continue if the sublist is empty
        if len(sublist) == 0:
            continue
        
        # if the sublist is not a list, just add it to the output
        if not isinstance(sublist[0], list):
            out += sublist
        # otherwise we will recursively call this function to get the elements out
        else:
            out += flatten_list(sublist)
    
    return out

def format_time(time_in_seconds: float) -> str:
  return str(datetime.timedelta(seconds=int(round(time_in_seconds))))

def update_param_dict(params, param_updates):
    """
    Update the parameter dictionary with new values.

    Parameters
    ----------
    params : dict
        The parameter dictionary to update.
    param_updates : dict
        The dictionary of parameters to update.

    Returns
    -------
    dict
        The updated parameter dictionary.
    """
    for key, value in param_updates.items():
        if isinstance(value, dict):
            params[key] = update_param_dict(params[key], value)
        else:
            params[key] = value
    return params

def load_json_file(path):
    with open(path) as f:
        json_context = json.load(f)
    return json_context

def save_json_file(path, data):
    if os.path.exists(path):
        print(f"File {path} already exists")
        path, ext = os.path.splitext(path)
        new_path = path + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ext
        print(f"Saving file as {new_path} instead")
        save_json_file(new_path, data)
        
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
def zip_model(model_dir):
    if model_dir.endswith("/"):
        model_dir = model_dir[:-1]
    # zip_dir = model_dir + ".zip"
    zip_dir = model_dir
    shutil.make_archive(zip_dir, 'zip', model_dir)
    
def unzip_model(zip_dir):
    model_dir = zip_dir.replace(".zip", "")
    shutil.unpack_archive(zip_dir, model_dir)
    
if __name__ == "__main__":
    # zip_model("models/language_models/max_len-500")
    # zip_model("models/similarity/model_20231129_221129")
    unzip_model("models/language_models/max_len-500.zip")
    unzip_model("models/similarity/model_20231129_221129.zip")
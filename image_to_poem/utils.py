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
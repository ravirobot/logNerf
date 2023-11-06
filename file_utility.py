import os

def get_filenames(dir_path, endings=None):
    if endings:
        endings = set(endings)
    res = []
    for (directory, _ , file_names) in os.walk(dir_path):
        for fname in file_names:
            if endings:
                for ending in endings:
                    if fname[-len(ending):] == ending:
                        res.append(os.path.join(directory, fname))
                        break
            else:
                res.append(os.path.join(directory, fname))
    return res
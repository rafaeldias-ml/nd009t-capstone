import pickle


def pickle_obj(obj, base_filename):
    """
    Serializes an object to disk
    ---
    Arguments
        obj: Object
            object to be serialized
        base_filename: string
            filename (without extension) of destination file
    Returns
        None
    """
    with open('%s.pickle' % base_filename, 'wb') as file:
        pickle.dump(obj, file, 4)

def unpickle_obj(base_filename):
    """
    Deserializes an object from disk
    ---
    Arguments
        base_filename: string
            filename (without extension) of source file
    Returns
        obj: Object
            Deserialized file
    """    
    with open('%s.pickle' % base_filename, 'rb') as file:
        obj = pickle.load(file)
        
        return obj

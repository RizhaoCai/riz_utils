import h5py as h5



def load_h5(db,fetct_list):
    """
        @param db:  an object of h5 dataset
        @param fetct_dict: a list of str that contains the data we want to fetch from the db
        @stride 
    """
    fetched_data = []
    for fl in fetct_list:
        fetched_data.append(db[fl])
    return fetched_data

def load_h5_to_mem(db,fetct_list,slice_operation):
    """
        load data from h5 to memory
        slice_operation: "[:]"
                         "[::stride]"
                         "[start:end]"
    """
    fetched_data = load_h5(db,fetct_list)
    fetched_data_mem = []
    for i in range(len(fetched_data)):
        data = eval("fetched_data[i]"+slice_operation)
        fetched_data_mem.append(data)
    return fetched_data_mem # numpy data

def load_h5_from_dir(h5_dir,fetct_list):
    db = h5.File(h5_dir,"r")
    return load_h5(db,fetct_list)

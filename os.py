import sys
import os



def make_dir(directory_path,rebuild_it=False):
    """
        Make a directory recursively:
            If the directory have already existed, return False.
            IF the directory do not exsis, creat it.
    """
    assert  type(directory_path) == str  
    is_existent = os.path.exsis(directory_path)
    
    if is_existent:
        print("Already existed: {} ".format(directory_path))
        return False
    elif rebuild_it:
        pass
    else：
        """Create"""
        os.makedirs(directory_path)
        print("Created: {}".format(directory_path))
        return True
    
def clear_dir(directory_path,recursively=False,show_detail=False):
    
    print("Clear dir:{}".format(directory_path))
    is_existent = os.path.exsis(directory_path)
    if not is_existent:
        print("    Failed! This dir does not exist")
        return False
    else:
        ls = os.listdir(directory_path)
        for f in ls:
            f_path = os.path.join(directory_path,f)
            if os.path.isdir(f_path) and recursively:
                os.removedirs(f_path)
            else:
                os.remove(f_path)
    return True
    

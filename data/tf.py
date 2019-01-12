"""
    Convert raw pixel data to TFRecords file format with example proto.s
    cairizhao@email.szu.edu.cn
"""
import tensorflow as tf
import numpy as np
import os
import sys
import h5py as h5
import numpy as np


from glob import glob


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_tf_features(X,Y):
    """
        @param X: a 4-D numpy array of shape [N, H, W, C]
        @param Y: a one-hot numpy array of shape [N,1] or [N] 
    """
    num_X = X[0]
    height_X = X[1]
    width_X =  X[2]
    channle_X = X[3]

    features = []
    for i in range(num_X):
        X_raw = X[i].tostring()
        feature=tf.train.Features(feature={
                    'X': _bytes_feature(X_raw ), 
                    'width_X': _int64_feature(width_X),
                    'height_X': _int64_feature(height_X),
                    'channel_X': _int64_feature(channel_X),
                    'Y':_int64_feature(Y)
                    })
                             
        features.append(feature)
    return features

def numpy_2_tfrecord(features,save_path):
    """
        @param examples: a list of [data,ground_truth] 
    """
    print('Writing numpy array to tfrecord: {}'.format(save_path))
    n_features = len(features)
    with tf.python_io.TFRecordWriter(save_path) as writer:
        for i in range(n_features):
            print("Transforming: {}".format(i))
            feature=tf.train.Features(feature=features)          
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            print(" Serializing Finished!")
    

def np2tfrecord(X, label, max=255, save_path='train.tfrecords'):
    # 125.71416745682146,102.20704946337348,96.51020497263258,37.684522777704224,31.584378331225047,29.870035466635755 
    num_samples = X.shape[0]
    size_X =X.shape[-2]

  
    channel_X = X.shape[-1]
    if max==255:
        X = norm(X) 
    print('Writing', save_path)
    with tf.python_io.TFRecordWriter(save_path) as writer:
        for i in range(num_samples):
            print("Transform: {} done".format(i))
            X_raw = X[i].tostring()
           
            lbl = label[i]
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'X': _bytes_feature(X_raw ),
                 
                    'size_X': _int64_feature(size_X),
                   
                    'channel_X': _int64_feature(channel_X),
                
                    'label':_int64_feature(lbl)
                        }))
            print("All Transformed! Serializing!")
            writer.write(example.SerializeToString())
            print(" Serializing Finished!")

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'X': tf.FixedLenFeature([], tf.string), 
          'width_X': tf.FixedLenFeature([], tf.int64),
          'height_X': tf.FixedLenFeature([], tf.int64),
          'channel_X': tf.FixedLenFeature([], tf.int64),       
           'label': tf.FixedLenFeature([], tf.int64),
      })
    image = tf.decode_raw(features['X'], tf.float32)
    height_X = tf.cast(features['height_X'],tf.int64)
    width_X = tf.cast(features['width_X'],tf.int64)
    channel_X = tf.cast(features['channel_X'],tf.int64)
    label = tf.cast(features['label'],tf.int64)
    # When converted to serialized data, the information of shape gets lost, so it should be recovered
    total_size = height_X*width_X*channel_X
    image_shape = [height_X,width_X,channel_X]

    #print("before decode: {}".format(face_image.shape) )
    image.set_shape(total_size)
    image=tf.reshape(image,image_shape)

    return image, label






def data_iterator(batch_size, num_epochs, files_path, is_training):
    """Reads input data num_epoch times
        Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
     Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
    """ 
    # the 'filename' could also be a list of filenames, which will be read in order
    dataset = tf.data.TFRecordDataset(files_path)
    #dataset = tf.data.TFRecordDataset([filename])
    # The map transformation taks a function and applies it to every element 
    # of the dataset
    dataset = dataset.map(decode)
    # The shuffle transformation uses a finite-sized buffer to shuffle elelemts
    # in memory. The parameter is the number of elements in the buffer. For 
    # completely uniform shuffling, set the parameter to be the same as the 
    # number of elements in the  dataset.
    # batch_first, and then repeat
    if is_training:
        dataset = dataset.shuffle(1000 + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator



def test_iterator(name_scope,batch_size, filepath):
    
    filename = filepath
    with tf.name_scope(name_scope):  
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode)
        dataset = dataset.batch(batch_size) 
        iterator = dataset.make_one_shot_iterator()
    return iterator

def test_iterator_alex(name_scope,batch_size, filepath):
    
    filename = filepath
    with tf.name_scope(name_scope):
        # the 'filename' could also be a list of filenames, which will be read in order
        dataset = tf.data.TFRecordDataset(filename)
        #dataset = tf.data.TFRecordDataset([filename])
        # The map transformation taks a function and applies it to every element 
        # of the dataset
        dataset = dataset.map(decode_alex)

        # The shuffle transformation uses a finite-sized buffer to shuffle elelemts
        # in memory. The parameter is the number of elements in the buffer. For 
        # completely uniform shuffling, set the parameter to be the same as the 
        # number of elements in the  dataset.
        # batch_first, and then repeat
        dataset = dataset.batch(batch_size) 
        iterator = dataset.make_one_shot_iterator()
    return iterator

def test_iterator_onehot(name_scope,batch_size, filepath):
    
    filename = filepath
    with tf.name_scope(name_scope):
        # the 'filename' could also be a list of filenames, which will be read in order
        dataset = tf.data.TFRecordDataset(filename)
        #dataset = tf.data.TFRecordDataset([filename])
        # The map transformation taks a function and applies it to every element 
        # of the dataset
        dataset = dataset.map(decode_onehot)

        # The shuffle transformation uses a finite-sized buffer to shuffle elelemts
        # in memory. The parameter is the number of elements in the buffer. For 
        # completely uniform shuffling, set the parameter to be the same as the 
        # number of elements in the  dataset.
        # batch_first, and then repeat
        dataset = dataset.batch(batch_size) 
        iterator = dataset.make_one_shot_iterator()
    return iterator 

def load_h5_data(mat, set_type, stride=1):
    X = mat[set_type+'_X'][::stride]
    label =  mat[set_type+'_LBL'][::stride]   
    #X = 1
    #depth_map=2
     
    return np.moveaxis(X,[1,2,3],[-1,-2,-3]),  label
     

def adv_tfrecord(file_dir,save_dir):
    db = h5.File(file_dir,mode='r')
    stride=1
    X = db["X"][::stride] # X is 4-D array of image data of float dtype, whose value ranges from 0~1
    Y = db["Y"][::stride] # Y is 1-D array of label, and 0 for fake, 1 for real
    X = tf.float32(X)
    print("Making TF Record: writing {}".format(save_dir))
    np2tfrecord(X,Y, 0, save_dir)

if __name__ == '__main__':
    
    """
        TRAIN_X, TRAIN_D, TRAIN_LBL
        TEST_X, TEST_D, TEST_LBL
    """
    # make training, tesing, validation examples 
    """
    
    
    import IPython; IPython.embed()
    """
    
    np2tfrecord(X,Y,0,save_dir)
    #root_path = "/data1/user_datas/cairizhao/lbp_cnn/CASIA/lbp_coded/fgm/vggnet"

    #for f in glob(root_path+"*.h5df"):
    #    tf_dir = "/data1/user_datas/cairizhao/lbp_cnn/CASIA/lbp_coded/fgm/vggnet/tfrecord/" + f.split("/")[10]
    #     tf_dir = tf_dir.replace(".h5df",".tfrecord")
    #    adv_tfrecord(f,tf_dir)
    # make advsarial examples of validation set

   

    

    





    

   

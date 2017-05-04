
# coding: utf-8

# In[5]:

import tensorflow as tf
import numpy as np
import os
import math
import pexpect
import time
import pandas as pd


# In[ ]:

def rnn_bootstrap_batch_sorter(features_data, targets_data, min_batch_size, num_steps, num_networks = 100, test_ratio = 0.15,
                     val_ratio = 0.15, silent = False, method = 'simple_block'):
    """
    Sorts RNN data properly (i.e. maximises the number of states which are passed on correctly). Will split into
    train/val and test data first, and then put the data in iterators to pass to the RNN
    
    features_data: shape [size_data, num_features]
    targets_data: shape [size_data, num_targets]
    min_batch_size: minimum batch_size for training data (for test batch_size will be one)
    num_steps: number of steps RNN rolled out for during training
    num_networks: the number of networks being trained at once
    test_ratio: proportion of data for test set. Val proportion will be determined by bootstrap draws.
    val_ratio: proportion of the train/val dataset used for val data (only applies if method = simple_block)
    silent: whether or not to print batch_sizes 
    method: the method of bootstrapping. One of ['simple_block']
    
    Returns:
    A dictionary with iterators holding datasets
    """
    
    assert targets_data.shape[0] == features_data.shape[0], 'Targets and Features data different sizes'
    assert len(targets_data.shape) == 2, 'Targets data needs to be at least 2D'
    
    num_data_points = features_data.shape[0]
    num_features = features_data.shape[1]
    num_targets = targets_data.shape[1]
        
    # Split the data into train/val and test sets
    test_size = int(math.ceil(test_ratio * num_data_points))
    train_size = int(num_data_points - test_size)
    
    train_f_data = features_data[0: train_size, :]
    train_t_data = targets_data[0: train_size, :]

    test_t_data = targets_data[train_size:, :]
    test_f_data = features_data[train_size:, :]
    
    if not silent:
        print "Train/ Val data: {} observations".format(train_t_data.shape[0])
        print "Test data: {} observations\n".format(test_t_data.shape[0])

    iter_dict = bootstrap_rnn_data(train_f_data, train_t_data, test_f_data, test_t_data, min_batch_size = min_batch_size,
                                  num_steps = num_steps, num_networks = num_networks, val_ratio = val_ratio, silent = silent,
                                  method = method)
    return iter_dict


# In[ ]:

def bootstrap_rnn_data(tr_features_data, tr_targets_data, test_features_data, test_targets_data, min_batch_size, 
                       num_steps, num_networks = 100, val_ratio = 0.15, silent = False, method = 'simple_block'):
    
    """
    Bootstraps RNN data properly (i.e. maximises the number of states which are passed on correctly). 
    Train/Test split already provided. Returns the data in iterators to pass to the RNN
    
    tr_features_data: shape [size_train_data, num_features], includes both val and train data
    tr_targets_data: shape [size_train_data, num_targets], includes both val and train data
    test_features_data: shape [size_test_data, num_features]
    test_targets_data: shape [size_test_data, num_targets]
    min_batch_size: minimum batch_size for training data (for test batch_size will be one)
    num_steps: number of steps RNN rolled out for during training
    num_networks: the number of networks being trained at once
    val_ratio: proportion of the train/val dataset used for val data (only applies if method = simple_block)
    silent: whether or not to print batch_sizes 
    method: the method of bootstrapping. One of ['simple_block']
    
    Returns:
    A dictionary with iterators holding datasets
    """
    train_size = tr_targets_data.shape[0]
    
    # Make the masks for each network's draw
    val_size = int(val_ratio * train_size)
    if method == 'simple_block':
        mask_list = []
        inf_mask_list = []  # A flattened version of the mask list for calculating the val/train losses separately
        for b in range(num_networks):
            val_start_index = np.random.choice(np.arange(train_size - val_size))
            mask = np.ones([tr_targets_data.shape[0], 1])
            mask[val_start_index:val_start_index + val_size, :] = 0
            mask_batches, _ = sort_batches(mask, min_batch_size = min_batch_size, num_steps = num_steps, 
                                           dtype = 'train', t_or_f = None)
            inf_mask_batches, _ = sort_batches(mask, min_batch_size = min_batch_size, num_steps = num_steps,
                                               dtype = 'test', t_or_f = None)
            mask_list.append(mask_batches)
            inf_mask_list.append(inf_mask_batches)
        mask_list = zip(*mask_list)
        inf_mask_list = zip(*inf_mask_list)
        
        all_mask_list = []
        for m in mask_list:
            all_mask_list.append(np.concatenate([np.expand_dims(i, 0) for i in m]))
            
        all_inf_mask_list = []
        for m in inf_mask_list:
            all_inf_mask_list.append(np.concatenate([np.expand_dims(i,0) for i in m]))
    
    # Sort the data into batches
    train_f_batches, tr_seq_lengths = sort_batches(tr_features_data, min_batch_size = min_batch_size, num_steps = num_steps, 
                                                   dtype = 'train', t_or_f = 'features', silent = silent)
    train_t_batches, _  = sort_batches(tr_targets_data, min_batch_size = min_batch_size, num_steps = num_steps,
                                       dtype = 'train', t_or_f = 'targets', silent = silent)                
    test_f_batches, test_seq_lengths = sort_batches(test_features_data, min_batch_size = min_batch_size, num_steps = num_steps,
                                                    dtype = 'test', t_or_f = 'features', silent = silent)
    test_t_batches, _  = sort_batches(test_targets_data, min_batch_size = min_batch_size, num_steps = num_steps,
                                      dtype = 'test', t_or_f = 'targets', silent = silent)
    
    # Train data with batch_size of 1 for inference - ensures correct state always passed on
    trinf_f_batches, trinf_seq_lengths = sort_batches(tr_features_data, min_batch_size = min_batch_size, num_steps = num_steps,
                                                      dtype = 'train_inference', t_or_f = 'features', silent = silent)
    trinf_t_batches, _  = sort_batches(tr_targets_data, min_batch_size = min_batch_size, num_steps = num_steps,
                                       dtype = 'train_inference', t_or_f = 'targets', silent = silent)
    
    # Make masks of all ones for the test and train_inf data
    test_masks = [np.ones([num_networks, 1, num_steps, 1]) for t in test_t_batches]
    trinf_masks = [np.ones([num_networks, 1, num_steps, 1]) for t in trinf_t_batches]

    # Put the data in interators
    # Iterator for training
    train_iter = rnn_batch_iterator(train_f_batches, train_t_batches, tr_seq_lengths, masks = all_mask_list)
    
    # Iterators for prediction once training finished
    test_iter = rnn_batch_iterator(test_f_batches, test_t_batches, test_seq_lengths, masks = test_masks)
    trinf_iter = rnn_batch_iterator(trinf_f_batches, trinf_t_batches, trinf_seq_lengths, masks = trinf_masks)
        
    # Iterator for evaluation of train/val loss during training
    tr_loss_iter = rnn_batch_iterator(trinf_f_batches, trinf_t_batches, trinf_seq_lengths, masks = all_inf_mask_list)
    
    return {'train': train_iter, 'test': test_iter, 'tr_inf': trinf_iter, 'tr_loss': tr_loss_iter}


# In[ ]:

def sort_batches(data, min_batch_size, num_steps, dtype, t_or_f, silent = True):
    """
    Sorts batches in most efficient way for RNN
    data: shape [num_data_points, targets/features dimension]
    min_batch_size: minimum batch_size for training data (for test batch_size will be one)
    num_steps: number of steps RNN rolled out for during training
    dtype: one of [train, test, 'train_inference']
    t_or_f: one of [features, targets]

    Returns:
    batches: the data sorted into batches
    seq_lengths: the corresponding sequence lengths 
    """

    num_data_points = data.shape[0]

    if dtype == 'train':
        # Calculate the batch size - basically keeping the same number of batches as would be required by min_batch_size,
        # but increasing the size to avoid wasting data
        batches_required = int(math.floor(num_data_points/float(min_batch_size * num_steps)))
        b_length = int(math.floor(num_data_points/float(batches_required * num_steps)))
        num_zeros = 0
    elif dtype in ['test', 'train_inference']:
        b_length = 1

        if num_data_points % (num_steps*b_length) == 0:
            num_zeros = 0
        else:
            num_zeros = int((num_steps*b_length) - (num_data_points % (num_steps*b_length)))             

    # Use NaN's at this point, so no mistakes if any of features or targets are actually 0
    filler = np.empty(shape = [num_zeros, data.shape[1]])
    filler[:] = np.NAN
    data = np.concatenate([data, filler], axis = 0)
    num_batches = int(len(data)/(b_length*num_steps))

    batches = []
    seq_lengths = []

    for i in range(num_batches):
        input_matrix = np.zeros((b_length, num_steps, data.shape[1]))
        for j in range(b_length):
            row_start_index = (0 + (num_steps*i)) + (j*num_batches*num_steps)
            input_matrix[j] = data[row_start_index:(row_start_index + num_steps)]

        # Find where the nan's are to calculate sequence length
        seq_len = np.ones([b_length]) * num_steps
        seq_len[-1] = num_steps - np.sum(np.isnan(input_matrix[-1,:,0]))
        seq_lengths.append(seq_len)

        # Replace nan's with zeros
        input_matrix[np.isnan(input_matrix)] = 0
        batches.append(input_matrix)

    if silent == False:
        print "{} {} {} batches created, of shape {} [batch_length * num_steps * num_features]".format(len(batches),
                                                                            dtype, t_or_f, batches[0].shape)

    return batches, seq_lengths


# In[7]:

class RNN(object):
    
    def __init__(self, sess, batch_iterators, num_steps, num_layers = 1, hidden_size = 10, cell_type = 'basic', 
                 activation_fn = tf.nn.tanh, learning_rate = 0.01, pass_state = False, model_name = 'rnn', 
                 checkpoint_dir = 'rnn_checkpoint'):
        
        self.sess = sess # A tensorflow session
        
        # Data iterators - hold the data sets, and return the next batch when next_batch() called
        self.train_iter = batch_iterators['train'] # Training data
        self.val_iter = batch_iterators['val'] # Val data - batch_length of 1
        self.test_iter = batch_iterators['test'] # Test data - batch_length of 1
        self.train_inf_iter = batch_iterators['tr_inf'] # Train data for predictions at end - batch_length of 1
        
        self.num_steps = num_steps # how many steps to unroll the RNN during training
        self.num_layers = num_layers # number of layers of the RNN
        self.hidden_size = hidden_size # the number of units in each RNN cell
        self.cell_type = cell_type # one of ['basic', 'gru', 'lstm'] - which RNN cell to use
        self.activation_fn = activation_fn # activation function for the output layer
        
        self.learning_rate = learning_rate # the learning rate for training
        self.pass_state = pass_state # whether to pass state on from last batch in epoch to first batch during training
                
        self.model_name = model_name # Name for the model - used in name of saved checkpoint files
        self.checkpoint_dir = checkpoint_dir # Directory in which to save checkpoint files
        
        self.targets_dim = self.train_iter.targets_dim # dimension of the target variable
        self.features_dim = self.train_iter.features_dim # number of features 
                
        self.build_model()
        
        self.saver = tf.train.Saver()
        
    def build_model(self):
        
        self.features_pl = tf.placeholder(tf.float32, [None, self.num_steps, self.features_dim], 'features_pl')
        self.targets_pl = tf.placeholder(tf.float32, [None, self.num_steps, self.targets_dim], 'targets_pl')
        self.seq_len_pl = tf.placeholder(tf.int32, shape = [None], name = 'seq_len_pl')
        self.b_size_pl = tf.placeholder(tf.int32, shape = (), name = 'b_size_pl')
        
        # Reshape targets to correct size and crop off any zero padding
        targets = tf.concat(tf.unstack(self.targets_pl, num = self.num_steps, axis = 1), axis = 0)
        self.targets = targets[:self.b_size_pl, :]
        
        # Unstack features into num_step length list of [batch_size * features_dim] tensors
        self.features = tf.unstack(self.features_pl, num = self.num_steps, axis = 1)
        
        # Set up the RNN cell
        if self.cell_type == 'basic':
            cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size, activation = self.activation_fn)
        elif self.cell_type == 'gru':
            cell = tf.contrib.rnn.GRUCell(self.hidden_size, activation = self.activation_fn)
        
        multi_cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple = True)
        
        # Placeholders for the states each layer of cells
        self.initial_state_list = []
        for l in range(self.num_layers):
            self.initial_state_list.append(tf.placeholder(tf.float32, [None, self.hidden_size], name = 'init_state_pl_'+ str(l)))
        
        # Set up the RNN
        self.cell_outputs, self.final_state  = tf.contrib.rnn.static_rnn(multi_cell, self.features, 
                                    initial_state = tuple(self.initial_state_list), sequence_length = self.seq_len_pl)
        
        # Join outputs into shape [b_length * num_steps, hidden_dim] and pass through output layer
        joined_outputs = tf.concat(self.cell_outputs, axis = 0)[:self.b_size_pl, :]
        self.output = tf.contrib.layers.fully_connected(joined_outputs, self.targets_dim, activation_fn=None, scope = 'final_layer')
        
        # Loss        
        self.loss = tf.reduce_mean(tf.pow(self.targets - self.output, 2))
                  
        # Optimizer
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
                
    def train(self, viz_every = 500, num_steps = 5000):
        """
        Train the network, and calculates final predictions and loss for each dataset once done
        """
        
        best_val_loss = float('inf')
        self.sess.run(tf.global_variables_initializer())
                
        for step in xrange(num_steps):
            
            if step == 0:
                state = self.fill_zero_state(self.train_iter)
            if self.train_iter.counter == 0:
                if self.pass_state:
                    state = self.update_train_state(state)                    
                else:
                    state = self.fill_zero_state(self.train_iter)
                
            f_batch, t_batch, seq_len = self.train_iter.next_batch()
            b_size = int(seq_len[-1] * f_batch.shape[0])
             
            # Fill the feed_dict    
            feed_dict = {self.features_pl: f_batch, self.targets_pl: t_batch, self.seq_len_pl: seq_len, 
                         self.b_size_pl: b_size}
            for l in range(self.num_layers):
                feed_dict[self.initial_state_list[l]] = state[l]
            
            # Run a train step
            ops = {"opt": self.opt, "final_state": self.final_state}            
            returns = self.sess.run(ops, feed_dict = feed_dict)
            
            # Pass final state of previous batch onto next batch
            state = returns["final_state"]

            # Check progress
            if step % viz_every == 0:
                _, TRAIN_LOSS, tr_state= self.run_data_set(self.train_inf_iter)
                _, VAL_LOSS, _ = self.run_data_set(self.val_iter, previous_state = tr_state)

                print "Step: {0}, Train Loss: {1:.4f}, Val Loss: {2:.4f}".format(step,TRAIN_LOSS, VAL_LOSS)            

                if VAL_LOSS < best_val_loss:
                    self.save()
                    best_val_loss = VAL_LOSS
                    
        # Restore the variables which achieved the lowest validation loss           
        self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.model_name)
        
        # Get final predictions and loss for all data sets
        self.TRAIN_PREDS, TRAIN_LOSS, tr_state = self.run_data_set(self.train_inf_iter)
        self.VAL_PREDS, VAL_LOSS, val_state = self.run_data_set(self.val_iter, previous_state = tr_state)
        self.TEST_PREDS, TEST_LOSS, _ = self.run_data_set(self.test_iter, previous_state = val_state)
                
        print "Final Losses, Train: {1:.4f}, Val: {2:.4f}, Test: {3:.4f}".format(step,
                                                                            TRAIN_LOSS, VAL_LOSS, TEST_LOSS) 
                
    def run_data_set(self, iterator, previous_state = None):
        """
        Calculates the predictions and average loss for the whole dataset stored in iterator
        """
        
        if previous_state == None:
            state = self.fill_zero_state(iterator)
        else:
            state = previous_state

        # Store starting value of iterator to return to
        counter_start = iterator.counter
        # Make sure we start from the first batch
        iterator.counter = 0
        
        # Lists for storing the returns from each batch
        preds_list = []
        loss_list = []
        
        for step in xrange(iterator.num_batches):
            
            # Get the next batches of data
            f_batch, t_batch, seq_len = iterator.next_batch()
            b_size = int(seq_len[-1] * f_batch.shape[0])
             
            # Fill the feed dict
            feed_dict = {self.features_pl: f_batch, self.targets_pl: t_batch, 
                         self.seq_len_pl:seq_len, self.b_size_pl: b_size}
            for l in range(self.num_layers):
                feed_dict[self.initial_state_list[l]] = state[l]
             
            # Run the ops
            ops = {"final_state": self.final_state, "loss": self.loss, "preds": self.output}          
            returns = self.sess.run(ops, feed_dict = feed_dict)
            
            # Pass final state onto next batch
            state = returns["final_state"]
            
            # Store the loss and predictions from current batch
            preds_list.append(returns["preds"])
            loss_list.append(returns["loss"])
        
        # Join the losses and predictions from all the batches
        loss = np.average(loss_list)
        preds = np.concatenate(preds_list, axis = 0)

        # Return iterator counter to starting value
        iterator.counter = counter_start
        
        return preds, loss, state
   
    def fill_zero_state(self, iter_):  
        """
        Returns state filled with zeros for start of training/eval
        iter_: data iterator
        """
        state = []
        for l in range(self.num_layers):
            state.append(np.zeros([iter_.batch_size, self.hidden_size]))
        return state

    def update_train_state(self, prev_state):
        """
        Takes the state from last training batch in epoch and shifts it down by one row, so 
        that correct state is passed to the first training batch in the next epoch
        prev_state: the state from the final batch of the previous epoch
        """
        old_state = list(prev_state)
        new_state = []
        for l in range(self.num_layers):
            s = old_state[l]
            new_s = np.concatenate([np.zeros([1, self.hidden_size], 
                                dtype = np.float64), s], axis = 0)[:-1,:]
            new_state.append(new_s)        
        return tuple(new_state)

    def save(self):
        """
        Saves a checkpoint file
        """
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir + '/' + self.model_name)


# In[8]:

def mse(targets, predictions):
    """
    Calculates the mean squared error
    """
    assert targets.shape == predictions.shape, 'Targets and predictions arrays not the same shape'
    residuals = targets - predictions
    residuals_squared = residuals ** 2
    MSE = np.average(residuals_squared)
    return MSE   


# In[2]:

def lin_scale(data, low = 0.0, high = 1.0):
    """
    Scales data between low and high
    """
    assert len(data.shape) == 2, "Data needs to be 2d"
    ratio = (high-low)/(np.max(data, axis = 0) - np.min(data, axis = 0))
    return low + (ratio * (data-np.min(data, axis = 0)))


# In[9]:

class rnn_batch_iterator(object):
    """
    RNN batch iterator - holds the data, and returns the next batch of features, targets, seq_lengths when 
    next_batch() is called
    masks: option to return masks as well when next_batch() called - used for masking out val data during training
    """
    
    def __init__(self, features_data, targets_data, seq_lengths, masks = None):

        self.features_data = features_data
        self.targets_data = targets_data
        self.seq_lengths = seq_lengths
        if masks:
            self.masks = masks
        else:
            self.masks = None
                
        self.counter = 0
        self.num_batches = len(targets_data)
        
        self.batch_size = targets_data[0].shape[0]
        self.targets_dim = targets_data[0].shape[2]
        self.features_dim = features_data[0].shape[2]

    def next_batch(self):
        
        features_batch = self.features_data[self.counter]
        targets_batch = self.targets_data[self.counter]
        seq_length = self.seq_lengths[self.counter]
        if self.masks:
            mask_batch = self.masks[self.counter]
        
        self.counter += 1
        
        if self.counter == self.num_batches:
            self.counter = 0
        
        if self.masks:
            return features_batch, targets_batch, seq_length, mask_batch
        else:
            return features_batch, targets_batch, seq_length
        
    def all_targets(self):
        return np.squeeze(np.concatenate(self.targets_data, axis = 1), axis = 0)


# In[10]:

def prepare_rnn_data(features_data, targets_data, data_set, min_batch_size, num_steps, silent = False):
    """
    Sorts RNN data properly (i.e. maximises the number of states which are passed on correctly)
    
    features_data: shape [size_data * num_features]
    targets_data: shape [size_data * num_targets]
    data_set: one of ['train', 'val', 'test', 'train_inference']
    train_batch_size: batch_size for training data (for val and test batch_size will be one)
    num_steps: number of steps RNN rolled out for during training
    """
    
    assert targets_data.shape[0] == features_data.shape[0], 'Targets and Features data different sizes'
    assert len(targets_data.shape) == 2, 'Targets data needs to be at least 2D'
    
    num_data_points = features_data.shape[0]
    num_features = features_data.shape[1]
    num_targets = targets_data.shape[1]
        
    if data_set == 'train':
         # Calculate the batch size - basically keeping the same number of batches as would be required by batch_size,
        # but increasing the size to avoid wasting data
        batches_required = int(math.floor(num_data_points/float(min_batch_size * num_steps)))
        b_length = int(math.floor(num_data_points/float(batches_required * num_steps)))
        num_zeros = 0
        
    if data_set in ['val', 'test', 'train_inference']:
        
        b_length = 1

        if num_data_points % (num_steps*b_length) == 0:
            num_zeros = 0
        else:
            num_zeros = int((num_steps*b_length) - (num_data_points % (num_steps*b_length)))
    

    def sort_batches(data):
        # Use NaN's at this point, so no mistakes if any of features or targets are actually 0
        if data_set in ['val', 'test', 'train_inference']:
            filler = np.empty(shape = [num_zeros, data.shape[1]])
            filler[:] = np.NAN
            data = np.concatenate([data, filler], axis = 0)
        num_batches = int(len(data)/(b_length*num_steps))
        
        batches = []
        seq_lengths = []
        
        for i in range(num_batches):
            input_matrix = np.zeros((b_length, num_steps, data.shape[1]))
            for j in range(b_length):
                row_start_index = (0 + (num_steps*i)) + (j*num_batches*num_steps)
                input_matrix[j] = data[row_start_index:(row_start_index + num_steps)]
            
            # Find where the nan's are to calculate sequence length
            seq_len = np.ones([b_length]) * num_steps
            seq_len[-1] = num_steps - np.sum(np.isnan(input_matrix[-1,:,0]))
            seq_lengths.append(seq_len)
            
            # Replace nan's with zeros
            input_matrix[np.isnan(input_matrix)] = 0
            batches.append(input_matrix)
            
            #batch_sizes.append(input_matrix.shape[0] * int(seq_len[-1]))
            #batches.append([np.squeeze(i, axis = 1) for i in np.split(input_matrix, num_steps, axis = 1)])
        return batches, seq_lengths
    
    features_batches,  seq_lengths = sort_batches(features_data)
    targets_batches, _ = sort_batches(targets_data)
        
    if silent == False:
        print "{} {} feature batches created, of shape {} [batch_length * num_steps * num_features]".format(len(features_batches), data_set, 
                                                                                                features_batches[0].shape)
        print "{} {} target batches created, in of shape {} [batch_length * num_steps * num_targets]\n" .format(len(targets_batches), data_set, 
                                                                                                targets_batches[0].shape) 
    
    return features_batches, targets_batches, seq_lengths


# In[1]:

def single_blog_graph():
    fig = plt.figure(figsize = [6,4])
    ax = plt.axes()    
    mpl.rc('axes', labelsize = 12)
    mpl.rc('figure', titlesize = 14)
    return fig, ax

colours = {'orange': '#f78500', 'yellow': '#fed16c', 'green': '#139277', 'blue': '#0072df',
               'dark_blue': '#001e78', 'pink': '#fd6d77'}


# In[ ]:

def lag_creator(time_series, num_lags):
    """
    Takes a time series (np array of shape (num_obs,)) and creates lags for it, returning the targets and features
    separately as numpy arrays
    """
    
    if len(time_series.shape) == 2:
        time_series = np.squeeze(time_series)
    
    features = np.zeros(shape = [len(time_series), num_lags])
    
    for num,obs in enumerate(time_series):
        if (num+1) <= num_lags:
            continue
        else:
            features[num, :] = time_series[num - num_lags:num][::-1]
    
    features = features[num_lags:,:]
    targets = np.expand_dims(time_series[num_lags:], 1)
    
    return targets, features


# In[ ]:

class monitor_gpu(object):
    """
    class to facilitate monitoring of gpu utilization during training. Call start_monitoring() when training starts
    and stop_monitoring() when it has finished.
    """
    def __init__(self):
        
        self.command = 'nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1 -f ./temp_gpu_log.csv'
    
    def start_monitoring(self):
        self.p = pexpect.spawn(self.command)
        
    def stop_monitoring(self):
        self.p.sendcontrol('c')
        time.sleep(1)
        df = pd.read_csv('temp_gpu_log.csv', sep = ' ')
        self.usage = df['utilization.gpu'].iloc[2:-1]
        self.average_use = np.average(self.usage)
        
        os.remove('temp_gpu_log.csv')


# In[11]:

def rnn_batch_sorter(features_data, targets_data, batch_size = 20, train_ratio = 0.7,
                 val_ratio = 0.15, test_ratio = 0.15, num_steps = 5):

            
    assert train_ratio + val_ratio + test_ratio == 1, 'Percentages don\'t add up for the data sets'
    assert len(targets_data) == len(features_data), 'Targets and features data different sizes'
    assert len(targets_data.shape) == 2, 'Targets data needs to be at least 2D'
        
    # Split the data into train, val and test sets
    train_size = int(math.ceil(train_ratio * len(targets_data)))
    test_size = int(math.ceil(test_ratio * len(targets_data)))
    val_size = int(len(targets_data) - train_size - test_size)
    
    train_f_data = features_data[0: train_size, :]
    train_t_data = targets_data[0: train_size, :]

    val_f_data = features_data[train_size: train_size + val_size, :]
    val_t_data = targets_data[train_size: train_size + val_size, :]

    test_t_data = targets_data[train_size + val_size :, :]
    test_f_data = features_data[train_size + val_size :, :]
    
    print "Train data: {} observations".format(train_t_data.shape[0])
    print "Val data: {} observations".format(val_t_data.shape[0])
    print "Test data: {} observations\n".format(test_t_data.shape[0])

    # Sort for RNN
    train_f, train_t, train_seq_lengths = prepare_rnn_data(train_f_data, train_t_data, data_set = 'train', 
                                                     min_batch_size = batch_size, num_steps = num_steps)
    val_f, val_t, val_seq_lengths = prepare_rnn_data(val_f_data, val_t_data, data_set = 'val', 
                                                         min_batch_size = batch_size, num_steps = num_steps)
    test_f, test_t, test_seq_lengths = prepare_rnn_data(test_f_data, test_t_data, data_set = 'test', 
                                                         min_batch_size = batch_size, num_steps = num_steps)
    trinf_f, trinf_t, trinf_seq_lengths = prepare_rnn_data(train_f_data, train_t_data, data_set = 'train_inference', 
                                                         min_batch_size = batch_size, num_steps = num_steps)

    # Create the iterators
    train_iter = rnn_batch_iterator(train_f, train_t, train_seq_lengths)
    val_iter = rnn_batch_iterator(val_f, val_t, val_seq_lengths)
    test_iter = rnn_batch_iterator(test_f, test_t, test_seq_lengths)
    tr_inf_iter = rnn_batch_iterator(trinf_f, trinf_t, trinf_seq_lengths)

    iter_dict = {'train': train_iter, 'val': val_iter, 'test': test_iter, 'tr_inf': tr_inf_iter}
    
    return iter_dict


# In[ ]:




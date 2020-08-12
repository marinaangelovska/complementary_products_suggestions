#hyperparameters
feature_dim = 300
regularizer = 0.01
activation='relu'
optimizer = 'adam'
padding = 'valid'
dropout_rate = 0.01
nb_epochs = 10
batch_size = 256
stop_epochs = 5
nb_neurons_dense = 100

#we use this only for lstm
nb_neurons_lstm = 150

# only used for cnn
nb_filter = 100
filter1_length = 3
filter2_length = 2
pool1_length = 5
pool2_length = 3

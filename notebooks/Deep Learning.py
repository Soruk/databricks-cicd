# Databricks notebook source
# MAGIC  %sh
# MAGIC  rm -r /dbfs/deepml_lab
# MAGIC  mkdir /dbfs/deepml_lab
# MAGIC  wget -O /dbfs/deepml_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split
   
# Load the data, removing any incomplete rows
df = spark.read.format("csv").option("header", "true").load("/deepml_lab/penguins.csv").dropna()
   
# Encode the Island with a simple integer index
# Scale FlipperLength and BodyMass so they're on a similar scale to the bill measurements
islands = df.select(collect_set("Island").alias('Islands')).first()['Islands']
island_indexes = [(islands[i], i) for i in range(0, len(islands))]
df_indexes = spark.createDataFrame(island_indexes).toDF('Island', 'IslandIdx')
data = df.join(df_indexes, ['Island'], 'left').select(col("IslandIdx"),
                   col("CulmenLength").astype("float"),
                   col("CulmenDepth").astype("float"),
                   (col("FlipperLength").astype("float")/10).alias("FlipperScaled"),
                    (col("BodyMass").astype("float")/100).alias("MassScaled"),
                   col("Species").astype("int")
                    )
   
# Oversample the dataframe to triple its size
# (Deep learning techniques like LOTS of data)
for i in range(1,3):
    data = data.union(data)
   
# Split the data into training and testing datasets   
features = ['IslandIdx','CulmenLength','CulmenDepth','FlipperScaled','MassScaled']
label = 'Species'
      
# Split data 70%-30% into training set and test set
x_train, x_test, y_train, y_test = train_test_split(data.toPandas()[features].values,
                                                    data.toPandas()[label].values,
                                                    test_size=0.30,
                                                    random_state=0)
   
print ('Training Set: %d rows, Test Set: %d rows \n' % (len(x_train), len(x_test)))

# COMMAND ----------

import torch
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
   
# Set random seed for reproducability
torch.manual_seed(0)
   
print("Libraries imported - ready to use PyTorch", torch.__version__)

# COMMAND ----------

# Create a dataset and loader for the training data and labels
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x,train_y)
train_loader = td.DataLoader(train_ds, batch_size=20,
    shuffle=False, num_workers=1)

# Create a dataset and loader for the test data and labels
test_x = torch.Tensor(x_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = td.TensorDataset(test_x,test_y)
test_loader = td.DataLoader(test_ds, batch_size=20,
                             shuffle=False, num_workers=1)
print('Ready to load data')

# COMMAND ----------

# Number of hidden layer nodes
hl = 10
   
# Define the neural network
class PenguinNet(nn.Module):
    def __init__(self):
        super(PenguinNet, self).__init__()
        self.fc1 = nn.Linear(len(features), hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, 3)
   
    def forward(self, x):
        fc1_output = torch.relu(self.fc1(x))
        fc2_output = torch.relu(self.fc2(fc1_output))
        y = F.log_softmax(self.fc3(fc2_output).float(), dim=1)
        return y
   
# Create a model instance from the network
model = PenguinNet()
print(model)

# COMMAND ----------

def train(model, data_loader, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Set the model to training mode
    model.train()
    train_loss = 0
       
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        #feedforward
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()
   
        # backpropagate adjustments to the weights
        loss.backward()
        optimizer.step()
   
    #Return average loss
    avg_loss = train_loss / (batch+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
              
               
def test(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Switch the model to evaluation mode (so we don't backpropagate)
    model.eval()
    test_loss = 0
    correct = 0
   
    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor
            # Get the predictions
            out = model(data)
   
            # calculate the loss
            test_loss += loss_criteria(out, target).item()
   
            # Calculate the accuracy
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target==predicted).item()
               
    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
       
    # return average loss for the epoch
    return avg_loss

# COMMAND ----------

# Specify the loss criteria (we'll use CrossEntropyLoss for multi-class classification)
loss_criteria = nn.CrossEntropyLoss()
   
# Use an optimizer to adjust weights and reduce loss
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()
   
# We'll track metrics for each epoch in these arrays
epoch_nums = []
training_loss = []
validation_loss = []
   
# Train over 100 epochs
epochs = 100
for epoch in range(1, epochs + 1):
   
    # print the epoch number
    print('Epoch: {}'.format(epoch))
       
    # Feed training data into the model
    train_loss = train(model, train_loader, optimizer)
       
    # Feed the test data into the model to check its performance
    test_loss = test(model, test_loader)
       
    # Log the metrics for this epoch
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from matplotlib import pyplot as plt
# MAGIC    
# MAGIC plt.plot(epoch_nums, training_loss)
# MAGIC plt.plot(epoch_nums, validation_loss)
# MAGIC plt.xlabel('epoch')
# MAGIC plt.ylabel('loss')
# MAGIC plt.legend(['training', 'validation'], loc='upper right')
# MAGIC plt.show()

# COMMAND ----------

for param_tensor in model.state_dict():
    print(param_tensor, "\n", model.state_dict()[param_tensor].numpy())

# COMMAND ----------

# Save the model weights
model_file = '/dbfs/penguin_classifier.pt'
torch.save(model.state_dict(), model_file)
del model
print('model saved as', model_file)

# COMMAND ----------

# New penguin features
x_new = [[1, 50.4,15.3,20,50]]
print ('New sample: {}'.format(x_new))
   
# Create a new model class and load weights
model = PenguinNet()
model.load_state_dict(torch.load(model_file))
   
# Set model to evaluation mode
model.eval()
   
# Get a prediction for the new data sample
x = torch.Tensor(x_new).float()
_, predicted = torch.max(model(x).data, 1)
   
print('Prediction:',predicted.item())

# COMMAND ----------

import horovod.torch as hvd
from sparkdl import HorovodRunner
   
def train_hvd(model):
    from torch.utils.data.distributed import DistributedSampler
       
    hvd.init()
       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Pin GPU to local rank
        torch.cuda.set_device(hvd.local_rank())
       
    # Configure the sampler so that each worker gets a distinct sample of the input dataset
    train_sampler = DistributedSampler(train_ds, num_replicas=hvd.size(), rank=hvd.rank())
    # Use train_sampler to load a different sample of data on each worker
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=20, sampler=train_sampler)
       
    # The effective batch size in synchronous distributed training is scaled by the number of workers
    # Increase learning_rate to compensate for the increased batch size
    learning_rate = 0.001 * hvd.size()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       
    # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
   
    # Broadcast initial parameters so all workers start with the same parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
   
    optimizer.zero_grad()
   
    # Train over 50 epochs
    epochs = 100
    for epoch in range(1, epochs + 1):
        print('Epoch: {}'.format(epoch))
        # Feed training data into the model to optimize the weights
        train_loss = train(model, train_loader, optimizer)
   
    # Save the model weights
    if hvd.rank() == 0:
        model_file = '/dbfs/penguin_classifier_hvd.pt'
        torch.save(model.state_dict(), model_file)
        print('model saved as', model_file)

# COMMAND ----------

# Reset random seed for PyTorch
torch.manual_seed(0)
   
# Create a new model
new_model = PenguinNet()
   
# We'll use CrossEntropyLoss to optimize a multiclass classifier
loss_criteria = nn.CrossEntropyLoss()
   
# Run the distributed training function on 2 nodes
hr = HorovodRunner(np=2, driver_log_verbosity='all') 
hr.run(train_hvd, model=new_model)
   
# Load the trained weights and test the model
test_model = PenguinNet()
test_model.load_state_dict(torch.load('/dbfs/penguin_classifier_hvd.pt'))
test_loss = test(test_model, test_loader)

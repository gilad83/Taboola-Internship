# Taboola-Intership

Description

### Supervised model (batch data):
We believe that the action conversion rate is correlated with the traffic and SLA in a certain data center. Thus, we would like to predict the action conversion rate based on the data center traffic rate and response time. 
You are going to build a supervised learning LSTM model that will get as input several metrics in a CSV format, such as p95, 5min rate and will try to predict the total success and failed action conversion rate. 

example for parameters:

 
* -train_size 0.8 
* -epochs 50
* -batch_size 72
* -n_nodes 50
* -time_steps 7*288

### Unsupervised model:
We would like to have a model that can predict a data center status based on several metrics given as input.
You are going to build an unsupervised LSTM autoencoder model that gets several metrics as CSV files, and predicts their behaviour in the future. Then, you will add weights to the predicted metrics, and calculate a data center score for a future period of time.

* -path 
* -train_size 0.8
* -epochs 50
* -batch_size 72
* -time_steps 7*288

"""
Training for ZipperNet
"""
import sys

import numpy as np
import torch
import torch.nn as nn

def train_zipper(zipper, train_dataloader, train_dataset, test_dataset, validation_size=None, monitor=False, outfile_prefix=""):
    
    zipper.train()
    
    number_of_training_epochs = 20
    if validation_size is None:
        validation_size=len(test_dataset)
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(zipper.parameters(), lr=learning_rate)
    
    losses, train_acc, validation_acc = [], [], []
    
    # Track best validation acc
    best_val_acc = 0.0

    for epoch in range(number_of_training_epochs):
        sys.stdout.write("\rEpoch {0}\r".format(epoch + 1))
        sys.stdout.flush()

        for i_batch, sample_batched in enumerate(train_dataloader):

            #Clear out all existing gradients on the loss surface to reevaluate for this step
            optimizer.zero_grad()

            #Get the CNN's current prediction of the training data
            output = zipper(sample_batched['lightcurve'], sample_batched['image'])

            #Calculate the loss by comparing the prediction to the truth
            loss = loss_function(output, sample_batched['label']) 

            #Evaluate all gradients along the loss surface using back propagation
            loss.backward()

            #Based on the gradients, take the optimal step in the weight space
            optimizer.step()

            #Performance monitoring if desired
            if monitor:
                if i_batch % 500 == 0:
                    train_output = zipper(train_dataset[0:validation_size]['lightcurve'], train_dataset[0:validation_size]['image'])
                    validation_output = zipper(test_dataset[0:validation_size]['lightcurve'], test_dataset[0:validation_size]['image'])

                    train_predictions = torch.max(train_output, 1)[1].data.numpy()
                    validation_predictions = torch.max(validation_output, 1)[1].data.numpy()

                    train_accuracy = np.sum(train_predictions == train_dataset[0:validation_size]['label'].numpy()) / validation_size
                    validation_accuracy = np.sum(validation_predictions == test_dataset[0:validation_size]['label'].numpy()) / validation_size

                    print("Epoch: {0} Batch: {1}  | Training Accuracy: {2:.3f} -- Validation Accuracy: {3:.3f} -- Loss: {4:.3f}".format(epoch + 1, i_batch + 1, train_accuracy, validation_accuracy, loss.data.numpy()))

                    losses.append(loss.data.numpy())
                    train_acc.append(train_accuracy)
                    validation_acc.append(validation_accuracy)
                    
                    # save best network
                    if validation_accuracy > best_val_acc:
                        torch.save(zipper.state_dict(), f"{outfile_prefix}_network.pt")
                        best_val_acc = validation_accuracy
   
    setattr(zipper, 'losses', losses)
    setattr(zipper, 'train_acc', train_acc)
    setattr(zipper, 'validation_acc', validation_acc)

    return zipper


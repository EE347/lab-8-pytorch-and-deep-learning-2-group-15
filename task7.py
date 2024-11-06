import time
import cv2
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
import torchvision.transforms.functional as TF  # For data augmentation

# Import necessary modules for confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns  # For plotting the confusion matrix

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    
    # Adjusted batch sizes
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=4, shuffle=False)

    # List to store all results
    all_results = []

    # Define the loss functions to use
    loss_functions = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
        'NLLLoss': torch.nn.NLLLoss()
    }

    # Adjusted learning rate
    learning_rate = 0.001

    # Number of epochs set to 3
    num_epochs = 3

    # Open the CSV file and write the headers
    with open('training_results.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Epoch', 'Loss Function', 'Train Loss', 'Test Loss', 'Test Accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over loss functions
        for loss_name, criterion in loss_functions.items():
            # Reset the model for each loss function
            model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Saving parameters
            best_train_loss = float('inf')

            # Loss lists
            train_losses = []
            test_losses = []

            # Epoch Loop (now runs for 3 epochs)
            for epoch in range(1, num_epochs + 1):

                # Start timer
                t = time.time_ns()

                # Train the model
                model.train()
                train_loss = 0

                # Batch Loop
                for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

                    # Move the data to the device (CPU or GPU)
                    images = images.reshape(-1, 3, 64, 64)
                    labels = labels.to(device)

                    # Data Augmentation: Random Horizontal Flip
                    flip_prob = 0.5
                    flip_mask = torch.rand(images.size(0)) < flip_prob
                    if flip_mask.any():
                        images[flip_mask] = torch.flip(images[flip_mask], dims=[3])

                    # Data Augmentation: Random Rotation (-10 to +10 degrees)
                    angles = torch.empty(images.size(0)).uniform_(-10, 10)
                    for idx, angle in enumerate(angles):
                        images[idx] = TF.rotate(images[idx], angle.item(), interpolation=TF.InterpolationMode.BILINEAR)

                    # Move images to device after augmentation
                    images = images.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(images)

                    # Apply LogSoftmax if using NLLLoss
                    if loss_name == 'NLLLoss':
                        outputs = F.log_softmax(outputs, dim=1)

                    # Compute the loss
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Update the parameters
                    optimizer.step()

                    # Accumulate the loss
                    train_loss += loss.item()

                # Test the model
                model.eval()
                test_loss = 0
                correct = 0
                total = 0

                # For confusion matrix
                all_preds = []
                all_labels = []

                # Batch Loop
                with torch.no_grad():
                    for images, labels in tqdm(testloader, total=len(testloader), leave=False):

                        # Move the data to the device (CPU or GPU)
                        images = images.reshape(-1, 3, 64, 64).to(device)
                        labels = labels.to(device)

                        # Forward pass
                        outputs = model(images)

                        # Apply LogSoftmax if using NLLLoss
                        if loss_name == 'NLLLoss':
                            outputs = F.log_softmax(outputs, dim=1)

                        # Compute the loss
                        loss = criterion(outputs, labels)

                        # Accumulate the loss
                        test_loss += loss.item()

                        # Get the predicted class
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)

                        # Accumulate the number of correct classifications
                        correct += (predicted == labels).sum().item()

                        # Store predictions and true labels for confusion matrix
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                # Calculate average losses and accuracy
                avg_train_loss = train_loss / len(trainloader)
                avg_test_loss = test_loss / len(testloader)
                accuracy = correct / total

                # Print the epoch statistics
                print(f'Epoch: {epoch}, Loss Function: {loss_name}, Train Loss: {avg_train_loss:.4f}, '
                      f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}, '
                      f'Time: {(time.time_ns() - t) / 1e9:.2f}s')

                # Update loss lists
                train_losses.append(avg_train_loss)
                test_losses.append(avg_test_loss)

                # Write results to CSV
                writer.writerow({
                    'Epoch': epoch,
                    'Loss Function': loss_name,
                    'Train Loss': avg_train_loss,
                    'Test Loss': avg_test_loss,
                    'Test Accuracy': accuracy
                })

                # Update the best model
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    torch.save(model.state_dict(), f'7best_model_{loss_name}.pth')

                # Save the current model
                torch.save(model.state_dict(), f'7current_model_{loss_name}.pth')

                # Create the loss plot
                plt.figure()
                plt.plot(train_losses, label='Train Loss')
                plt.plot(test_losses, label='Test Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'Loss Curve ({loss_name})')
                plt.savefig(f'7loss_plot_{loss_name}.png')
                plt.close()

                # Compute confusion matrix
                cm = confusion_matrix(all_labels, all_preds)

                # Plot confusion matrix
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                plt.title(f'Confusion Matrix (Epoch {epoch}, {loss_name})')
                plt.savefig('7confusion_matrix.png')  # Overwrite the image each epoch
                plt.close()

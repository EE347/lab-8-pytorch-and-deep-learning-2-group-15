import time
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

    # Determine the actual number of images per class
    n_images_train = 40  # Adjust based on your dataset
    n_images_test = 10   # Adjust based on your dataset

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=n_images_train, train=True)
    testset = TeamMateDataset(n_images=n_images_test, train=False)

    # Adjusted batch sizes
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
    testloader = DataLoader(testset, batch_size=4, shuffle=False)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001

    # Initialize the model
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Parameters for training
    num_epochs = 3  # Adjusted number of epochs
    best_val_loss = float('inf')

    # Lists to store losses and accuracy
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Open the CSV file and write the headers
    with open('training_results.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Epoch Loop
        for epoch in range(1, num_epochs + 1):

            # Start timer
            start_time = time.time()

            # Train the model
            model.train()
            running_train_loss = 0.0

            # Batch Loop
            for images, labels in tqdm(trainloader, total=len(trainloader), leave=False):

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

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                running_train_loss += loss.item() * images.size(0)

            # Calculate average training loss
            avg_train_loss = running_train_loss / len(trainset)
            train_losses.append(avg_train_loss)

            # Validate the model
            model.eval()
            running_val_loss = 0.0
            correct = 0
            total = 0

            # For confusion matrix
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in tqdm(testloader, total=len(testloader), leave=False):

                    # Move the data to the device (CPU or GPU)
                    images = images.reshape(-1, 3, 64, 64).to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)

                    # Compute the loss
                    loss = criterion(outputs, labels)

                    # Accumulate the loss
                    running_val_loss += loss.item() * images.size(0)

                    # Get the predicted class
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)

                    # Accumulate the number of correct classifications
                    correct += (predicted == labels).sum().item()

                    # Store predictions and true labels for confusion matrix
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate average validation loss and accuracy
            avg_val_loss = running_val_loss / len(testset)
            val_losses.append(avg_val_loss)
            accuracy = correct / total
            val_accuracies.append(accuracy)

            # Print the epoch statistics
            elapsed_time = time.time() - start_time
            print(f'Epoch: {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
                  f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}, '
                  f'Time: {elapsed_time:.2f}s')

            # Write results to CSV
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Validation Accuracy': accuracy
            })

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), '8best_model.pth')

            # Save the current model
            torch.save(model.state_dict(), '8current_model.pth')

            # Create the loss and accuracy plots
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, epoch + 1), train_losses, label='Train Loss')
            plt.plot(range(1, epoch + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curve')

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epoch + 1), val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy Curve')

            plt.tight_layout()
            plt.savefig('8training_progress.png')
            plt.close()

            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix (Epoch {epoch})')
            plt.savefig('8confusion_matrix.png')  # Overwrite the image each epoch
            plt.close()

    # Training complete, save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print('Training complete. Best model saved as "best_model.pth".')

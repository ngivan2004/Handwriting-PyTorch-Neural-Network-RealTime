import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import NNPy, NNPyCNN

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define transformations
standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

augmented_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomAffine(degrees=30, translate=(
        0.2, 0.2), scale=(0.25, 1), shear=(-30, 30, -30, 30))
])

# Load test datasets
standard_test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=standard_transform)
augmented_test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=augmented_transform)

# Dataloaders
standard_test_loader = DataLoader(
    standard_test_data, batch_size=1, shuffle=False)
augmented_test_loader = DataLoader(
    augmented_test_data, batch_size=1, shuffle=False)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Function to evaluate the model


def evaluate_model(model, dataloader):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct / size
    return size, accuracy


def main():
    # Load models
    nnpy_model = NNPy().to(device)
    nnpy_model.load_state_dict(torch.load('model_state.pth'))

    nnpycnn_model = NNPyCNN().to(device)
    nnpycnn_model.load_state_dict(torch.load('model_state_augmented+cnn.pth'))

    # Evaluate NNPy on standard and augmented test data
    nnpy_standard_size, nnpy_standard_accuracy = evaluate_model(
        nnpy_model, standard_test_loader)
    nnpy_augmented_size, nnpy_augmented_accuracy = evaluate_model(
        nnpy_model, augmented_test_loader)

    # Evaluate NNPyCNN on standard and augmented test data
    nnpycnn_standard_size, nnpycnn_standard_accuracy = evaluate_model(
        nnpycnn_model, standard_test_loader)
    nnpycnn_augmented_size, nnpycnn_augmented_accuracy = evaluate_model(
        nnpycnn_model, augmented_test_loader)

    # Print results
    print(
        f"NNPy Performance on Standard MNIST Test Data: Tested {nnpy_standard_size} inputs, Accuracy: {nnpy_standard_accuracy*100:.2f}%")
    print(
        f"NNPy Performance on Augmented MNIST Test Data: Tested {nnpy_augmented_size} inputs, Accuracy: {nnpy_augmented_accuracy*100:.2f}%")
    print(
        f"NNPyCNN Performance on Standard MNIST Test Data: Tested {nnpycnn_standard_size} inputs, Accuracy: {nnpycnn_standard_accuracy*100:.2f}%")
    print(
        f"NNPyCNN Performance on Augmented MNIST Test Data: Tested {nnpycnn_augmented_size} inputs, Accuracy: {nnpycnn_augmented_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()

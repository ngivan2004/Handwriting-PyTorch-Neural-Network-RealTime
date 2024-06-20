import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

from model import NN

learning_rate = 1e-3
batch_size = 64
epochs = 10


# Possibly handy function to use in the future
def one_hot_encode(y):
    # Create a zero tensor of shape (num_classes,)
    one_hot = torch.zeros(10, dtype=torch.float)
    # Scatter the value 1 at the index specified by y
    one_hot.scatter_(dim=0, index=torch.tensor(y), value=1)
    return one_hot


# loading the datasets with both training and testing
training_data = datasets.MNIST(root="data", train=True, download=True,
                               transform=ToTensor())
testing_data = datasets.MNIST(root="data", train=False, download=True,
                              transform=ToTensor())


# dataloaders
training_dataloader = DataLoader(training_data, batch_size=batch_size)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size)


def training_loop(dataloader, model, loss_fn, optimizer):
    # size is the total number of samples in the dataset. This helps us calculate the progress later.
    size = len(dataloader.dataset)

    # training mode (layer dropout, batch normalization)
    model.train()

    for batch, (one_batch_of_input_data, target_label) in enumerate(dataloader):
        prediction = model(one_batch_of_input_data)
        loss = loss_fn(prediction, target_label)

        # does back prop
        loss.backward()

        # Updates the parameters
        optimizer.step()

        # clears gradients to not accumulate.
        optimizer.zero_grad()

        # For ever 100 matches, print progress
        if batch % 100 == 0:
            # loss.item() returns value of loss.
            loss = loss.item()
            # current number of samples processed so far
            current = batch * batch_size + len(one_batch_of_input_data)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def testing_loop(dataloader, model, loss_fn):

    # eval mode
    model.eval()
    size = len(dataloader.dataset)
    batch_num = len(dataloader)
    loss = 0
    correct = 0

    # no grad is used because we want to ensure no gradients will be computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            prediction = model(X)
            loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) ==
                        y).type(torch.float).sum().item()

    loss /= batch_num
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")


def main():
    model = NN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loop(training_dataloader, model, loss_fn, optimizer)
        testing_loop(testing_dataloader, model, loss_fn)

    torch.save(model.state_dict(), 'model_state.pth')

    print("Done!")


if __name__ == "__main__":
    main()

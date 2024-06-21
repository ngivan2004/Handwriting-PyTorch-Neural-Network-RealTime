import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import matplotlib.pyplot as plt

from model import NNPyCNN


learning_rate = 1e-3
batch_size = 128
epochs = 30

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Possibly handy function to use in the future


def one_hot_encode(y):
    # Create a zero tensor of shape (num_classes,)
    one_hot = torch.zeros(10, dtype=torch.float)
    # Scatter the value 1 at the index specified by y
    one_hot.scatter_(dim=0, index=torch.tensor(y), value=1)
    return one_hot


transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomAffine(degrees=30,
                            translate=(0.2, 0.2),
                            scale=(0.25, 1),
                            shear=(-30, 30, -30, 30))
])

# loading the datasets with both training and testing
training_data = datasets.MNIST(root="data", train=True, download=True,
                               transform=transformations)
testing_data = datasets.MNIST(root="data", train=False, download=True,
                              transform=transformations)


# dataloaders
training_dataloader = DataLoader(training_data, batch_size=batch_size)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size)


""" figure = plt.figure(figsize=(8, 8))

sample_idx = torch.randint(len(training_data), size=(1,)).item()
img, label = training_data[sample_idx]

# Add a single subplot
ax = figure.add_subplot(1, 1, 1)
plt.title(label)
plt.axis("off")
plt.imshow(img.squeeze(), cmap="gray")

# Overlay pixel values
img_data = img.squeeze().numpy()
for y in range(img_data.shape[0]):
    for x in range(img_data.shape[1]):
        pixel_value = img_data[y, x]
        ax.text(x, y, f'{pixel_value:.1f}', color='red',
                fontsize=6, ha='center', va='center')

# Show the figure
plt.show()
 """


def training_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (one_batch_of_input_data, target_label) in enumerate(dataloader):
        one_batch_of_input_data, target_label = one_batch_of_input_data.to(
            device), target_label.to(device)
        prediction = model(one_batch_of_input_data)
        loss = loss_fn(prediction, target_label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(one_batch_of_input_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def testing_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    batch_num = len(dataloader)
    loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            loss += loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) ==
                        y).type(torch.float).sum().item()

    loss /= batch_num
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")


def main():
    model = NNPyCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loop(training_dataloader, model, loss_fn, optimizer)
        testing_loop(testing_dataloader, model, loss_fn)

    torch.save(model.state_dict(), 'model_state_augmented+cnn.pth')

    print("Done!")


if __name__ == "__main__":
    main()

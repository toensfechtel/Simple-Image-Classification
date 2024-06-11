import torch, torchvision
import torchvision.transforms as transforms
from PIL import Image
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, TRAIN_DIR, TEST_DIR
from pytorch_lightning.callbacks import EarlyStopping

# Define the CNN architecture
Net = lambda c1o=4, c2o=8, f1o=64, f2o=32, f3o=4: torch.nn.Sequential(
    torch.nn.Conv2d(1, c1o, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(c1o, c2o, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(), torch.nn.Linear(c2o * 53 * 53, f1o), torch.nn.ReLU(),
    torch.nn.Linear(f1o, f2o), torch.nn.ReLU(), torch.nn.Linear(f2o, f3o))


def main():

    # Define the data transformation pipeline
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Load the datasets using ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    for epoch in range(NUM_EPOCHS):
        running_loss = sum(
            criterion(net(inputs), labels).backward().item()
            for inputs, labels in train_loader)
        print(
            f'Epoch {epoch+1}, Mean Train Loss per Batch: {running_loss / len(train_loader):.3f}'
        )

        # Evaluate the model on the validation data
        val_loss = sum(
            criterion(net(images), labels).item()
            for images, labels in test_loader) / len(test_loader)
        print(f'Epoch {epoch+1}, Mean Val Loss per Batch: {val_loss:.3f}')

        # Apply early stopping based on the validation loss
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    # Evaluate the final model on the test data
    correct = sum(
        torch.max(net(images), 1)[1] == labels
        for images, labels in test_loader)
    total = len(test_dataset)
    print(
        f'Accuracy of the network on the test images: {100 * correct / total:.2f}%'
    )


if __name__ == '__main__':
    main()

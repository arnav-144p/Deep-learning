from prefect import task, flow
import torch, torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

@task
def dataLoader():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train = torchvision.datasets.OxfordIIITPet('./data', split='trainval', transform=train_transform, download=True)
    test = torchvision.datasets.OxfordIIITPet('./data', split='test', transform=test_transform, download=True)

    trainLoader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print(len(train))
    print("Task 1: Data Loading")
    return trainLoader, testLoader


@task
def build_model():
    model = resnet18(weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 37)

    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)

    print("Resnet Ready")
    return model


@task(retries=2, retry_delay_seconds=2)
def train_model(model, trainLoader):
    model.train()

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        running_loss = 0

        for img, labels in trainLoader:
            img = img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} running loss is {running_loss}")

    return model


@task
def test_model(model, testLoader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for img, labels in testLoader:
            img = img.to(device)
            labels = labels.to(device)

            output = model(img)
            _, pred = torch.max(output, 1)

            correct += (pred == labels).sum().item()
            total += labels.shape[0]

    acc = (correct / total) * 100
    print("Accuracy", acc)


@task
def save_model(model):
    torch.save(model.state_dict(), "/kaggle/working/pet_resnet18_finetuned.pth")
    print("Saved model")


@flow
def pet_pipeline():
    trainLoader, testLoader = dataLoader()

    model = build_model()

    model = train_model(model, trainLoader)

    test_model(model, testLoader)

    save_model(model)


if __name__ == "__main__":
    pet_pipeline()
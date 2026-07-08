from prefect import task, flow
import torch, torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

@task
def dataLoader():
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train = torchvision.datasets.OxfordIIITPet('./data', split='trainval', transform=transform, download=True)
    test = torchvision.datasets.OxfordIIITPet('./data', split='test', transform=transform, download=True)
    
    trainLoader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    testLoader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False)

    print(len(train))
    print("Task 1: Data Loading")
    return trainLoader, testLoader

@task
def build_model():
    model = resnet18(weights="DEFAULT")

    #Freeze old weights
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 37)

    print("Resnet Ready")
    
    return model

@task(retries=2, retry_delay_seconds=2)
def train_model(model, trainLoader):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        running_loss = 0
        for img, labels in trainLoader:
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
    correct = 0
    total = 0
    with torch.no_grad():
        for img, labels in testLoader:
            output = model(img)
            
            _, pred = torch.max(output, 1)

            correct += (pred == labels).sum().item()
            total += labels.shape[0]
        
    acc = (correct/total) * 100
    print("Accuracy", acc)


@task
def save_model(model):
    torch.save(model.state_dict(), "pet_resnet18.pth")

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



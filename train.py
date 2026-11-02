import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(10),  
    transforms.RandomCrop(32, padding=4),  
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define CNN layers
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3,  padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)  
        
        self.dropout1 = nn.Dropout(0.5)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #  feature map + ReLU + Maxpool
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 256 * 2 *2) 

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  

        x = self.fc2(x) 
        return x
    
net = Net()

criterion = nn.CrossEntropyLoss() 

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
optimizer = optim.Adam(net.parameters(),lr=0.001)

net = net.to(device)
criterion = criterion.to(device)

for epoch in range(50): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()


    epoch_loss = running_loss / len(trainloader)  # Average loss per epoch
    print(f'Epoch [{epoch + 1}/50], Loss: {epoch_loss:.4f}')
print('Finished Training')

torch.save(net.state_dict(), 'cifar_net.pth')

correct = 0
total = 0

net.eval()

with torch.no_grad(): 
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Training Accuracy: {100 * correct / total:.2f}%")

net.load_state_dict(torch.load('cifar_net.pth'))

correct = 0
total = 0
with torch.no_grad(): 
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct = [0] * 10  
total = [0] * 10
total_correct = 0
total_samples = 0

net.eval()

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        # Update correct and total counts for each class
        for i in range(len(labels)):
            label = labels[i].item()
            total[label] += 1
            total_samples += 1
            if predicted[i] == label:
                correct[label] += 1
                total_correct += 1

# Calculate the accuracy for each class
class_accuracies = []
for i in range(10):
    accuracy = (correct[i] / total[i]) * 100
    class_accuracies.append((classes[i], accuracy))

class_accuracies.sort(key=lambda x: x[1])

for class_name, accuracy in class_accuracies:
    print(f"Accuracy for class {class_name}: {accuracy:.2f}%")

overall_accuracy = (total_correct / total_samples) * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
from torchvision.models import resnet18

# ---------------- Model Parts (Split ResNet18) ----------------
class ResNetPart1(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.part1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
    def forward(self, x):
        return self.part1(x)

class ResNetPart2(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        self.part2 = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
            resnet.fc  # Linear(512, 1000) → Replace for CIFAR-10
        )
        self.part2[-1] = nn.Linear(512, 10)  # Replace final layer

    def forward(self, x):
        return self.part2(x)

# ---------------- Training ----------------
def train(rank, dataloader):
    print(f"[Rank {rank}] Starting training process...")

    model_part1 = ResNetPart1()
    model_part2 = ResNetPart2()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model_part1.parameters()) + list(model_part2.parameters()), lr=0.001)

    model_part1.train()
    model_part2.train()

    for epoch in range(5):  # Increase this to 20+ for better accuracy
        for images, labels in dataloader:
            out1 = model_part1(images)
            out2 = model_part2(out1)
            loss = criterion(out2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Rank {rank}] Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

    if rank == 0:
        torch.save(model_part1.state_dict(), "resnet_part1.pt")
        torch.save(model_part2.state_dict(), "resnet_part2.pt")
        print("[Rank 0] ResNet parts saved to disk.")

# ---------------- Evaluation ----------------
def evaluate(model_part1, model_part2, test_loader):
    model_part1.eval()
    model_part2.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            out1 = model_part1(images)
            out2 = model_part2(out1)
            _, predicted = torch.max(out2, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"\n✅ Test Accuracy: {acc:.2f}%")

# ---------------- Main ----------------
def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

    part1_len = len(dataset) // 2
    part2_len = len(dataset) - part1_len
    subset1, subset2 = random_split(dataset, [part1_len, part2_len])

    dataloader1 = DataLoader(subset1, batch_size=64, shuffle=True)
    dataloader2 = DataLoader(subset2, batch_size=64, shuffle=True)

    # Parallel processes for data parallelism
    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=train, args=(0, dataloader1))
    p2 = ctx.Process(target=train, args=(1, dataloader2))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print("\n✅ Training done. Now evaluating the combined ResNet model...\n")

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_part1 = ResNetPart1()
    model_part2 = ResNetPart2()
    model_part1.load_state_dict(torch.load("resnet_part1.pt"))
    model_part2.load_state_dict(torch.load("resnet_part2.pt"))

    evaluate(model_part1, model_part2, test_loader)

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

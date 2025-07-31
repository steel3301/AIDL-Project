import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp

# ---------------- Model Parts ----------------
class TinyModelPart1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16
        )
    def forward(self, x):
        return self.block(x)

class TinyModelPart2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.block(x)

# ---------------- Training ----------------
def train(rank, dataloader):
    print(f"[Rank {rank}] Starting training process...")

    model_part1 = TinyModelPart1()
    model_part2 = TinyModelPart2()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model_part1.parameters()) + list(model_part2.parameters()), lr=0.001)

    for epoch in range(3):
        for images, labels in dataloader:
            out1 = model_part1(images)
            out2 = model_part2(out1)
            loss = criterion(out2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Rank {rank}] Epoch [{epoch+1}/3], Loss: {loss.item():.4f}")

    # Save models only from rank 0
    if rank == 0:
        torch.save(model_part1.state_dict(), "model_part1.pt")
        torch.save(model_part2.state_dict(), "model_part2.pt")
        print("[Rank 0] Model saved to disk.")

# ---------------- Evaluation ----------------
def evaluate(model_part1, model_part2, test_loader):
    model_part1.eval()
    model_part2.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            out1 = model_part1(images)
            out2 = model_part2(out1)
            _, predicted = torch.max(out2.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"\n✅ Test Accuracy: {acc:.2f}%")

# ---------------- Main ----------------
def main():
    # Load CIFAR-10 and split it
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    part1_len = len(dataset) // 2
    part2_len = len(dataset) - part1_len
    subset1, subset2 = random_split(dataset, [part1_len, part2_len])

    dataloader1 = DataLoader(subset1, batch_size=64, shuffle=True)
    dataloader2 = DataLoader(subset2, batch_size=64, shuffle=True)

    # Launch 2 parallel training processes
    ctx = mp.get_context("spawn")
    p1 = ctx.Process(target=train, args=(0, dataloader1))
    p2 = ctx.Process(target=train, args=(1, dataloader2))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print("\n✅ Training done. Now evaluating the combined model...\n")

    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load saved models
    model_part1 = TinyModelPart1()
    model_part2 = TinyModelPart2()
    model_part1.load_state_dict(torch.load("model_part1.pt"))
    model_part2.load_state_dict(torch.load("model_part2.pt"))

    evaluate(model_part1, model_part2, test_loader)

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()

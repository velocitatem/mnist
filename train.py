import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SimpleCNN
import os


def train():
    # Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    lr = 0.01
    momentum = 0.5
    save_dir = "checkpoints"

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Initialize model
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
            f"({100.0 * correct / len(test_loader.dataset):.2f}%)\n"
        )

        # Save checkpoint
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"mnist_cnn_epoch_{epoch}.pt")
        )

    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()

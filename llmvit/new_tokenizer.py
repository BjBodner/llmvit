import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.optim as optim

# Initialize tokenizer with special tokens
vision_tokenizer = AutoTokenizer.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    # extra_special_tokens={
    #     "image_token": "<image>",
    #     "boi_token": "<image_start>",
    #     "eoi_token": "<image_end>"
    # }
)

# Custom dataset class to combine MNIST images with text
class MNISTWithCaptions(Dataset):
    def __init__(self, is_train=True):
        self.mnist = datasets.MNIST(
            root="./data",
            train=is_train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.captions = [
            f"This is a handwritten digit {label}" for _, label in self.mnist
        ]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        # Create text description
        # boi_token = vision_tokenizer.special_tokens_map.get('boi_token', '<image_start>')
        # eoi_token = vision_tokenizer.special_tokens_map.get('eoi_token', '<image_end>')
        # caption = f"This is a handwritten digit {label}"
        # caption = f"{vision_tokenizer.special_tokens['boi_token']}This is a handwritten digit {label}{vision_tokenizer.special_tokens['eoi_token']}"

        # Tokenize the caption

        return {
            "image": image,
            "label": label,
            "caption": self.captions[idx],
            # 'input_ids': encoded_caption['input_ids'].squeeze(),
            # 'attention_mask': encoded_caption['attention_mask'].squeeze()
        }


# Create dataloaders
def get_dataloaders(batch_size=32):
    train_dataset = MNISTWithCaptions(is_train=True)
    test_dataset = MNISTWithCaptions(is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        x += self.pos_embed
        x = self.norm(x)
        return x


# Custom model combining vision and language
class VisionLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Vision encoder (simple CNN for MNIST)
        self.vision_encoder = PatchEmbed(embed_dim=512)
        # Language decoder
        self.embedding = nn.Embedding(vocab_size + 2, 512)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.fc = nn.Linear(512, vocab_size + 2)

    def forward(self, images, input_ids, attention_mask):
        # Encode images
        vision_features = self.vision_encoder(images)
        # vision_features = vision_features.unsqueeze(0)  # Add sequence dimension

        # Embed text tokens
        embedded = self.embedding(input_ids)

        # Combine vision and language
        output = self.transformer(
            embedded,
            vision_features,
            # tgt_key_padding_mask=~attention_mask.bool().T
        )

        # Project to vocabulary
        output = self.fc(output)
        return output


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in train_loader:

        encoded_caption = vision_tokenizer(
            batch["caption"],
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt",
        )
        batch = {**batch, **encoded_caption.data}

        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Shift tokens for causal language modeling
        target_ids = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, batch["label"].to(device))

        loss.backward()
        optimizer.step()
        print(loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


# Main training loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders()

    model = VisionLanguageModel(vocab_size=vision_tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vision_tokenizer.pad_token_id)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")


if __name__ == "__main__":
    main()

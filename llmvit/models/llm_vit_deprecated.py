import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

DEFAULT_LORA_CFG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    use_dora=False,
)
DEFAULT_BNB_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        embed_dim: int = 768,
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


class VisionTransformer(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        depth: int = 1,
        num_classes: int = 10,
        frozen_backbone_steps: int = -1,
        lora_config: LoraConfig = DEFAULT_LORA_CFG,
        bnb_config: BitsAndBytesConfig = DEFAULT_BNB_CFG,
    ) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.transformer = self._init_transformer_backbone(
            pretrained_model_name_or_path, depth, num_classes, lora_config, bnb_config
        )
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, self.transformer.config.hidden_size
        )
        self.frozen_backbone_steps = frozen_backbone_steps
        self.curr_steps = 0
        if self.frozen_backbone_steps > 0:
            self.freeze_backbone()
            self.backbone_frozen = True
        else:
            self.backbone_frozen = False

    def freeze_backbone(self):
        for name, param in self.transformer.named_parameters():
            if ("classifier" not in name) and ("score" not in name):
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.transformer.parameters():
            try:
                param.requires_grad = True
            except RuntimeError:
                pass

    def increment_frozen_backbone_steps(self):
        if self.training and self.backbone_frozen:
            self.curr_steps += 1
            if self.curr_steps >= self.frozen_backbone_steps:
                self.unfreeze_backbone()
                self.backbone_frozen = False

    def _init_transformer_backbone(
        self,
        pretrained_model_name_or_path: str,
        depth: int,
        num_classes: int,
        lora_config: LoraConfig,
        bnb_config: BitsAndBytesConfig,
    ) -> None:
        transformer = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_classes,
            num_hidden_layers=depth,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        transformer = prepare_model_for_kbit_training(transformer)
        transformer = get_peft_model(transformer, lora_config)
        return transformer

    def _get_embedding_size(self) -> int:
        return self.transformer.config.hidden_size

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return self.transformer(inputs_embeds=x).logits

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        preds = self._forward(images)
        loss = self.criterion(preds, labels) if labels is not None else None
        self.increment_frozen_backbone_steps()
        return {"loss": loss, "logits": preds}


if __name__ == "__main__":
    model = VisionTransformer("distilbert-base-uncased").cuda()
    print(model)
    img = torch.randn(1, 1, 28, 28).cuda()
    out = model(img)
    print(out["logits"].shape)
    print(out["loss"])

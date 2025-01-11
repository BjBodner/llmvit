import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import bitsandbytes as bnb


class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 
                      kernel_size=patch_size, stride=patch_size),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        x += self.pos_embed
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, pretrained_model_name_or_path, img_size=28, patch_size=7, in_channels=1, 
                embed_dim=768, depth=1, num_heads=8, num_classes=10):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.transformer = self._init_transformer_backbone(pretrained_model_name_or_path, embed_dim, depth, num_heads, num_classes)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        # self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.transformer.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def _init_transformer_backbone(self, pretrained_model_name_or_path, embed_dim, depth, num_heads, num_classes):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        transformer = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_classes,
            num_hidden_layers=depth,
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            attn_implementation = "eager",
            quantization_config=bnb_config,
        )
        transformer = prepare_model_for_kbit_training(transformer)
        
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="lora_only",
            # target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
            target_modules="all-linear",
        )
        transformer = get_peft_model(transformer, config)
        return transformer

    def _forward(self, x):
        x = self.patch_embed(x)
        return self.transformer(inputs_embeds=x).logits

    def forward(self, images, labels=None):
        preds = self._forward(images)
        loss = self.criterion(preds, labels)
        return {"loss": loss, "logits": preds}

if __name__ == "__main__":
    model = VisionTransformer("distilbert-base-uncased")
    print(model)
    img = torch.randn(1, 1, 28, 28)
    out = model(img)
    print(out["logits"].shape)
    print(out["loss"])
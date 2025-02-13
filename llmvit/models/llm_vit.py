import torch
import torch.nn as nn
from transformers import PreTrainedModel

# TODO: add an auxiliary loss to push the embeddings to be close to the text embeddings

from processor.img_processor import ImgProcessor, EncoderConfig, EmbedderConfig
from utils.train_utils import gaussian_kl_loss
class LLMVIT(nn.Module):
    def __init__(self, 
                 model: PreTrainedModel, 
                 img_processor_config: EncoderConfig = EncoderConfig(), 
                 embedder_config: EmbedderConfig = EmbedderConfig(), 
                 frozen_backbone_steps: int = -1, 
                 always_freeze_backbone: bool = False,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 embedding_loss_weight: float = 0.,
        ):
        super().__init__()
        self.model = model
        self.img_processor = ImgProcessor(model.get_input_embeddings(), img_processor_config, embedder_config)
        self.always_freeze_backbone = always_freeze_backbone
        self.frozen_backbone_steps = frozen_backbone_steps if not always_freeze_backbone else float("inf")
        self.curr_steps = 0
        self.criterion = criterion
        self.embedding_loss_weight = embedding_loss_weight
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

    def forward(self, images: torch.Tensor, labels: torch.Tensor = None) -> dict[str, torch.Tensor]:
        inputs_embeds, img_encoder_output = self.img_processor(images)
        outputs = self.model(inputs_embeds=inputs_embeds)
        self.increment_frozen_backbone_steps()

        if self.training and labels is not None:
            loss = self.criterion(outputs.logits, labels)
            if self.embedding_loss_weight > 0:                
                emb_mean = torch.mean(self.model.get_input_embeddings().weight, dim=0)
                emb_std = torch.std(self.model.get_input_embeddings().weight, dim=0)
                batch_mean = torch.mean(img_encoder_output, dim=1)
                batch_std = torch.std(img_encoder_output, dim=1)
                embedding_loss = gaussian_kl_loss(batch_mean, batch_std, emb_mean, emb_std)

                loss += self.embedding_loss_weight * embedding_loss
            return {"loss": loss, "logits": outputs.logits}
        return {"logits": outputs.logits, "loss": None}

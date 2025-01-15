import torch
import torch.nn as nn
from transformers import PreTrainedModel

# TODO: add an auxiliary loss to push the embeddings to be close to the text embeddings

from processor.img_processor import ImgProcessor, EncoderConfig, EmbedderConfig

class LLMVIT(nn.Module):
    def __init__(self, 
                 model: PreTrainedModel, 
                 img_processor_config: EncoderConfig, 
                 embedder_config: EmbedderConfig, 
                 frozen_backbone_steps: int = -1, 
                 always_freeze_backbone: bool = False,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 embedding_loss_weight: float = 0.1,
                 embedding_loss: str = nn.MSELoss(),
        ):
        super().__init__()
        self.model = model
        self.img_processor = ImgProcessor(model.get_input_embeddings(), img_processor_config, embedder_config)
        self.always_freeze_backbone = always_freeze_backbone
        self.frozen_backbone_steps = frozen_backbone_steps if not always_freeze_backbone else float("inf")
        self.curr_steps = 0
        self.criterion = criterion
        self.embedding_loss_weight = embedding_loss_weight
        self.embedding_loss = embedding_loss
        self.word_embedddings = self.model.get_input_embeddings()
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

    def forward(self, img: torch.Tensor, labels: torch.Tensor = None) -> dict[str, torch.Tensor]:
        inputs = self.img_processor(img)
        outputs = self.model(**inputs)
        self.increment_frozen_backbone_steps()
        if self.training and labels is not None:
            loss = self.criterion(outputs.logits, labels)
            if self.embedding_loss_weight > 0:
                embedding_loss = self.embedding_loss(inputs["inputs_embeds"], self.word_embedddings.weight.detach())
                loss += self.embedding_loss_weight * embedding_loss
            return {"loss": loss, "logits": outputs.logits}
        return {"logits": outputs.logits, "loss": None}

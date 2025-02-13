import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

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


def image_collator(
    features: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    images = torch.stack([f[0] for f in features])
    labels = torch.tensor([f[1] for f in features])
    return {"images": images, "labels": labels}


def gaussian_kl_loss(mean1, std1, mean2, std2):
    """
    Compute KL divergence between two Gaussians.
    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    """
    var1, var2 = std1.pow(2), std2.pow(2)
    eps = 1e-8
    kl_div = (
        torch.log(std2 + eps)
        - torch.log(std1 + eps)
        + (var1 + (mean1 - mean2.detach()).pow(2)) / (2 * var2.detach() + eps)
        - 0.5
    )

    return kl_div.mean()


def get_classification_model(
    pretrained_model_name_or_path: str,
    depth: int,
    num_classes: int,
    lora_config: LoraConfig,
    bnb_config: BitsAndBytesConfig,
) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=num_classes,
        num_hidden_layers=depth,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
    return model

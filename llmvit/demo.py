import os

import torchvision
from dotenv import load_dotenv

torchvision.disable_beta_transforms_warning()
from huggingface_hub import login
from transformers import Trainer, TrainingArguments, set_seed

import wandb
from llmvit import LLMVIT, EmbedderConfig, EncoderConfig
from llmvit.utils.datasets import get_dataset
from llmvit.utils.metrics import compute_metrics
from llmvit.utils.train_utils import (
    DEFAULT_BNB_CFG,
    DEFAULT_LORA_CFG,
    get_classification_model,
    image_collator,
)

load_dotenv()
set_seed(42)
login(
    token=os.getenv("HUGGINGFACE_TOKEN")
)  # add your Hugging Face token to a .env file


def main():

    wandb.init(project="llm-vit", entity="wandb", mode="disabled")

    model = LLMVIT(
        model=get_classification_model(
            "distilbert-base-uncased",
            depth=2,
            num_classes=10,
            lora_config=DEFAULT_LORA_CFG,
            bnb_config=DEFAULT_BNB_CFG,
        ),
        img_processor_config=EncoderConfig(type="patch_embed", in_channels=1),
        embedder_config=EmbedderConfig(type="direct", k=2, temperature=1.0),
    )

    dataset = get_dataset("MNIST")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./outputs/mnist_vit",
            run_name="mnist_vit",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            optim="paged_adamw_32bit",
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=image_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()

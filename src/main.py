import os
import sys
import torch
from transformers import Trainer, TrainingArguments, set_seed
from huggingface_hub import login
import dotenv
dotenv.load_dotenv()

sys.path.append(os.getcwd())

from src.models.llm_vit import VisionTransformer
from utils.datasets import get_dataset
from src.utils.metrics import compute_metrics
from utils.train_utils import image_collator

set_seed(42)
login(token=os.getenv("HUGGINGFACE_TOKEN"))

def main():
    model = VisionTransformer("distilbert-base-uncased")
    # model = VisionTransformer("microsoft/phi-4")
    # model = VisionTransformer("meta-llama/Meta-Llama-3.1-8B-Instruct")
    dataset = get_dataset("MNIST")
    trainer = Trainer(
       model=model,
       args=TrainingArguments(
           output_dir="./outputs/mnist_vit",
           run_name="mnist_vit",
           num_train_epochs=3,
           per_device_train_batch_size=128,
           learning_rate=2e-3,
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

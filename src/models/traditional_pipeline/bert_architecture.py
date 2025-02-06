import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


class ABSABERT:
    def __init__(self, aspects, model_name='indolem/indobert-base-uncased', load_models=True):
        self.aspects = aspects
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.models = {}
        if load_models:
            self.models = {
                a: AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
                for a in aspects
            }
        # Dynamically set device based on CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader_dict, val_loader_dict, epochs=10, lr=2e-5):
        scaler = GradScaler()  # For mixed precision
        for aspect in self.aspects:
            print(f"\nTraining BERT for {aspect}".ljust(50, '-'))
            model = self.models[aspect].to(self.device)
            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(train_loader_dict[aspect]) * epochs
            )

            best_loss = float('inf')
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for batch in train_loader_dict[aspect]:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    optimizer.zero_grad()

                    with autocast():  # Mixed precision
                        outputs = model(**batch)
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader_dict[aspect])
                # Validation step
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader_dict[aspect]:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        with autocast():
                            outputs = model(**batch)
                        val_loss += outputs.loss.item()
                avg_val_loss = val_loss / len(val_loader_dict[aspect])
                print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), f"best_{aspect}.pt")
            model.load_state_dict(torch.load(f"best_{aspect}.pt"))


    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        # Save tokenizer to the main directory
        self.tokenizer.save_pretrained(model_dir)
        # Save each model to its own subdirectory
        for aspect in self.aspects:
            aspect_dir = model_dir / aspect
            aspect_dir.mkdir(exist_ok=True)
            # Save model and tokenizer (if needed)
            self.models[aspect].save_pretrained(aspect_dir)

    @classmethod
    def load(cls, model_dir, aspects):
        model_dir = Path(model_dir)
        # Create instance without initializing models
        instance = cls(aspects, model_name=model_dir, load_models=False)

        # Load models and move them to the instance's device
        instance.models = {
            a: AutoModelForSequenceClassification.from_pretrained(model_dir / a).to(instance.device)
            for a in aspects
        }
        return instance
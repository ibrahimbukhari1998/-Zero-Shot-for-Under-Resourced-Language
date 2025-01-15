#----------------------------Headers----------------------------#
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizerFast,  # Changed from XLMRobertaTokenizer to XLMRobertaTokenizerFast
    XLMRobertaForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import evaluate
from datasets import Dataset, load_dataset
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



#----------------------------Reducing Dataset Size----------------------------#

def reduce_dataset_size(texts, tags, fraction=0.1, random_seed=42):
    random.seed(random_seed)

    # Pair texts with their corresponding tags for sampling
    paired_data = list(zip(texts, tags))

    # Sample a fraction of the paired data
    reduced_data = random.sample(paired_data, int(len(paired_data) * fraction))

    # Unzip the reduced data back into separate lists
    reduced_texts, reduced_tags = zip(*reduced_data)
    return list(reduced_texts), list(reduced_tags)


#----------------------------Part of Speech Dataset Processing----------------------------#

class POSDataset(Dataset):
    def __init__(self, texts: List[List[str]], tags: List[List[str]], tokenizer, tag2id: Dict[str, int], max_len: int = 128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx]
        tags = self.tags[idx]

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label_ids = []
        word_ids = encoded.word_ids()

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.tag2id[tags[word_idx]])

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }



#----------------------------XLM-R Pipeline----------------------------#

class POSTaggingPipeline:
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LOG: Using device: {self.device}")

        self.model_name = model_name
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
        logger.info("LOG: Initialized XLMRoberta Tokenizer Fast")

        self.tag2id = None
        self.id2tag = None
        self.model = None

    def prepare_data(self, dataset_name: str, split: str = "train") -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load and prepare data from Universal Dependencies dataset.

        Args:
            dataset_name (str): Name of the UD dataset (e.g., 'en_ewt', 'wo_wtb')
            split (str): Dataset split ('train', 'validation', 'test')

        Returns:
            Tuple[List[List[str]], List[List[str]]]: Tuple of (texts, tags)
        """
        logger.info(f"Loading {dataset_name} dataset, {split} split")
        try:
            # Load the dataset
            dataset = load_dataset("universal_dependencies", dataset_name)

            # Get the specified split
            data_split = dataset[split]

            # Extract texts and tags
            texts = [item['tokens'] for item in data_split]
            tags = [item['upos'] for item in data_split]

            logger.info(f"Loaded {len(texts)} sentences from {dataset_name} {split} split")

            # Basic validation
            assert all(len(text) == len(tag) for text, tag in zip(texts, tags)), \
                "Mismatch between text and tag lengths"

            return texts, tags

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise

    def initialize_model(self, num_labels: int):
        """Initialize the model with proper classification head"""
        logger.info(f"Initializing model with {num_labels} labels")
            
        self.model = XLMRobertaForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
        logger.info("LOG: Initialized XLM-R Model")

        # Initialize classification layer
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        torch.nn.init.zeros_(self.model.classifier.bias)

        self.model = self.model.to(self.device)
        logger.info("Model initialized and moved to device")

    def create_tag_mappings(self, tags: List[List[str]]):
        """Create tag to ID mappings"""
        unique_tags = sorted(list(set(tag for seq in tags for tag in seq)))
        self.tag2id = {tag: i for i, tag in enumerate(unique_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}
        logger.info(f"Created mappings for {len(unique_tags)} unique tags: {unique_tags}")

    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with tokens"""
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True,
            is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["upos"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def train(self, train_texts: List[List[str]], train_tags: List[List[str]],
          eval_texts: List[List[str]] = None, eval_tags: List[List[str]] = None,
          epochs: int = 3, batch_size: int = 16, push_to_hub: bool = False):
        
        """Train the model on source language data"""
        if self.tag2id is None:
            self.create_tag_mappings(train_tags)

        self.initialize_model(len(self.tag2id))
        
        # Prepare datasets
        train_examples = {
            "tokens": train_texts,
            "upos": [[self.tag2id[t] for t in tag_seq] for tag_seq in train_tags]
        }
        train_dataset = Dataset.from_dict(train_examples)
        
        if eval_texts:
            eval_examples = {
                "tokens": eval_texts,
                "upos": [[self.tag2id[t] for t in tag_seq] for tag_seq in eval_tags]
            }
            eval_dataset = Dataset.from_dict(eval_examples)
        else:
            eval_dataset = None

        # Initialize data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"xlmr_model_{self.model_name.split('/')[-1]}",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_texts else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_texts else False,
            push_to_hub=push_to_hub
        )

        # Define metrics computation
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = classification_report(
                [t for l in true_labels for t in l],
                [t for l in true_predictions for t in l],
                output_dict=True
            )
            return {
                "precision": results["weighted avg"]["precision"],
                "recall": results["weighted avg"]["recall"],
                "f1": results["weighted avg"]["f1-score"],
                "accuracy": results["accuracy"],
            }

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Push to hub if requested
        if push_to_hub:
            trainer.push_to_hub()

    def evaluate(self, eval_loader: DataLoader) -> Dict:
        """Evaluate the model"""
        self.model.eval()
        true_labels = []
        pred_labels = []

        logger.info("Starting evaluation...")
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=2)

                for i in range(labels.shape[0]):
                    true_seq = [self.id2tag[l.item()] for l in labels[i] if l.item() != -100]
                    pred_seq = [self.id2tag[p.item()] for p, l in zip(predictions[i], labels[i]) if l.item() != -100]
                    true_labels.extend(true_seq)
                    pred_labels.extend(pred_seq)

        return classification_report(true_labels, pred_labels, output_dict=True)

    def predict(self, texts: List[List[str]]) -> List[List[str]]:
        """
        Predict POS tags for new texts while maintaining original token alignment
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        self.model.eval()
        predictions = []

        logger.info("Starting prediction...")
        with torch.no_grad():
            for text in texts:
                # Tokenize the text while keeping track of word IDs
                encoded = self.tokenizer(
                    text,
                    is_split_into_words=True,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )

                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                word_ids = encoded.word_ids()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

                # Initialize predictions for this sentence
                sent_predictions = []
                prev_word_idx = None

                # Align predictions with original tokens
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue
                    if word_idx != prev_word_idx:
                        sent_predictions.append(self.id2tag[preds[token_idx]])
                        prev_word_idx = word_idx

                # Ensure predictions match the input length
                sent_predictions = sent_predictions[:len(text)]

                # If predictions are shorter (due to truncation), pad with the most common tag
                if len(sent_predictions) < len(text):
                    most_common_tag = max(set(self.id2tag.values()), key=list(self.id2tag.values()).count)
                    sent_predictions.extend([most_common_tag] * (len(text) - len(sent_predictions)))

                predictions.append(sent_predictions)

        logger.info(f"Generated predictions for {len(predictions)} sentences")

        # Verify prediction lengths match input lengths
        for text, pred in zip(texts, predictions):
            assert len(text) == len(pred), f"Mismatch in lengths: text={len(text)}, pred={len(pred)}"

        return predictions
    
    def get_Classification_Report(self, gold_tags, predictions):
        report = classification_report([tag for sent in gold_tags for tag in sent],[tag for sent in predictions for tag in sent])
        return report




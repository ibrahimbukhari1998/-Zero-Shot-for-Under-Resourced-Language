import torch 
import logging
import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from typing import List, Dict, Tuple, Optional
from data_utils import standardize_dataset_size, add_noise_to_text 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class POSDataset(Dataset):
    def __init__(self, texts: List[List[str]], tags: List[List[str]], 
                 tokenizer, tag2id: Dict[str, int], max_len: int = 128,
                 bpe_dropout: float = 0.1):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        self.bpe_dropout_prob = bpe_dropout

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
            return_tensors='pt',
            dropout=self.bpe_dropout_prob
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
    
#----------------------------  POSpipeline Glot500  ----------------------------#

class POSpipeline:
    
    def __init__(self, train_data_code:str, model_name = "cis-lmu/glot500-base") -> None:
        
        self.train_dataset = load_dataset("universal_dependencies", train_data_code)
        self.label_list = self.train_dataset["train"].features[f"upos"].feature.names
        self.train_data_code = train_data_code

        
        self.id2label = {idx:label for idx, label in enumerate(self.label_list)}
        self.label2id = {self.id2label[idx]:idx for idx in self.id2label}
        
        
        self.seqeval = evaluate.load("seqeval")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                    num_labels=len(self.id2label), 
                                                                    id2label=self.id2label, 
                                                                    label2id=self.label2id )
        
        
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.tokenized_dataset = self.train_dataset.map(self.tokenize_and_align_labels, batched=True)
    
    def prepare_data(self, dataset_name: str, split: str = "train",
                second_dataset_name: Optional[str] = None,
                standardize_size: bool = True,
                add_noise: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
        """Load and prepare data from Universal Dependencies datasets"""
        logger.info(f"Loading {dataset_name} dataset, {split} split")
        
        # Load primary dataset
        dataset1 = load_dataset("universal_dependencies", dataset_name)
        texts1 = [item['tokens'] for item in dataset1[split]]
        tags1 = [item['upos'] for item in dataset1[split]]
        
        if second_dataset_name:
            logger.info(f"Loading {second_dataset_name} dataset, {split} split")
            dataset2 = load_dataset("universal_dependencies", second_dataset_name)
            texts2 = [item['tokens'] for item in dataset2[split]]
            tags2 = [item['upos'] for item in dataset2[split]]
            
            texts = texts1 + texts2
            tags = tags1 + tags2
        else:
            texts = texts1
            tags = tags1

        if standardize_size:
            texts, tags = standardize_dataset_size(texts, tags)
            
        if add_noise:
            texts = add_noise_to_text(texts)

        logger.info(f"Final dataset size: {len(texts)} sentences")
        return texts, tags

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"upos"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    
    def train(self, epochs = 2, batch_size = 16, push_to_hub=False, learning_rate: float = 2e-5, bpe_dropout: float = 0.1):
        
        training_args = TrainingArguments(
            output_dir=f"glot500_model_{self.train_data_code}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            no_cuda=True,
            push_to_hub=True
            )
        
        # Create datasets with BPE dropout
        train_data = self.train_dataset["train"]
        eval_data = self.train_dataset["test"]
        
        train_dataset = POSDataset(
            texts=[item['tokens'] for item in train_data],
            tags=[item['upos'] for item in train_data],
            tokenizer=self.tokenizer,
            tag2id=self.label2id,
            bpe_dropout=bpe_dropout
        )
        
        eval_dataset = POSDataset(
            texts=[item['tokens'] for item in eval_data],
            tags=[item['upos'] for item in eval_data],
            tokenizer=self.tokenizer,
            tag2id=self.label2id,
            bpe_dropout=0.0  # No dropout for eval
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        
        if push_to_hub:
            trainer.push_to_hub()
    
    
    def predict(self, input_text:str):
        
        tokens = self.tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.model(**tokens).logits
        
        predictions = torch.argmax(logits, dim=2)
        predictions = predictions[0].tolist()
        
        return predictions
    
    
    def evaluate(self, data_code:str):
        
        test_dataset = load_dataset("universal_dependencies", data_code)
        tokenized_dataset = test_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        prediction_tags = []
        true_tags = []
        
        
        for input in tokenized_dataset['train']:
            true_pred = []
            true_label = []
            
            pred_tags = self.predict(input_text=input['text'])
            true_labels = input['labels']
            
            for p, l in zip(pred_tags, true_labels):
                if l != -100:
                    true_pred.append(self.id2label[p])
                    true_label.append(self.id2label[l])
            
            prediction_tags.append(true_pred)
            true_tags.append(true_label)
        
        # result = self.seqeval.compute(predictions=prediction_tags, references=true_tags)
        result = classification_report([tag for sent in true_tags for tag in sent],[tag for sent in prediction_tags for tag in sent])
        return result

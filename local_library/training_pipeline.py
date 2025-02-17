import torch
import torch.nn as nn
import random
import logging
import evaluate
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from typing import List
from transformers import XLMRobertaTokenizer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DropoutConfig: 
    ''' Configuration for different dropout strategies '''
    model_type: str
    dropout_prob: float = 0.1
    training: bool = True

class TokenizerWithDropout:
    def __init__(self, model_name: str, config: DropoutConfig):
        self.config = config
        if config.model_type == 'xlmr':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def tokenize(self, tokens, training=True):
        if not training:
            return self.tokenizer(tokens, is_split_into_words=True, return_tensors='pt')['input_ids'][0].tolist()

        if self.config.model_type == 'xlmr':
            # BPE dropout for XLM-R
            subword_tokens = self.tokenizer.tokenize(" ".join(tokens))
            kept_tokens = [token for token in subword_tokens if random.random() >= self.config.dropout_prob]
            return self.tokenizer.convert_tokens_to_ids(kept_tokens)

        else:
            # Word dropout for Glot500
            kept_tokens = [token for token in tokens if random.random() >= self.config.dropout_prob]
            return self.tokenizer(" ".join(kept_tokens), return_tensors='pt')['input_ids'][0].tolist()

    # Delegate `pad` to the underlying tokenizer
    def pad(self, *args, **kwargs):
        return self.tokenizer.pad(*args, **kwargs)

    # Delegate `convert_tokens_to_ids`
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    # Delegate `decode`
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # Optionally delegate other properties if needed
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    
#----------------------------  POS Dataset ----------------------------#

class POSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, tag2id, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

        # Debugging: print unique labels in the dataset
        unique_labels = set(l for label_seq in self.labels for l in label_seq)
        print(f"✅ Unique labels in dataset: {unique_labels}")
        print(f"✅ tag2id mapping: {self.tag2id}")

        # Check if any label is out of range
        max_label_id = max(self.tag2id.values())
        invalid_labels = [l for l in unique_labels if l not in self.tag2id]
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}, max label should be {max_label_id}")

    
    def __len__(self):
        return len(self.texts)
    
    
    def __getitem__(self, idx):
      sentence = self.texts[idx]
      labels = self.labels[idx]

      encoded = self.tokenizer(
          sentence,
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
              # 🚀 Debugging: Check if label exists in tag2id
              if labels[word_idx] not in self.tag2id:
                  raise ValueError(f"Found unknown label {labels[word_idx]} in sentence {sentence}")

              label_ids.append(self.tag2id[labels[word_idx]])

      return {
          'input_ids': encoded['input_ids'].squeeze(),
          'attention_mask': encoded['attention_mask'].squeeze(),
          'labels': torch.tensor(label_ids)
      }




#----------------------------  POS Training Pipeline  ----------------------------#

class POSpipeline:
    
    def __init__(self, train_data_codes:List[str], model_name = str, 
                character_level_injection=False, injection_vocab="", injection_prob=0.2, 
                sample_threshold=0, use_dropout=False, dropout_prob=0.1) -> None:
        
        # Loading the data
        self.sample_threshold = sample_threshold
        self.train_data_codes = train_data_codes
        self.train_data, self.train_labels, self.eval_data, self.eval_labels, self.label_list = self.__perpare_data(train_data_codes)

        # Adding Character Level Noise
        self.injection_vocab = injection_vocab
        self.injection_prob = injection_prob
        
        if character_level_injection:
            self.train_data = self.character_level_augmentation(self.train_data)
            print("Character Level Noise Added")

        # initalize dropout if enabled
        self.use_dropout = use_dropout
        if use_dropout:
            model_type = 'xlmr' if 'xlmr-roberta' in model_name.lower() else 'glot500'
            dropout_config = DropoutConfig(
                model_type=model_type,
                dropout_prob=dropout_prob
            )
            self.tokenizer = TokenizerWithDropout(model_name, dropout_config)
        else: 
            # regular tokenizer for other experiments
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Creating label2id & id2label dictionaries
        # self.label2id = {label: idx for idx, label in enumerate(set(label for labels in self.train_labels for label in labels))}
        self.label2id = {idx: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Loading the model
        self.seqeval = evaluate.load("seqeval")
        print(f"Setting up model with {len(self.label2id)} labels")
        self.model = AutoModelForTokenClassification.from_pretrained(
          model_name,
          num_labels=len(self.label2id),  # Ensure correct number of labels
          id2label=self.id2label,
          label2id=self.label2id
          )

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Converting Data to TF_Dataset format
        self.train_dataset = POSDataset(
          texts=self.train_data,
          labels=self.train_labels,
          tokenizer=self.tokenizer,
          tag2id=self.label2id  # ✅ Add this argument
        )

        
        self.eval_dataset = POSDataset(
            texts=self.eval_data,
            labels=self.eval_labels,
            tokenizer=self.tokenizer,
            tag2id=self.label2id
        )

    #=================== Prepate Data ===================#

    def __perpare_data(self, data_codes: List[str]):
        train_data, train_labels = [], []
        eval_data, eval_labels = [], []
        label_list = None
        
        for data_code in data_codes:
            # Load the dataset
            dataset = load_dataset("universal_dependencies", data_code)
            label_list = dataset["train"].features[f"upos"].feature.names
            
            # Extract the texts and tags
            train_texts=[item['tokens'] for item in dataset['train']]
            train_tags=[item['upos'] for item in dataset['train']]
            eval_texts=[item['tokens'] for item in dataset['validation']]
            eval_tags=[item['upos'] for item in dataset['validation']]
            
            # Sample the data
            if self.sample_threshold > 0.0:
                if self.sample_threshold < len(train_texts):
                    train_texts = train_texts[:int(self.sample_threshold)]
                    train_tags = train_tags[:int(self.sample_threshold)]
            
            # Add the data to the lists
            train_data.extend(train_texts)
            train_labels.extend(train_tags)
            
            eval_data.extend(eval_texts)
            eval_labels.extend(eval_tags)
        
        return train_data, train_labels, eval_data, eval_labels, label_list
    
    #=================== Adding Character Level Noise ===================#
    
    def character_level_augmentation(self, data):
        """Apply character-level noise"""  
        new_data = []
        
        print("DATA\n", data)
        
        for sentence in data:
            new_sentence = []
            
            for word in sentence:
                
                actions = ["insert", "replace"]
                word = list(word)
                prob = self.injection_prob
                
                if random.random() > prob:
                    new_sentence.append("".join(word))
                    continue
                
                action = random.choice(actions)
                if action == "insert":
                    idx = random.randint(0, len(word))
                    char = random.choice(self.injection_vocab)
                    word.insert(idx, char)
                elif action == "replace" and len(word) > 0:
                    idx = random.randint(0, len(word) - 1)
                    char = random.choice(self.injection_vocab)
                    word[idx] = char
                
                new_sentence.append("".join(word))
            
            new_data.append(new_sentence)
        
        print("NEW DATA\n", new_data)
        
        return new_data

    #==================== Helper Functions ============================#

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
    
    #=================== Training ===================#
    
    def train(self, epochs = 2, batch_size = 16, set_name=False, output_name=''):
        # enable dropout training mode if using dropout
        if self.use_dropout:
            self.tokenizer.config.training = True

        if set_name == False:
            joined_codes = "_".join(self.train_data_codes)
            name = f"glot500_{joined_codes}"
        else:
            name = output_name
        
        self.tuned_model_name = name
        
        training_args = TrainingArguments(
            output_dir=name,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # no_cuda=True,
            push_to_hub=True
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print(f"Started Training on Data: {' | '.join(self.train_data_codes)}")
        self.trainer.train()
        
        # disable dropout after training
        if self.use_dropout:
            self.tokenizer.config.training = False

    
    def push_to_hub(self):
        self.trainer.push_to_hub()
        print(f"Model Pushed to Hub with name: {self.tuned_model_name}")
        
    #=================== Evaluating ===================#
    
    def predict(self, input_text:str):
        
        tokens = self.tokenizer(input_text, return_tensors="pt")
        
        # Ensure all tensors are on the same device as the model
        device = next(self.model.parameters()).device  # Get the model's device
        tokens = {k: v.to(device) for k, v in tokens.items()} # Move input tensors to the model's device
        
        with torch.no_grad():
            logits = self.model(**tokens).logits
        
        predictions = torch.argmax(logits, dim=2)
        predictions = predictions[0].tolist()
        
        return predictions
    
    
    def evaluate(self, data_code:str):

        # disable dropout for evaluation
        if self.use_dropout:
            self.tokenizer.config.training = False
        
        test_dataset = load_dataset("universal_dependencies", data_code)
        tokenized_dataset = test_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        prediction_tags = []
        true_tags = []
        
        
        for input in tokenized_dataset['test']:
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

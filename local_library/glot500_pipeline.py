import torch 
import random
import logging
import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer




#----------------------------  POSpipeline Glot500  ----------------------------#

class POSpipeline:
    
    def __init__(self, train_data_code:str, model_name = "cis-lmu/glot500-base", character_level_injection=False, injection_vocab="", injection_prob=0.2) -> None:
        
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
        self.injection_vocab = injection_vocab
        self.injection_prob = injection_prob
        
        if character_level_injection:
            self.tokenized_dataset = self.train_dataset.map(self.augment_tokenize_and_align_labels, batched=True)
        else:
            self.tokenized_dataset = self.train_dataset.map(self.tokenize_and_align_labels, batched=True)

    #=================== Adding Character Level Noise ===================#
    
    def character_level_augmentation(self, word):
        """Apply character-level augmentation to a single word."""
        actions = ["insert", "delete", "replace", "swap"]
        word = list(word)
        prob = self.injection_prob
        
        if random.random() > prob:
            return "".join(word)
        
        action = random.choice(actions)
        if action == "insert":
            idx = random.randint(0, len(word))
            char = random.choice(self.injection_vocab)
            word.insert(idx, char)
        elif action == "delete" and len(word) > 1:
            idx = random.randint(0, len(word) - 1)
            word.pop(idx)
        elif action == "replace" and len(word) > 0:
            idx = random.randint(0, len(word) - 1)
            char = random.choice(self.injection_vocab)
            word[idx] = char
        elif action == "swap" and len(word) > 1:
            idx = random.randint(0, len(word) - 2)
            word[idx], word[idx + 1] = word[idx + 1], word[idx]
        
        return "".join(word)
    
    def augment_tokenize_and_align_labels(self, examples):
    
        augmented_tokens = [
            [self.character_level_augmentation(token) for token in sentence]
            for sentence in examples["tokens"]
        ]
        
        tokenized_inputs = self.tokenizer(augmented_tokens, truncation=True, is_split_into_words=True)
        
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
    
    #===================================================================#
    
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
    
    
    def train(self, epochs = 2, batch_size = 16, set_name=False, output_name='', train_size=10000):
        
        if set_name == False:
            name = f"glot500_model_{self.train_data_code}"
        else:
            name = output_name
        
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
            no_cuda=True,
            push_to_hub=True
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"].take(train_size),
            eval_dataset=self.tokenized_dataset["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        self.trainer.train()

    
    def push_to_hub(self):
        self.trainer.push_to_hub()
    
    
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

import torch 
import random
import logging
import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from typing import List
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#----------------------------  POS Dataset ----------------------------#

class POSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    
    def __len__(self):
        return len(self.texts)
    
    
    def __getitem__(self,idx):
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
                label_ids.append(labels[word_idx])

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }


#----------------------------  POS Training Pipeline  ----------------------------#

class POSpipeline:
    
    def __init__(self, train_data_codes:List[str], model_name = str, 
                character_level_injection=False, injection_vocab="", injection_prob=0.2, 
                sample_threshold=0) -> None:
        
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


        # Creating label2id & id2label dictionaries
        self.label2id = {label: idx for idx, label in enumerate(set(label for labels in self.train_labels for label in labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Loading the tokenizer and model
        self.seqeval = evaluate.load("seqeval")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                    num_labels=len(self.id2label), 
                                                                    id2label=self.id2label, 
                                                                    label2id=self.label2id )
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        
        # Converting Data to TF_Dataset format
        self.train_dataset = POSDataset(
            texts=self.train_data,
            labels=self.train_labels,
            tokenizer=self.tokenizer,
        )
        
        self.eval_dataset = POSDataset(
            texts=self.eval_data,
            labels=self.eval_labels,
            tokenizer=self.tokenizer,
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
            eval_texts=[item['tokens'] for item in dataset['test']]
            eval_tags=[item['upos'] for item in dataset['test']]
            
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






#----------------------------  Fine-tune and Evaluation: Single Test ----------------------------#

def tuning_and_evaluating(fine_tune_data_codes:List[str], test_data_code:str, model_name:str, tuned_model_name:str,
                        character_level_injection=bool, injection_vocab=str, injection_prob=float,
                        sample_threshold=int):
    """
    Input:
        fine_tune_data_codes: List of data codes to fine-tune the model
        test_data_code: Data code to test the model
        model_name: Name of the model to train
        character_level_injection: Whether to inject character level noise
        injection_vocab: Vocabulary for character level injection
        injection_prob: Probability of injecting the noise

    Output:
        Classification report for the model on the test data
    """
    
    # Fine Tuning the model on multiple datasets
    multiple_data_pipeline = POSpipeline(
                                                        train_data_codes=fine_tune_data_codes, 
                                                        model_name=model_name,
                                                        character_level_injection=character_level_injection,
                                                        injection_vocab=injection_vocab,
                                                        injection_prob=injection_prob,
                                                        sample_threshold=sample_threshold
                                                        )
    multiple_data_pipeline.train(set_name=True, output_name=tuned_model_name)
    multiple_data_pipeline.push_to_hub()
    
    # Evaluating the model on the test data
    result = multiple_data_pipeline.evaluate(test_data_code)
    
    return result



#----------------------------  Batch Running ----------------------------#

def batch_tune_eval(parameters:List[dict]):
    """
    Input:
        parameters: List of dictionaries containing the parameters for the pipeline
        
        paramters = [
            {
                'tuning_codes': List of data codes to fine-tune the model
                'test_code': string of Data code to test the model
                'model_name': string Name of the model to train
                'tuned_model_name': string Name of the fine-tuned model
                'character_level_injection': Bool of Whether to inject character level noise
                'injection_vocab': string Vocabulary for character level injection
                'injection_prob': float Probability of injecting the noise
                'sample_threshold': int Number of samples to consider
            }
        ]
    Output:
        List of dictionaries containing the results for the pipeline
    """
    
    
    results =[]
    
    # Running the pipeline for multiple parameters
    for param in parameters:
        
        # Extracting the parameters
        tuning_codes = param['tuning_codes']
        test_code = param['test_code']
        model_name = param['model_name']
        tuned_model_name = param['tuned_model_name']
        character_level_injection = param['character_level_injection']
        injection_vocab = param['injection_vocab']
        injection_prob = param['injection_prob']
        sample_threshold = param['sample_threshold']
        
        # Running the pipeline
        result = tuning_and_evaluating(tuning_codes, test_code, model_name, tuned_model_name,
                                        character_level_injection, injection_vocab, injection_prob,
                                        sample_threshold)
        
        # Storing the results
        print("="*100)
        results.append({'tuning_codes':tuning_codes, 'test_code':test_code, 'model_name':model_name, 'result':result})
        print(f"Model: {model_name}\n, Tuning Data : {tuning_codes}\n, Test Data: {test_code}\n, Result: {result}")
        print("="*100)
    
    return results



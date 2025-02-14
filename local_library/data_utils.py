from typing import List, Tuple
import random
import numpy as np
from sklearn.utils import resample

def standardize_dataset_size(texts: List[List[str]], 
                            tags: List[List[str]], 
                            target_size: int = 10000,
                            random_seed: int = 42) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Standardize dataset size through over/under sampling to reach target_size.
    """
    current_size = len(texts)
    
    if current_size == target_size:
        return texts, tags
        
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    combined_data = list(zip(texts, tags))
    
    if current_size > target_size:
        standardized_data = resample(combined_data,
                                    n_samples=target_size,
                                    random_state=random_seed,
                                    replace=False)
    else:
        standardized_data = resample(combined_data,
                                    n_samples=target_size,
                                    random_state=random_seed,
                                    replace=True)
    
    standardized_texts, standardized_tags = zip(*standardized_data)
    return list(standardized_texts), list(standardized_tags)

def add_noise_to_text(texts: List[List[str]], 
                        noise_ratio: float = 0.15,
                        random_seed: int = 42) -> List[List[str]]:
    """
    Add noise to text data by randomly modifying tokens.
    """
    random.seed(random_seed)
    noisy_texts = []
    
    for text in texts:
        noisy_text = text.copy()
        num_tokens_to_modify = int(len(text) * noise_ratio)
        
        if num_tokens_to_modify == 0:
            noisy_texts.append(noisy_text)
            continue
            
        indices_to_modify = random.sample(range(len(text)), num_tokens_to_modify)
        
        for idx in indices_to_modify:
            token = text[idx]
            noise_op = random.choice(['swap', 'delete', 'insert', 'substitute'])
            
            if noise_op == 'swap' and idx < len(text) - 1:
                noisy_text[idx], noisy_text[idx + 1] = noisy_text[idx + 1], noisy_text[idx]
            elif noise_op == 'delete':
                noisy_text[idx] = ''
            elif noise_op == 'insert':
                pos = random.randint(0, len(token))
                char = chr(random.randint(97, 122))
                noisy_text[idx] = token[:pos] + char + token[pos:]
            elif noise_op == 'substitute':
                if len(token) > 0:
                    pos = random.randint(0, len(token)-1)
                    char = chr(random.randint(97, 122))
                    noisy_text[idx] = token[:pos] + char + token[pos+1:]
        
        noisy_text = [t for t in noisy_text if t != '']
        noisy_texts.append(noisy_text)
    
    return noisy_texts
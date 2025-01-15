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
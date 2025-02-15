from typing import List
import local_library.training_pipeline as pipeline

"""
    Training Pipeline on Single Datasets
    Training Pipeline on Multiple Datasets
    Training Pipeline with character level injection
    Training Pipeline with bpe dropout

    Evaluating the model on Low resource languages
"""


#----------------------------  Fine-tune and Evaluation: Single Test + dropout (optional) ----------------------------#

def tuning_and_evaluating(fine_tune_data_codes:List[str], test_data_code:str, model_name:str, tuned_model_name:str,
                        character_level_injection=bool, injection_vocab=str, injection_prob=float, use_dropout=False, dropout_prob=0.0,
                        sample_threshold=int):
    """
    Input:
        fine_tune_data_codes: List of data codes to fine-tune the model
        test_data_code: Data code to test the model
        model_name: Name of the model to train
        character_level_injection: Whether to inject character level noise
        injection_vocab: Vocabulary for character level injection
        injection_prob: Probability of injecting the noise
        use_dropout: whether to use dropout (BPE for xlmr, word for glot500)
        dropout_prob: Probability of dropout

    Output:
        Classification report for the model on the test data
    """
    
    # Fine Tuning the model on multiple datasets
    multiple_data_pipeline = pipeline.POSpipeline(
                                                        train_data_codes=fine_tune_data_codes, 
                                                        model_name=model_name,
                                                        character_level_injection=character_level_injection,
                                                        injection_vocab=injection_vocab,
                                                        injection_prob=injection_prob,
                                                        use_dropout=use_dropout,
                                                        dropout_prob=dropout_prob,
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
                'use_dropout': Bool of whether to use dropout
                'dropout_prob': float Probability of dropout
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

        use_dropout = param.get('use_dropout', False)
        dropout_prob = param.get('dropout_prob', 0.0)

        sample_threshold = param['sample_threshold']

        # print experiment type and settings
        print("\n" + "="*100)
        if use_dropout:
            model_type = 'xlmr' if 'xlm-roberta' in model_name.lower() else 'glot500'
            dropout_type = 'BPE dropout' if model_type == 'xlmr' else 'Word dropout'
            print(f"Running experiment with {dropout_type} on {model_type}")
            print(f"Dropout probability: {dropout_prob}")
        elif character_level_injection:
            print(f"Running experiment with character level noise injection")
            print(f"Injection probability: {injection_prob}")
            print(f"Injection vocabulary: {injection_vocab}")
        print(f"Model: {model_name}")
        print(f"Training data: {tuning_codes}")
        print(f"Test data: {test_code}")
        print("="*100)
        
        # Running the pipeline
        result = tuning_and_evaluating(tuning_codes, test_code, model_name, tuned_model_name,
                                        character_level_injection, injection_vocab, injection_prob,
                                        use_dropout, dropout_prob,
                                        sample_threshold)
        
        # Store experiment details and results
        experiment_details = {
            'tuning_codes': tuning_codes,
            'test_code': test_code,
            'model_name': model_name,
            'experiment_type': 'dropout' if use_dropout else 'noise_injection'
        }
        
        # Add experiment-specific parameters
        if use_dropout:
            experiment_details.update({
                'model_type': model_type,
                'dropout_type': dropout_type,
                'dropout_prob': dropout_prob
            })
        elif character_level_injection:
            experiment_details.update({
                'injection_vocab': injection_vocab,
                'injection_prob': injection_prob
            })
            
        experiment_details['result'] = result
        results.append(experiment_details)
        
        # Print results
        print("\nResults:")
        print(result)
        print("="*100)
    
    return results

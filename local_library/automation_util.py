from typing import List
import local_library.training_pipeline as pipeline

"""
    Training Pipeline on Single Datasets
    Training Pipeline on Multiple Datasets
    Training Pipeline with character level injection
    Training Pipeline with Tokenization before tokenization

    Evaluating the model on Low resource languages
"""


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
    multiple_data_pipeline = pipeline.POSpipeline(
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


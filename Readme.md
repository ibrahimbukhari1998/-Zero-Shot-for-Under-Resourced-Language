# Cross-Lingual POS Tagging: XLM-R vs. Glot500

## Project Goal

Compare the performance of XLM-R and Glot500 when fine-tuned for POS tagging on a better-resourced language and then applied directly to a low-resource language without further training. Analyze the impact of subword tokenization on cross-lingual transfer.

## Results

### Cross-Lingual Transfer Performance: Mono-Lingual Fine-Tuning Dataset


| Language Pair | Model | Accuracy | F1 Score |
|--------------|---------|----------|-----------|
| English → Wolof | | | |
|                 | XLM-R | 37% | 34% |
|                 | Glot500 | 47% | 46% |
| Standard Arabic → Urdu | | | |
|                 | XLM-R | 17% | 11% |
|                 | Glot500 | 24% | 10% |
| French → Catalan | | | |
|                 | XLM-R | 90% | 89% |
|                 | Glot500 | 69% | 69% |
| Russian → Ukrainian | | | |
|                 | XLM-R | 91% | 90% |
|                 | Glot500 | 76% | 73% |


### Cross-Lingual Transfer Performance: Mono-Lingual Fine-Tuning Dataset with Noise Injection


| Language Pair | Model | 25% Noise |  | 75% Noise |  |
|--------------|--------|------------|------------|------------|------------|
| | | Accuracy | F1 Score | Accuracy | F1 Score |
| English → Wolof | | | | | |
| | XLM-R | 23% | 20% | 22% | 19% |
| | Glot500 | 42% | 41% |38% | 38% |
| Standard Arabic → Urdu | | | | | |
| | XLM-R | 24% | 11% | 23% | 11% |
| | Glot500 | 24% | 11% | 24% | 11% |
| French → Catalan | | | | | |
| | XLM-R | 46% | 46% | 46% | 45% |
| | Glot500 | 68% | 67% | 67% | 66% |
| Russian → Ukrainian | | | | | |
| | XLM-R | 54% | 53% | 51% | 50% |
| | Glot500 | 66% | 61% | 59% | 53% |


### Cross-Lingual Transfer Performance: Mono-Lingual Fine-Tuning with BPE/Word Dropout
| High Resource | Low Resource | Model   | Dropout Prob | Accuracy | F1 Score |
|--------------|--------------|---------|----------|----------|-----------|
| English      | Wolof        | XLM-R   |0.1      |   48%        |  30%    |
| English      | Wolof        | XLM-R   |0.2     |   41%        |  21%    |
| English      | Wolof        | Glot500 |0.1      |   41%  | 21%     |
| Standard Arabic | Urdu      | XLM-R   |0.1      | 31%      | 8%      |
| Standard Arabic | Urdu      | Glot500 |         | %      | %      |
| French       | Catalan      | XLM-R   |0.1      |  40%     |   21%  |
| French       | Catalan      | XLM-R   |0.2      |  35%     |   15%  |
| French       | Catalan      | Glot500 |0.1      | 35%      | 15%      |
| Russian      | Ukrainian    | XLM-R   |         | %      | %      |
| Russian      | Ukrainian    | Glot500 |         | %      | %      |

## Project Steps

1. **Model Fine-Tuning**
   - Use datasets from better-resourced languages within the Universal Dependencies framework

2. **Zero-Shot Transfer**
   - Apply the model to low-resource languages with existing POS annotated corpora (for evaluation purposes)

3. **Subword Tokenization Analysis**
   - Investigate how differences in tokenization between source and target languages impact the performance of zero-shot POS tagging

4. **Dataset Selection**
   - Choose high-resource languages (source)
   - Choose low-resource languages (target)
   - Potential language pairs (low-high):
     - Yoruba - English
     - Ukrainian - Russian
     - Catalan - French
     - Other Arabics - Arabic
     
     Celtic Tests:
     - Bretons - Irish, Welsh, Manx
     - Bretons - Irish, Welsh, French
     - Irish - English
     - Scottish Gaelic - Irish
     - Bretons - French
     - Bretons - Welsh
     - Welsh - Irish
   
   Universal Dependencies Dataset Size Per Language:
    - Irish: 7,699 sentences
    - Welsh: 12,000 sentences
    - Manx: 300 sentences
    - Bretons: 888 sentences
    - English: 47,643 sentences; 760,268 words, 751,522 tokens
    - Yoruba: 2,504 sentences
    - Ukranian: 7,060 sentences
    - Russian: 55,000 sentences
    - French: 52,000 sentences

5. **Data Preprocessing**
   - Clean and normalize data
   - Apply various tokenization techniques
   - Create multiple versions of source and target datasets

6. **Model Training**
   - Fine-tune XLM-R and Glot500 models using different versions of source datasets

7. **Evaluation**
   - Evaluate all versions of fine-tuned models on different versions of target datasets

8. **Results Compilation and Analysis**
   - Compile evaluation results in tabular form
   - Analyze results
   - If needed, repeat process with different source and target datasets

## Resources

### Papers

1. [XLM-R Paper](https://arxiv.org/pdf/1911.02116)
2. [Glot500 Paper](https://aclanthology.org/2023.acl-long.61.pdf)
3. [Zero-shot Transfer for POS Tagging](https://hal.science/hal-04381414v1/document)
4. [Tokenization Manipulation for Cross-Lingual Transfer](https://aclanthology.org/2023.vardial-1.5.pdf)
5. [Subwords in Multi-task Parsing](https://aclanthology.org/2024.lrec-main.215.pdf)
6. [Cross-Lingual Transfer for POS Tagging](https://aclanthology.org/2022.acl-long.529.pdf)

### Multilingual Models

1. **XLM-R**: Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. [DOI: 10.18653/v1/2020.acl-main.747](https://doi.org/10.18653/v1/2020.acl-main.747)

2. **Glot500**: ImaniGooghari, A., et al. (2023). *Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages*. [DOI: 10.18653/v1/2023.acl-long.61](https://doi.org/10.18653/v1/2023.acl-long.61)

3. Justification for Low-Resource Languages
- **EU Commission for Digital Language Equality**: <br>[https://www.europarl.europa.eu/cmsdata/257076/Giagkou.pdf](https://www.europarl.europa.eu/cmsdata/257076/Giagkou.pdf)

### Datasets for POS Tagging

- [Universal Dependencies](https://universaldependencies.org/)
- [Zenodo POS tagging resources](https://zenodo.org/communities/restaure/records?q=&f=subject%3ACorpus&f=subject%3APart-of-speech&l=list&p=1&s=10&sort=newest)

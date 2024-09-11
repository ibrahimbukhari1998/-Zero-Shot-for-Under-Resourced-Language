# Project Goal
- Comparison of performance of XLM-R and Glot500 models in zero-shot POS tagging scenarios.
- Evaluate how well these models transfer POS tagging capabilities from a source language to target low-resource languages
- Analyze the impact of subword tokenization on cross-lingual transfer

## Key Points:
- **Well-defined task**
- **Low-resourced languages**: To be determined (TBD)
- **Contact**: [dbernhard@unistra.fr](mailto:dbernhard@unistra.fr)

## Steps for the Project:
1. **Model Fine-Tuning**  
   - Use datasets from better-resourced languages within the Universal Dependencies framework.
   
2. **Zero-Shot Transfer**  
   - Apply the model to low-resource languages with existing POS annotated corpora (for evaluation purposes).
   
3. **Subword Tokenization Analysis**  
   - Investigate how differences in tokenization between source and target languages impact the performance of zero-shot POS tagging.


# Resources

## Papers

- **XLM-R Paper**: <br>[https://arxiv.org/pdf/1911.02116](https://arxiv.org/pdf/1911.02116)
- **Glot500 Paper**:<br> [https://aclanthology.org/2023.acl-long.61.pdf](https://aclanthology.org/2023.acl-long.61.pdf)
- **Zero-shot Transfer for POS Tagging**:<br>[https://hal.science/hal-04381414v1/document](https://hal.science/hal-04381414v1/document)
- **Does Manipulating Tokenization Aid Cross-Lingual Transfer? A Study on POS Tagging for Non-Standardized Languages**:<br>[https://aclanthology.org/2023.vardial-1.5.pdf](https://aclanthology.org/2023.vardial-1.5.pdf)
- **Bits and Pieces: Investigating the Effects of Subwords in Multi-task Parsing across Languages and Domains**:<br>[https://aclanthology.org/2024.lrec-main.215.pdf](https://aclanthology.org/2024.lrec-main.215.pdf)
- **Make the Best of Cross-Lingual Transfer: Evidence from POS Tagging with over 100 Languages**:<br> [https://aclanthology.org/2022.acl-long.529.pdf](https://aclanthology.org/2022.acl-long.529.pdf)

## Multilingual Models

- **Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., … Stoyanov, V.** (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 8440–8451. Online: Association for Computational Linguistics. [https://doi.org/10.18653/v1/2020.acl-main.747](https://doi.org/10.18653/v1/2020.acl-main.747)
- **ImaniGooghari, A., Lin, P., Kargaran, A. H., Severini, S., Sabet, M. J., Kassner, N., … Schütze, H.** (2023). *Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages*. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1082-1117. Toronto, Canada: Association for Computational Linguistics. [https://doi.org/10.18653/v1/2023.acl-long.61](https://doi.org/10.18653/v1/2023.acl-long.61)

## Datasets for POS Tagging

- Universal Dependencies: [https://universaldependencies.org/](https://universaldependencies.org/)
- Zenodo POS tagging resources: [https://zenodo.org/communities/restaure/records?q=&f=subject%3ACorpus&f=subject%3APart-of-speech&l=list&p=1&s=10&sort=newest](https://zenodo.org/communities/restaure/records?q=&f=subject%3ACorpus&f=subject%3APart-of-speech&l=list&p=1&s=10&sort=newest)

<h1 align="center">Natural Language Processing</h1>

#### Good reads
- [spaCy course](https://course.spacy.io)
- [spaCy blog](https://explosion.ai/blog)
- [Transformers explanation](http://www.peterbloem.nl/blog/transformers)
- [DistilBERT model by huggingface](https://medium.com/huggingface/distilbert-8cf3380435b5)
- [NLP infographic](https://www.analyticsvidhya.com/blog/2019/08/complete-list-important-frameworks-nlp/)

#### Libraries

| Library                                                                           | Description               |    |
|-----------------------------------------------------------------------------------|---------------------------|----|
| 🔤 [**spaCy**](https://numpy.org)                                                 | Industrial-Strength NLP   | ⭐ |
| 🤗 [**pytorch-transformers**](https://github.com/huggingface/pytorch-transformers)| 8 pretrained transformers | ⭐ |

# Embedings (Word2Vect)

# Deep learning models

🤗 Means availability (pretrained PyTorch implementation) on [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) package developed by huggingface.

| Model              | Creator         | Date      | Parameters   | Breif description         | 🤗 |
|--------------------|-----------------|-----------|--------------|---------------------------|----|
| **1st Transformer**| Google          | Jun. 2017 |              | Attention Is All You Need |    |
| **ULMFiT**         | Fast.ai         | Jan. 2018 |              | Regular LSTM              |    |
| **ELMo**           | AllenNLP        | Feb. 2018 | 94M          | Bidirectional LSTM        |    |
| **GPT**            | OpenAI          | Jun. 2018 | 110M         | Transformer on normal LM  | ✔ |
| **BERT**           | Google          | Oct. 2018 | 110M or 340M | Transformer on masked LM  | ✔ |
| **Transformer-XL** | Google/CMU      | Jan. 2019 | 128M or 151M |                           | ✔ |
| **XLM/mBERT**      | Facebook        | Jan. 2019 | 665M         | Multilingual LM           | ✔ |
| **Transf. ELMo**   | AllenNLP        | Jan. 2019 | 465M         |                           |    |
| **GPT-2**          | OpenAI          | Feb. 2019 | 1500M        | Good text generation      | ✔ |
| **ERNIE**          | Baidu research  | Apr. 2019 |              |                           |    |
| **XLNet**:         | Google/CMU      | Jun. 2019 | 340M         | BERT + Transformer-XL     | ✔ |
| **RoBERTa**        | Facebook        | Jul. 2019 | 125M or 355M | Optimized BERT            | ✔ |
| **MegatronLM**     | Nvidia          | Aug. 2019 | 8300M        | Too big                   |    |
| **DistilBERT**     | Hugging Face    | Aug. 2019 | 66M          | Compressed BERT           | ✔ |


  
- **Attention**: (Aug 2015)
  - Allows the network to refer back to the input sequence, instead of forcing it to encode all information into ane fixed-lenght vector.
  - Paper: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
  - [blog](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  - [attention and memory](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
- **1st Transformer**: (Google AI, jun. 2017)
  - Introduces the transformer architecture: Encoder with self-attention, and decoder with attention.
  - Surpassed RNN's State of the Art
  - Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - [blog](https://jalammar.github.io/illustrated-transformer).
- **ULMFiT**: (Fast.ai, Jan. 2018)
  - Regular LSTM Encoder-Decoder architecture with no attention.
  - Introduces the idea of transfer-learning in NLP:
    1. Take a trained tanguge model: Predict wich word comes next. Trained with Wikipedia corpus for example (Wikitext 103).
    2. Retrain it with your corpus data
    3. Train your task (classification, etc.)
  - Paper: [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
- **ELMo**: (AllenNLP, Feb. 2018)
  - Context-aware embedding = better representation. Useful for synonyms.
  - Made with bidirectional LSTMs trained on a language modeling (LM) objective.
  - Parameters: 94 millions
  - Paper: [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
  - [site](https://allennlp.org/elmo).
- **GPT**: (OpenAI, Jun. 2018)
  - Made with transformer trained on a language modeling (LM) objective.
  - Same as transformer, but with transfer-learning for ther NLP tasks.
  - First train the decoder for language modelling with unsupervised text, and then train other NLP task.
  - Parameters: 110 millions
  - Paper: [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - [*site*](https://blog.openai.com/language-unsupervised/), [*code*](https://github.com/openai/finetune-transformer-lm).
- **BERT**: (Google AI, oct. 2018)
  - Bi-directional training of transformer:
    - Replaces language modeling objective with "masked language modeling".
    - Words in a sentence are randomly erased and replaced with a special token ("masked").
    - Then, a transformer is used to generate a prediction for the masked word based on the unmasked words surrounding it, both to the left and right.
  - Parameters:
    - BERT-Base: 110 millions
    - BERT-Large: 340 millions
  - Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - [Official code](https://github.com/google-research/bert)
  - [blog](http://jalammar.github.io/illustrated-bert)
  - [fastai alumn blog](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)
  - [blog3](http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/)
  - [slides](https://nlp.stanford.edu/seminar/details/jdevlin.pdf)
- **Transformer-XL**: (Google/CMU, Jan. 2019)
  - Learning long-term dependencies
  - Resolved Transformer's Context-Fragmentation
  - Outperforms BERT in LM
  - Paper: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
  - [blog](https://medium.com/dair-ai/a-light-introduction-to-transformer-xl-be5737feb13)
  - [google blog](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)
  - [code](https://github.com/kimiyoung/transformer-xl).
- **XLM/mBERT**: (Facebook, Jan. 2019)
  - Multilingual Language Model (100 languages)
  - SOTA on cross-lingual classification and machine translation
  - Parameters: 665 millions
  - Paper: [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)
  - [code](https://github.com/facebookresearch/XLM/)
- **Transformer ELMo**: (AllenNLP, Jan. 2019)
  - Parameters: 465 millions
- **GPT-2**: (OpenAI, Feb. 2019)
  - Zero-Shot task learning
  - Coherent paragraphs of generated text
  - Parameters: 1500 millions
  - [Site](https://blog.openai.com/better-language-models/)
  - Paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **ERNIE** (Baidu research, Apr. 2019)
  - World-aware, Structure-aware, and Semantic-aware tasks
  - Continual pre-training
  - Paper: [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)
- **XLNet**: (Google/CMU, Jun. 2019)
  - Auto-Regressive methods for LM
  - Best both BERT + Transformer-XL
  - Parameters: 340 millions
  - Paper: [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
  - [code](https://github.com/zihangdai/xlnet/)
- **RoBERTa** (Facebook, Jul. 2019)
  - Facebook's improvement over BERT
  - Optimized BERT's training process and hyperparameters
  - Parameters:
    - RoBERTa-Base: 125 millions
    - RoBERTa-Large: 355 millions
  - Trained on 160GB of text
  - Paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- **MegatronLM** (Nvidia, Aug. 2019)
  - Too big
  - Parameters: 8300 millions
- **DistilBERT** (Hugging Face, Aug. 2019)
  - Compression of BERT with Knowledge distillation (teacher-student learning)
  - A small model (DistilBERT) is trained with the output of a larger model (BERT)
  - Comparable results to BERT using less parameters
  - Parameters: 66 millions

#### Libraries

- [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) by huggingface
  - Contains PyTorch implementations with pre-trained weights for the following models:
    - BERT
    - GPT
    - GPT-2
    - Transformer-XL
    - XLNet
    - XLM
    - RoBERTa
    - DistilBERT
- **StandfordNLP**
  - Neural models for text precessing
  - 53 human languages

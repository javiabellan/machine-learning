<h1 align="center">Natural Language Processing</h1>

#### Good reads
- Fast.ai NLP course: [playlist](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9)
- [spaCy course](https://course.spacy.io)
- [spaCy blog](https://explosion.ai/blog)
- [Transformers explanation](http://www.peterbloem.nl/blog/transformers)
- [DistilBERT model by huggingface](https://medium.com/huggingface/distilbert-8cf3380435b5)
- [NLP infographic](https://www.analyticsvidhya.com/blog/2019/08/complete-list-important-frameworks-nlp/)
- [BERT, RoBERTa, DistilBERT, XLNet. Which one to use?](https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8)

## NLP Applications


| Application                           | Description                                                               | Type |
|---------------------------------------|---------------------------------------------------------------------------|------|
| 🏷️ **Part-of-speech tagging (POS)**   | Identify if each word is a noun, verb, adjective, etc.                      | 🔤 |
| 📍 **Named entity recognition (NER)** | Identify names, organizations, locations, medical codes, time, etc.         | 🔤 |
| 🔍 **Text categorization**            | Identify topics present in a text (sports, politics, etc).                  | 🔤 |
| ❓ **Question answering**             | Answer questions of a given text (SQuAD dataset).                           | 💭 |
| 👍🏼 👎🏼 **Sentiment analysis**          | Possitive or negative comment/review classification.                       | 💭 |
| 🔮 **Language modeling**              | Predict the next word (or character) in a document.                        | 💭 |
| 📗→📄 **Summarization**               | Crate a short version of a text.                                           | 💭 |
| 🈯→🆗 **Translation**                 | Translate into a different language.                                       | 💭 |
| 🆓→🆒 **Dialogue bot**                | Interact in a conversation.                                                | 💭 |
| 💁🏻→🔠 **Speech recognition**          | Speech to text. See [AUDIO](/AUDIO.md) cheatsheet                          | 🗣️ |
| 🔠→💁🏻 **Speech generation**           | Text to speech. See [AUDIO](/AUDIO.md) cheatsheet                          | 🗣️ |

- 🔤: Natural Language Processing (NLP)
- 💭: Natural Language Understanding (NLU)
- 🗣️: Speech and sound (speak and listen)

## NLP pipeline

1. **Preprocess**
   - **Tokenization**: Split the text into sentences and the sentences into words.
   - **Lowercasing**: Usually done in **Tokenization**
   - **Punctuation removal**: Remove words like `.`, `,`, `:`. Usually done in **Tokenization**
   - **Stopwords removal**: Remove words like `and`, `the`, `him`. The list of stopword is smaller (or empty) nowadays. 
   - **Lemmatization**: Verbs to root form: `organizes`, `will organize` `organizing` → `organize` This is better.
   - **Stemming**: Nouns to root form: `democratic`, `democratization` → `democracy`. This is faster.
2. **Extract features**
   - **Tf-idf**: sklearn.feature_extraction.text.TfidfVectorizer
   - **N-gram**: Probability of N words together
   - **Bag of Words (BoW)**: Count the number of occurrences of each word in a text.
   - **Word embeddings**: Pre-trained Word2Vec model.
3. **Build model**
   - Linear algebra/matrix decomposition
     - Latent Semantic Analysis (LSA) that uses Singular Value Decomposition (SVD).
     - Non-negative Matrix Factorization (NMF)
     - Latent Dirichlet Allocation (LDA)
   - Neural nets
     - Recurrent NNs (LSTM, GRU)
     - Transformer (GPT, BERT, ...)
   - Hidden Markov Models


#### Others
- **Regular expressions**: (Regex) Find patterns.
- **Parse trees**: Syntax od a sentence




## NLP Python Packages


| Package                                          | Description                                                               | Type |
|:------------------------------------------------:|---------------------------------------------------------------------------|------|
| <img src="img/logo/spacy.png" height="40">       | Parse trees, execelent tokenizer (8 languages)                            | 🔤 |
| <img src="img/logo/fastai.png" height="50">      | Fast.ai NLP: ULMFiT fine-tuning                                           | 🔤 |
| <img src="img/logo/pytorch.png" height="30">     | TorchText (Pytorch subpackage)                                            | 🔤 |
| <img src="img/logo/fasttext.png" height="50">    | Word vector representations and sentence classification (157 languages)   | 🔤 |
| <img src="img/logo/huggingface.png" height="50"> | pytorch-transformers: 8 pretrained Pytorch transformers                   | 🔤 |
| <img  src="img/logo/spacy.png" height="30">+<img src="img/logo/huggingface.png" height="40"> | SpaCy + pytorch-transformers  | 🔤 |
| <h3>fast-bert</h3>                               | Super easy library for BERT based models                                  | 🔤 |
| <img src="img/logo/stanfordnlp.jpg" height="50"> | Pretrained models for 53 languages                                        | 🔤 |
| <h3>NLTK</h3>                                    | Very broad NLP library. Not SotA.                                         | 🔤 |
| <img src="img/logo//gensim.jpg" height="30">     | Semantic analysis, topic modeling and similarity detection.               | 🔤 |
| <h3>PyText</h3>                                  |                                                                           | 🔤 |
| <h3>SentencePiece</h3>                           | Unsupervised text tokenizer by Google                                     | 🔤 |


## Topic modeling
[Topic modeling with gensim](https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925)

You have a collection of documents (texts)

1. Preprocess text
2. Convert to Bag of Words (BoW) each document.
3. Running LDA (Latent Dirichlet Allocation)


<h1><img height="50" src="img/logo/spacy.png"></h1>

### Installation

```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md
```

### Usage

```python
import spacy

nlp = spacy.load("en_core_web_sm")  # Load English small model
nlp = spacy.load("es_core_news_sm") # Load Spanish small model without Word2Vec
nlp = spacy.load('es_core_news_md') # Load Spanish medium model with Word2Vec


text = nlp("Hola, me llamo Javi")   # Text from string
text = nlp(open("file.txt").read()) # Text from file


spacy.displacy.render(text, style='ent', jupyter=True)  # Display text entities
spacy.displacy.render(text, style='dep', jupyter=True)  # Display word dependencies
```

### Word2Vect

`es_core_news_md` has 534k keys, 20k unique vectors (50 dimensions)

```python
coche = nlp("coche")
moto  = nlp("moto")
print(coche.similarity(moto)) # Similarity based on cosine distance

coche[0].vector      # Show vector
```


<h1><img height="50" src="img/logo/fastai.png"></h1>


### Videos
1. [What is NLP?](https://youtu.be/cce8ntxP_XI)
2. [Topic Modeling with SVD & NMF](https://youtu.be/tG3pUwmGjsc)
3. [Topic Modeling & SVD revisited](https://youtu.be/lRZ4aMaXPBI)
4. [Sentiment Classification with Naive Bayes](https://youtu.be/hp2ipC5pW4I)
5. [Sentiment Classification with Naive Bayes & Logistic Regression, contd.](https://youtu.be/dt7sArnLo1g)
6. [Derivation of Naive Bayes & Numerical Stability](https://youtu.be/z8-Tbrg1-rE)
7. [Revisiting Naive Bayes, and Regex](https://youtu.be/Q1zLqfnEXdw)
8. [Intro to Language Modeling](https://youtu.be/PNNHaQUQqW8)
9. [Transfer learning](https://youtu.be/5gCQvuznKn0)
10. [ULMFit for non-English Languages](https://youtu.be/MDX_x6rKXAs)
11. [Understanding RNNs](https://youtu.be/l1rlFh0PmZw)
12. [Seq2Seq Translation](https://youtu.be/IfsjMg4fLWQ)
13. [Word embeddings quantify 100 years of gender & ethnic stereotypes](https://youtu.be/boxV8Od4jqQ)
14. [Text generation algorithms](https://youtu.be/3oEb_fFmPnY)
15. [Implementing a GRU](https://youtu.be/Bl6WVj6wQaE)
16. [Algorithmic Bias](https://youtu.be/pThqge9QDn8)
17. [Introduction to the Transformer](https://youtu.be/AFkGPmU16QA)
18. [The Transformer for language translation](https://youtu.be/KzfyftiH7R8)
19. [What you need to know about Disinformation](https://youtu.be/vbva2RN-rbQ)

# Deep learning models

🤗 Means availability (pretrained PyTorch implementation) on [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) package developed by huggingface.

| Model              | Creator         | Date      | Parameters   | Breif description         | 🤗 |
|--------------------|:---------------:|-----------|:------------:|---------------------------|----|
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
| **XLNet**:         | Google/CMU      | Jun. 2019 | 110M or 340M | BERT + Transformer-XL     | ✔ |
| **RoBERTa**        | Facebook        | Jul. 2019 | 125M or 355M | Optimized BERT            | ✔ |
| **MegatronLM**     | Nvidia          | Aug. 2019 | 8300M        | Too big                   |    |
| **DistilBERT**     | Hugging Face    | Aug. 2019 | 66M          | Compressed BERT           | ✔ |
| **[MiniBERT](https://arxiv.org/abs/1909.00100)**  | Google   | Aug. 2019 |  | Compressed BERT  |  |

  
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
  - Paper: [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
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

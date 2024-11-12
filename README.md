# Slot Tagging with CNN vs RNN Variants

**Darian Lee**  
UC Santa Cruz  
Sunnyvale, CA, 94087  
[daeilee@ucsc.edu](mailto:daeilee@ucsc.edu)

---

## Abstract
This paper examines the strengths and weaknesses of various models for slot tagging of film-related utterances. I compare LSTM and GRU variants to CNNs. I find that the best model for this task is a standard CNN with no dilated layers. The final model consists of three convolutional layers with a kernel size of 3. The ReLU activation function was used to introduce non-linearity, and a dropout rate of 0.4 was applied to prevent overfitting. The model achieved a final testing macro F1 score of 0.78. I believe obtaining a significantly higher macro F1 on this dataset is unlikely due to noisy labels and insufficient training data for certain classes.

## Introduction
The goal of this project was to develop a model for slot tagging movie-related queries using IOB tagging. IOB tagging is a technique used to identify key information in an utterance by labeling each word in a sequence as either `O` (indicating the word is "outside" of an important phrase, i.e., irrelevant), `B_class` (indicating the word marks the beginning of an important phrase related to a specific class, e.g., `B_movie` for the start of a movie title), or `I_class` (indicating the word is inside an important phrase related to that class). Slot tagging is a crucial intermediate task in natural language processing, fundamental for applications such as question answering, information extraction, and intent classification. This project aimed to improve the efficient retrieval of film information based on specific queries and has potential applications in areas like film recommendation systems, customer support, and similar domains. The project utilized supervised learning on a labeled dataset sourced from Kaggle, authored by anonymous contributors.

| **ID** | **Utterances**                           | **IOB Slot Tags**                                   |
|--------|------------------------------------------|----------------------------------------------------|
| 1      | who plays luke on star wars new hope     | O O B_char O B_movie I_movie I_movie I_movie       |
| 2      | show credits for the godfather           | O O O B_movie I_movie                              |

*Table 1: Sample rows from the training set*

## Models and Experimentation
In this section, I will outline the architecture and motivations behind my three most distinct and developed model classes. The first class contains RNN variations, such as LSTMs or GRUs. The second class incorporates both LSTMs and GRUs with bidirectionality, while the final class consists of CNN variants. For each model class, I will describe its motivation from research, the performance of the best model, the experimentation leading to it, and the potential shortcomings of the model class. The training data was split such that 10% served as validation data. I measured the macro F1 score for model selection, as it provides a balanced evaluation across all classes, which is essential for ensuring performance on underrepresented tags in this slot-tagging task.

### Class 1: LSTM and GRU

#### Motivation
The motivation for using LSTMs and GRUs to develop a baseline model is based on a literature review suggesting that they outperform simple RNNs and CRFs in slot-tagging tasks [Yao et al., 2014]. While CRFs can achieve high accuracy, they often rely on hand-crafted features, which increase model complexity and introduce challenges in feature engineering. Additionally, CRFs suffer from longer training times due to their reliance on complex dependencies between output labels. Standard RNNs are also less effective for slot-tagging because they struggle with learning long-range dependencies due to the vanishing gradient problem. Thus, I chose LSTMs and GRUs, which handle long-range dependencies through their gating mechanisms and learn from raw data without the need for hand-crafted features, making them more efficient and accurate for slot-tagging.

#### Experimentation and Performance
All models in this class, whether using LSTMs or GRUs, demonstrated similar F1 scores around 0.69–0.70. After grid-searching many hidden embedding and layer combinations, the highest-performing model was a 4-layer LSTM with a hidden dimension of 1000 and a dropout rate of 0.4, achieving a testing F1 score of 0.705. For each model in this class, I used pretrained embeddings from `glove-twitter-200` as the initial weights for the embedding layer. Model selection was made using the highest macro F1 score on the test set.

#### Potential Shortcomings
LSTM-based models suffer from longer training times and higher RAM usage, limiting their maximum size on devices without GPUs. Additionally, their ability to capture long-range dependencies may lead to overfitting or be unnecessary, as the utterances in this dataset are short. Similarly, GRUs may underfit tasks requiring fine-grained contextual distinctions, as their simplified gating mechanism limits the model’s ability to capture subtle dependencies between tokens. This trade-off in simplicity can result in missed nuances for tags with intricate sequential relationships, even though GRUs are more computationally efficient.

| **Model Type**     | **Average Testing Macro F1** |
|--------------------|------------------------------|
| GRU-based models   | 0.683                        |
| LSTM-based models  | 0.704                        |

*Table 2: Average testing macro F1 results based on 3 GRU models and 3 LSTM models.*

![Labeled LSTM diagram showing its complex information storage system](labelled_lstm.png)
*Figure 1: labeled LSTM diagram showing its complex information storage system*

![Labeled GRU diagram showing its simplified gating system](GRU.png)
*Figure 2: labeled GRU diagram showing its simplified gating system*

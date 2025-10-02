# Project #1  
**Natural Language Processing Chatbot with Intent Recognition and Response Generation**

## Authors
- Faid Faisal  
- Paddy Zheng  

**Date:** September 9, 2025  

---

## Project Overview
This project involved designing and implementing a **transformer-based chatbot** trained on a combination of conversational datasets and coding datasets. The chatbot is capable of both **intent recognition** and **contextually appropriate response generation**.  

The training process included:
- Text preprocessing and vectorization  
- Transformer model training with attention mechanisms, feedforward layers, and layer normalization  

---

## Introduction
The goal of this project was to build a natural language processing (NLP) chatbot capable of:
- Understanding user intents  
- Generating contextually relevant responses  

Key objectives:
- Explore **core NLP concepts**  
- Implement **transformer-based architectures**  
- Design chatbot logic with **80% accuracy in intent recognition**  

---

## Methodology

### Data Preparation & Preprocessing
We selected pre-existing conversational and coding datasets. Preprocessing steps included:
- Tokenization and cleaning  
- Stopword removal  
- Vectorization (TF-IDF and CountVectorizer)  

### Transformer-Based Model Training
The chatbot was trained using a transformer architecture with:
- Scaled dot-product & multi-headed attention  
- Positional encoding  
- Feedforward layers and layer normalization  
- **CrossEntropyLoss** and **Adam optimizer**  

The trained model parameters can be saved/loaded for integration.  

### Chatbot Logic
- Intent prediction via the trained transformer  
- Pre-written response mapping with random variation  
- **DialoGPT fallback** for unrecognized intents  
- Optional follow-up prompts & basic memory for improved flow  

### Optimization & Interface
- Enhanced response variety  
- Conversational memory  
- Initial steps toward a web-based user interface  

---

## Results
- **Accuracy:** ~80% in intent recognition  
- **Responses:** Generated coherent, contextually appropriate answers  
- **Design:** Modular, allowing future improvements in memory and UI integration  

---

## Conclusion
This project provided hands-on experience with:
- NLP preprocessing  
- Transformer model training  
- Chatbot design and logic  

The chatbot prototype successfully demonstrated the ability to **process text, classify intents, and generate responses**. The codebase is documented and organized for future development and research.  

---

## Acknowledgements
We gratefully acknowledge the assistance of **ChatGPT** in providing guidance, debugging insights, and implementation strategies during the development of this project.  

---

## References
1. Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Cornell Movie-Dialogs Corpus. Cornell University. Retrieved September 8, 2025, from [https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)  
2. Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). DailyDialog. Retrieved September 8, 2025, from [http://yanran.li/dailydialog](http://yanran.li/dailydialog)  
3. Puri, R., Kung, D., Janssen, G., Zhang, W., Domeniconi, G., Zolotov, V., ... Chen, T. (2021). Project CodeNet: A large-scale AI for code dataset for learning a diversity of coding tasks [Dataset]. IBM. [https://github.com/IBM/Project_CodeNet](https://github.com/IBM/Project_CodeNet)  
4. Karthik Gopalakrishnan, Behnam Hedayatnia, Qinlang Chen, Anna Gottardi, Sanjeev Kwatra, Anu Venkatesh, Raefer Gabriel, Dilek Hakkani-Tür. Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations. Retrieved September 8, 2025, from [https://github.com/alexa/Topical-Chat](https://github.com/alexa/Topical-Chat) 

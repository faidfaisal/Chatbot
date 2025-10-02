# Transformer-Based Conversational Chatbot

## Project Goals

[cite_start]This project presents the design and implementation of a transformer-based chatbot trained on conversational datasets[cite: 2]. [cite_start]Our system combines both intent recognition and response generation[cite: 3].

The primary goals included:
* [cite_start]Getting hands on experience with transformers[cite: 10].
* [cite_start]Exploring data preprocessing and intent recognition[cite: 11].
* [cite_start]Demonstrating the integration of NLP and generative AI into an interactive chatbot[cite: 12].

[cite_start]Future plans include expanding this project to allow individuals to input professor recordings, for the chatbot to train on and create "virtual office hours" which can imitate professors' way of talking, specifically for coding based classes[cite: 13].

## Methodology

### Data Preparation and Datasets

[cite_start]We selected pre-existing datasets including conversational datasets and coding datasets[cite: 16]:

| Dataset | Number of Samples | Avg. Training Length\* | Purpose | License Used |
| :--- | :--- | :--- | :--- | :--- |
| **Cornell Movie-Dialog Corpus** | [cite_start]$\sim$80k [cite: 27] | [cite_start]$\sim$13h per 8 epoches [cite: 27] | [cite_start]Movie like conversations [cite: 27] | Not Officially Licensed |
| **DailyDialog** | [cite_start]$\sim$160k [cite: 27] | [cite_start]$\sim$26h per 8 epoches [cite: 27] | [cite_start]Daily Text based dialogue [cite: 27, 28] | CC BY-NC-SA 4.0 |
| **Project CodeNet** | [cite_start]$\sim$5.5 million [cite: 28] | [cite_start]N/A (Code Only) [cite: 28] | [cite_start]Programming [cite: 28] | CDLA-Permissive 2.0 |

[cite_start]*\*Training statistics based on a 3080 TI GPU[cite: 27, 29].*

[cite_start]**Note on CodeNet:** Although this dataset was used, we do not plan on training on all of the coding languages present within the dataset[cite: 20]. [cite_start]We specifically focused on C and C++ for our model[cite: 21].

**Preprocessing Steps**:
* [cite_start]Tokenization and cleaning[cite: 23].
* [cite_start]Stopword removal[cite: 24].
* [cite_start]Vectorization using TfidfVectorizer and CountVectorizer[cite: 25].

### Model Architecture and Logic

[cite_start]We implemented a transformer architecture with[cite: 31]:
* [cite_start]Scaled dot-product attention and multi-headed attention[cite: 32].
* [cite_start]Positional encoding[cite: 33].
* [cite_start]Feedforward layers and layer normalization[cite: 34].

The model was trained using:
* [cite_start]CrossEntropyLoss and Adam optimizer[cite: 35].
* [cite_start]Trained to classify user intents and save/load the trained parameters for seamless integration[cite: 36].

**Chatbot Logic**:
[cite_start]User inputs were processed through the trained transformer to predict intent, with responses generated using[cite: 37, 38]:
* [cite_start]Pre-written response mapping with random selection for variety[cite: 39].
* [cite_start]DialoGPT fallback for unrecognized intents[cite: 40].
* [cite_start]Optional follow-up prompts and basic memory for improved conversation flow[cite: 41].

### Optimization and Interface
Enhancements included:
* [cite_start]Model saving and loading mechanisms using PyTorch[cite: 44].
* [cite_start]Conversational memory[cite: 45].
* [cite_start]Initial steps toward a user interface, with potential for web-based deployment[cite: 46].
* [cite_start]Ability to choose which dataset to train on, or a mix of datasets, with adjusted weights[cite: 47].

## Results

The chatbot achieved:
* [cite_start]Accuracy of around $\sim$80% on intent recognition[cite: 50].
* [cite_start]Using the trained model, it generates an adaptive response in accordance with the user prompt[cite: 51].

[cite_start]We noted that some results were incoherent, highlighting the need for more intensive training and improvements in our algorithms[cite: 52].

## Remarks and Future Plans

[cite_start]Although we are optimistic about the potential of our program, we still believe that much fine tuning is needed[cite: 58].

Future improvements may include:
* [cite_start]Enhancing memory for long term context recognition[cite: 60].
* [cite_start]Larger conversation dataset implementation[cite: 61].
* [cite_start]Experiment with lower learning rates and adaptive optimizers to reduce overfitting[cite: 62].
* [cite_start]Developing a web UI for general use[cite: 62].
* [cite_start]Training our model on the CodeNet C/C++ datasets, allowing for our model to understand coding syntax, errors in code, and how to generate code for specific prompts given by the user[cite: 63].

## Dataset Licenses & Credits

### DailyDialog
- **License:** **CC BY-NC-SA 4.0**
- [cite_start]**Terms:** You may remix, transform, or build upon this dataset, but must distribute derivatives under the same license and only for **non-commercial** use[cite: 3, 71, 72].
- [cite_start]**Source:** http://yanran.li/dailydialog [cite: 72]
- **Full license text:** [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Citation:** Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). [cite_start]*DailyDialog*[cite: 71].

---

### Topical-Chat
- **License:** **CDLA-Sharing-1.0**
- **Terms:** You must share any new versions or derivatives of the dataset you create under the same license (copyleft for data).
- **Source:** https://github.com/alexa/Topical-Chat
- **Full license text:** [https://cdla.dev/sharing-1-0/](https://cdla.dev/sharing-1-0/)
- **Citation:** Gopalakrishnan, K., Hedayatnia, B., Chen, Q., Gottardi, A., Kwatra, S., Venkatesh, A., Gabriel, R., Hakkani-Tür, D. *Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations*.

---

### Cornell Movie-Dialogs Corpus
- **License / Terms:** Not officially licensed; use is typically restricted to **research/educational purposes**. [cite_start]Confirm permissions for commercial use[cite: 17, 69, 70].
- [cite_start]**Source:** http://www.cs.cornell.edu/~cristian/Cornell\_Movie-Dialogs\_Corpus.html [cite: 70]
- **Citation:** Danescu-Niculescu-Mizil, C., & Lee, L. (2011). [cite_start]*Cornell Movie-Dialogs Corpus*[cite: 69].

---

### Project CodeNet
- **License:** **CDLA-Permissive 2.0**
- [cite_start]**Terms:** You may use, modify, and redistribute the dataset (including commercially) as long as you include the **full license text** with any redistribution of the dataset[cite: 73, 74].
- [cite_start]**Source:** https://github.com/IBM/Project\_CodeNet [cite: 74]
- **Full license text:** [https://cdla.dev/permissive-2-0/](https://cdla.dev/permissive-2-0/)
- **Citation:** Puri, R., Kung, D., Janssen, G., Zhang, W., Domeniconi, G., Zolotov, V.,... Chen, T. (2021). [cite_start]*Project CodeNet: A large-scale Al for code dataset for learning a diversity of coding tasks*[cite: 73].

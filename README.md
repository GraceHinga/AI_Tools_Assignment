# 🧠 AI Tools Assignment — Full Project

## Overview

This project demonstrates the **end-to-end AI workflow** using multiple tools and frameworks. It covers:

1. **Theory** — Conceptual understanding of AI frameworks.  
2. **Practical Implementation** — Hands-on projects in Classical ML, Deep Learning, and NLP.  
3. **Ethics & Optimization** — Model fairness, debugging, and optional deployment.

---

## 📂 Project Structure

```

AI_Tools_Assignment/
├── Part1_Theory/
│   └── AI_Tools_Assignment_Part1_Theory.pdf
├── Part2_Practical/
│   ├── iris_classifier.ipynb
│   ├── mnist_cnn_tf.ipynb
│   ├── spacy_ner_sentiment.ipynb
│   ├── mnist_app.py               ← Optional Streamlit app (local)
│   └── images/
│       ├── iris_pairplot.png
│       ├── iris_confusion_matrix.png
│       ├── iris_decision_tree.png
│       ├── mnist_samples.png
│       ├── mnist_training_curves.png
│       ├── mnist_confusion_matrix.png
│       ├── mnist_predictions.png
│       └── mnist_streamlit_demo.png  ← Screenshot of app
├── Part3_Ethics/
│   ├── AI_Tools_Assignment_Part3_Ethics.ipynb
│   └── ethics_reflection.pdf
└── README.md

````

---

## 🧩 Part 1 — Theory

Contains the conceptual understanding of AI frameworks, including:

- **TensorFlow vs PyTorch**  
- **Scikit-learn vs TensorFlow**  
- **spaCy NLP advantages**  
- Use cases of **Jupyter Notebook**

> See: `Part1_Theory/AI_Tools_Assignment_Part1_Theory.pdf`

---

## 🧩 Part 2 — Practical Implementation

### 1. Classical ML: Iris Dataset 🌸

- Notebook: `iris_classifier.ipynb`  
- Model: Decision Tree  
- Visualizations:
  - Pairplots  
  - Confusion matrix  
  - Decision tree diagram  

### 2. Deep Learning: MNIST Dataset ✋

- Notebook: `mnist_cnn_tf.ipynb`  
- Model: CNN for handwritten digit classification  
- Saved plots in `images/`:
  - Sample digits  
  - Training curves  
  - Confusion matrix  
  - Sample predictions  
- Optional Streamlit app:
  - `mnist_app.py` (local)
  - **Screenshot:**  

![MNIST Streamlit App](Part2_Practical/images/mnist_streamlit_demo.png)

### 3. NLP: Amazon Reviews 📝

- Notebook: `spacy_ner_sentiment.ipynb`  
- Tasks:
  - Named Entity Recognition (NER) using spaCy  
  - Sentiment Analysis using TextBlob/VADER  
  - Visualization of entity extraction and sentiment distribution  

---

## 🧩 Part 3 — Ethics & Optimization ⚖️

- Ethical considerations and bias analysis for MNIST & Amazon Reviews models  
- Debugging challenges and fixes for TensorFlow code  
- Reflection documented in: `Part3_Ethics/ethics_reflection.pdf`  
- Bonus Task: Streamlit app screenshot included for demonstration  

---

## 🛠️ Dependencies

- Python 3.x  
- Packages:
```text
tensorflow
torch
torchvision
scikit-learn
spacy
pandas
numpy
matplotlib
seaborn
textblob
streamlit
pillow
````

* SpaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## 📌 How to Run (Local)

1. Activate your virtual environment:

```bash
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open notebooks for hands-on exploration:

```bash
jupyter notebook
```

4. (Optional) Run Streamlit app locally:

```bash
streamlit run Part2_Practical/mnist_app.py
```

---

## 📝 Notes

* All images and visualizations are saved in `Part2_Practical/images/`
* The Streamlit app is optional; a screenshot is included for Part 3 bonus
* All notebooks are self-contained and documented for clarity

---

## 📚 References

* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* PyTorch Documentation: [https://pytorch.org/](https://pytorch.org/)
* Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
* spaCy Documentation: [https://spacy.io/](https://spacy.io/)
* MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---

import re
import numpy as np
import networkx as nx
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk.cluster.util import cosine_distance
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import scrolledtext
from PyPDF2 import PdfReader
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
eff_factor = 60
# Initialize spaCy
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

class EnhancedTextRank:
    def _init_(self, document):
        self.document = document
        self.sentences = []
        self.raw_sentences = []
        self.set_raw_sentences()
        self.break_into_tokens()
        self.remove_stop_words()
        self.lemmatize_words()
        self.rank_sentences()

    def set_raw_sentences(self):
        document = re.sub("<[^>]+>", "", self.document).strip()
        self.raw_sentences = sent_tokenize(document)

    def break_into_tokens(self):
        document = re.sub("<[^>]+>", "", self.document).strip()
        document = document.lower()
        sentences = sent_tokenize(document)
        for s in sentences:
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\w*\d\w*', '', s)
            self.sentences.append(s)
        self.words = [word_tokenize(word) for word in self.sentences]

    def remove_stop_words(self):
        english_stop = set(stopwords.words('english'))
        self.words = [[w for w in word if w not in english_stop] for word in self.words]

    def lemmatize_words(self):
        self.words = [[lemmatizer.lemmatize(w) for w in word] for word in self.words]

    def sentence_similarity(self, sentence1, sentence2):
        sentence1 = [word for word in sentence1]
        sentence2 = [word for word in sentence2]
        all_words = list(set(sentence1 + sentence2))
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        for w in sentence1:
            vector1[all_words.index(w)] += 1
        for w in sentence2:
            vector2[all_words.index(w)] += 1
        return 1 - cosine_distance(vector1, vector2)

    def similarity_matrix(self):
        similarity_matrix = np.zeros((len(self.words), len(self.words)))
        for index1 in range(len(self.words)):
            for index2 in range(len(self.words)):
                if index1 == index2:
                    continue
                similarity_matrix[index1][index2] = self.sentence_similarity(self.words[index1], self.words[index2])
        return similarity_matrix

    def rank_sentences(self):
        similarity_graph = nx.from_numpy_array(self.similarity_matrix())
        score = nx.pagerank(similarity_graph, alpha=0.85)
        ranked_sentence = sorted(((score[i], s) for i, s in enumerate(self.raw_sentences)), reverse=True)
        self.ranked_sentence = ranked_sentence

    def summarize(self, top_sentence=4):
        summarized = ""
        for i in range(top_sentence):
            summarized += str(self.ranked_sentence[i][1]) + " "
        return summarized.strip()

def summarize_with_enhanced_textrank_cosine(text, num_sentences=3):
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    sentence_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_graph, weight='weight')
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summarized_text = ' '.join([sentence for score, sentence in ranked_sentences[:num_sentences]])
    return summarized_text

def summarize_with_enhanced_tfidf(text, num_sentences=3):
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:][::-1]]
    return ' '.join(ranked_sentences)

def summarize_with_bert(text, num_sentences=3):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = sent_tokenize(text)
    sentence_embeddings = model.encode(sentences)
    similarity_matrix = np.inner(sentence_embeddings, sentence_embeddings)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ' '.join([ranked_sentences[i][1] for i in range(num_sentences)])

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        return text

def calculate_efficiency(text):
    doc = nlp(text)
    tokens = len(doc)
    sentences = len(list(doc.sents))
    words = len([token.text for token in doc if token.is_stop != True and token.is_punct != True])
    return round((words/tokens) * (tokens/sentences) + eff_factor , 2)

app = Flask(_name_)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        num_sentences = int(request.form.get("num_sentences", 5))
        file_txt = request.files.get("file_txt")
        file_pdf = request.files.get("file_pdf")
        if file_txt:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_txt.filename))
            file_txt.save(file_path)
            with open(file_path, 'r') as file:
                input_text = file.read()
        if file_pdf:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_pdf.filename))
            file_pdf.save(file_path)
            input_text = extract_text_from_pdf(file_path)

        textrank_summary = EnhancedTextRank(input_text).summarize(num_sentences)
        textrank_cosine_summary = summarize_with_enhanced_textrank_cosine(input_text, num_sentences)
        tfidf_summary = summarize_with_enhanced_tfidf(input_text, num_sentences)
        bert_summary = summarize_with_bert(input_text, num_sentences)

        summaries = {
            "textrank": textrank_summary,
            "textrank_cosine": textrank_cosine_summary,
            "tfidf": tfidf_summary,
            "bert": bert_summary
        }

        # Calculate efficiencies
        efficiencies = {}
        for method, summary in summaries.items():
            efficiency = calculate_efficiency(summary)
            efficiencies[method] = efficiency

        # Determine the best summary method based on highest efficiency
        best_summary_method = max(efficiencies, key=efficiencies.get)
        best_summary = summaries[best_summary_method]
        final_efficiency = efficiencies[best_summary_method]

        return render_template(
            "result.html",
            summaries=summaries,
            efficiencies=efficiencies,
            best_summary_method=best_summary_method,
            best_summary=best_summary,
            final_efficiency=final_efficiency
        )
    return render_template("index.html")


if _name_ == "_main_":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

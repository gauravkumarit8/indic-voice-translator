import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
 
class GlossaryMatcher:
    def __init__(self, csv_path="glossary.csv"):
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
 
        if not os.path.exists(csv_path):
            self.indexes = {}
            self.data = {}
            return
 
        self.df = pd.read_csv(csv_path)
        self.indexes = {}
        self.data = {}
 
        lang_pairs = ['hi-kn', 'hi-te', 'kn-hi', 'te-hi', 'hi-en', 'en-hi']
 
        for pair in lang_pairs:
            src_col, tgt_col = pair.split('-')
            if src_col not in self.df.columns or tgt_col not in self.df.columns:
                continue
 
            subset = self.df[[src_col, tgt_col]].dropna()
            if len(subset) < 1:
                continue
 
            embeds = self.embedder.encode(subset[src_col].tolist(), show_progress_bar=False)
            index = faiss.IndexFlatL2(embeds.shape[1])
            index.add(np.array(embeds).astype('float32'))
 
            self.indexes[pair] = index
            self.data[pair] = subset.values.tolist()
 
    def replace(self, text, src, tgt):
        pair = f"{src}-{tgt}"
        if pair not in self.indexes:
            return text
        if not text or len(text.strip()) < 2:
            return text
 
        words = text.split()
        new_words = []
        index = self.indexes[pair]
        data = self.data[pair]
 
        for w in words:
            emb = self.embedder.encode([w], show_progress_bar=False)
            D, I = index.search(emb.astype('float32'), 1)
            if D[0][0] < 0.45:
                new_words.append(data[I[0][0]][1])
            else:
                new_words.append(w)
 
        return " ".join(new_words)
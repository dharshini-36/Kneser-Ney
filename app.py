import streamlit as st
from collections import defaultdict
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

N = 5
DISCOUNT = 0.75

# ==========================================
# SAMPLE CORPUS
# (Replace this with a corpus.txt later if needed)
# ==========================================

corpus = [
    "best places to visit in india",
    "best places to visit in chennai",
    "best places to visit during summer",
    "best places to eat in chennai",
    "best hotels in chennai",
    "places to visit in india",
    "places to visit near me",
    "places to visit during winter",
    "visit temples in tamil nadu",
    "best tourist places in india",
    "best tourist places in chennai",
    "best places to visit in kerala",
    "restaurants near me",
    "restaurants in chennai",
    "restaurants in bangalore",
    "weather in chennai",
    "weather in india",
    "cricket world cup",
    "latest iphone launch",
    "latest ai news"
]

# ==========================================
# LANGUAGE MODEL
# ==========================================

class LanguageModel:

    def __init__(self):

        self.ngram_counts = [defaultdict(int) for _ in range(N)]

        self.history_counts = [defaultdict(int) for _ in range(N)]

        self.unique_followers = defaultdict(set)

        self.unique_predecessors = defaultdict(set)

        self.continuation_count = defaultdict(set)

        self.vocab = set()

    # -------------------------

    def tokenize(self, sentence):

        words = sentence.lower().split()

        return ["<s>"]*(N-1) + words + ["</s>"]

    # -------------------------

    def train(self, corpus):

        for sentence in corpus:

            tokens = self.tokenize(sentence)

            self.vocab.update(tokens)

            length = len(tokens)

            for n in range(1, N+1):

                for i in range(length-n+1):

                    gram = tuple(tokens[i:i+n])

                    self.ngram_counts[n-1][gram] += 1

                    if n > 1:

                        history = gram[:-1]

                        word = gram[-1]

                        self.history_counts[n-1][history] += 1

                        self.unique_followers[history].add(word)

                        self.unique_predecessors[word].add(history)

                        if n == 2:
                            self.continuation_count[word].add(history)

    # -------------------------

        def statistics(self):
        
            return {
        
                "Vocabulary": len(self.vocab),
        
                "1-Grams": len(self.ngram_counts[0]),
        
                "2-Grams": len(self.ngram_counts[1]),
        
                "3-Grams": len(self.ngram_counts[2]),
        
                "4-Grams": len(self.ngram_counts[3]),
        
                "5-Grams": len(self.ngram_counts[4])
        
            }
        
        # ==========================================
        # TRAIN MODEL
        # ==========================================
        
        model = LanguageModel()
        
        model.train(corpus)
        
        stats = model.statistics()
        
        # ==========================================
        # STREAMLIT PAGE
        # ==========================================
        
        st.set_page_config(
        page_title="Search AutoComplete",
        page_icon="🔍",
        layout="wide"
        )
        
        # ==========================================
        # CSS
        # ==========================================
        
        st.markdown("""
        <style>
        
        .main{
        background:#f7f7f7;
        }
        
        .title{
        text-align:center;
        color:#0F62FE;
        font-size:40px;
        font-weight:bold;
        }
        
        .subtitle{
        text-align:center;
        color:gray;
        }
        
        .search{
        padding-top:20px;
        }
        
        </style>
        """, unsafe_allow_html=True)
        
        # ==========================================
        # HEADER
        # ==========================================
        
        st.markdown("<div class='title'>🔍 Search AutoComplete</div>",
        unsafe_allow_html=True)
        
        st.markdown(
        "<div class='subtitle'>5-Gram Language Model using Kneser-Ney Smoothing</div>",
        unsafe_allow_html=True)
        
        st.write("")
        
        # ==========================================
        # SIDEBAR
        # ==========================================
        
        st.sidebar.title("Project")
        
        st.sidebar.success("5-Gram Language Model")
        
        st.sidebar.write("Discount")
        
        discount = st.sidebar.slider(
        "Discount",
        0.1,
        1.0,
        0.75
        )
        
        topk = st.sidebar.slider(
        "Suggestions",
        1,
        10,
        5
        )
        
        # ==========================================
        # SEARCH BOX
        # ==========================================
        
        query = st.text_input(
        "Search",
        placeholder="Type your search..."
        )
        
        # ==========================================
        # MODEL STATISTICS
        # ==========================================
        
        st.subheader("Model Statistics")
        
        col1,col2,col3 = st.columns(3)
        
        col1.metric("Vocabulary",stats["Vocabulary"])
        col2.metric("1-Grams",stats["1-Grams"])
        col3.metric("2-Grams",stats["2-Grams"])
        
        col4,col5,col6 = st.columns(3)
        
        col4.metric("3-Grams",stats["3-Grams"])
        col5.metric("4-Grams",stats["4-Grams"])
        col6.metric("5-Grams",stats["5-Grams"])
        
        st.divider()
        
        st.subheader("Suggestions")
        
        if query:
            suggestions = model.predict(query, topk)
        
            if len(suggestions) == 0:
            
                st.warning("No Suggestions Found")
            
            else:
            
                words = []
                probs = []
            
                for word, score in suggestions:
            
                    words.append(word)
                    probs.append(score)
            
                df = pd.DataFrame({
            
                    "Suggestion": words,
                    "Probability": probs
            
                })
            
                st.dataframe(df, use_container_width=True)
            
                st.bar_chart(
                    df.set_index("Suggestion")
                )
            
                st.success("Top Predictions")
            
                for i, (word, score) in enumerate(suggestions, start=1):
            
                    st.write(
                        f"{i}. **{word}**   (Probability : {score:.6f})"
                    )
        
        else:
            st.info("Start typing to get autocomplete suggestions.")
            st.divider()
            st.subheader("Test Sentence Probability")
        
            sentence = st.text_input(
            "Enter a complete sentence",
            placeholder="best places to visit in india"
            )
        
            if sentence:
            
                prob = model.sentence_probability(sentence)
                
                st.metric(
                    "Sentence Probability",
                    f"{prob:.10f}"
                )
            
            st.divider()
            
            st.caption("Developed using a Sparse 5-Gram Language Model with Recursive Kneser-Ney Smoothing.")
        
        # --------------------------------------------------
        # Continuation Probability (Unigram)
        # --------------------------------------------------
        
        def continuation_probability(self, word):
        
            total_bigrams = len(self.ngram_counts[1])
        
            if total_bigrams == 0:
                return 1 / max(len(self.vocab), 1)
        
            return len(self.continuation_count[word]) / total_bigrams
        
        
        # --------------------------------------------------
        # Recursive Kneser-Ney Probability
        # --------------------------------------------------
        
        def kneser_ney_probability(self, history, word):
        
            order = len(history) + 1
        
            # Base case (Unigram)
            if order == 1:
                return self.continuation_probability(word)
        
            history = tuple(history)
        
            gram = history + (word,)
        
            gram_count = self.ngram_counts[order - 1].get(gram, 0)
        
            history_count = self.history_counts[order - 1].get(history, 0)
        
            # Backoff if history never occurred
            if history_count == 0:
        
                if len(history) > 0:
                    return self.kneser_ney_probability(history[1:], word)
        
                return self.continuation_probability(word)
        
            discounted = max(gram_count - DISCOUNT, 0) / history_count
        
            lambda_weight = (
                DISCOUNT *
                len(self.unique_followers[history])
                / history_count
            )
        
            if len(history) > 0:
                backoff = self.kneser_ney_probability(history[1:], word)
            else:
                backoff = self.continuation_probability(word)
        
            return discounted + lambda_weight * backoff
        
        
        # --------------------------------------------------
        # Sentence Probability
        # --------------------------------------------------
        
        def sentence_probability(self, sentence):
        
            tokens = self.tokenize(sentence)
        
            probability = 1
        
            for i in range(N - 1, len(tokens)):
        
                history = tokens[i - (N - 1):i]
        
                word = tokens[i]
        
                probability *= self.kneser_ney_probability(history, word)
        
            return probability
        
        
        # --------------------------------------------------
        # Top K Predictions
        # --------------------------------------------------
        
        def predict(self, query, top_k=5):
        
            tokens = self.tokenize(query)
        
            history = tokens[-(N - 1):]
        
            predictions = []
        
            for word in self.vocab:
        
                if word in ["<s>", "</s>"]:
                    continue
        
                score = self.kneser_ney_probability(history, word)
        
                predictions.append((word, score))
        
            predictions.sort(key=lambda x: x[1], reverse=True)
        
            return predictions[:top_k]

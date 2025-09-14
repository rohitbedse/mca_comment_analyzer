import pandas as pd
import torch
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
import random
from datetime import datetime, timedelta
from langdetect import detect
from deep_translator import GoogleTranslator

nltk.download('stopwords', quiet=True)

class MCACommentAnalyzer:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        print("Using device:", "GPU" if device==0 else "CPU")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=device
        )
        self.stop_words = set(stopwords.words('english'))

    def translate_to_english(self, text):
        try:
            lang = detect(text)
            if lang != "en":
                return GoogleTranslator(source='auto', target='en').translate(text)
            return text
        except:
            return text

    def map_sentiment(self, pred, text):
        text_lower = text.lower()
        violation_keywords = ["violation", "violates", "illegal", "non-compliant", "breach", "unlawful", "risk", "penalty"]
        suggestion_keywords = ["should", "recommend", "suggest", "advise", "better if", "could", "need to"]
        positive_keywords = ["clear", "helpful", "good", "appreciate", "support"]
        negative_keywords = ["confusing", "unclear", "bad", "problem", "needs clarification"]

        if any(w in text_lower for w in violation_keywords):
            return "Violation"
        if any(w in text_lower for w in suggestion_keywords):
            return "Suggestion"
        if any(w in text_lower for w in positive_keywords):
            return "Positive"
        if any(w in text_lower for w in negative_keywords):
            return "Negative"

        label = pred['label'].upper()
        if label == "POSITIVE":
            return "Positive"
        elif label == "NEGATIVE":
            return "Negative"
        else:
            return "Neutral"

    def process_comment(self, comment):
        translated_comment = self.translate_to_english(comment)
        pred = self.sentiment_model(translated_comment)[0]
        sentiment = self.map_sentiment(pred, translated_comment)

        # Summary
        if len(translated_comment.split()) < 10:
            summary_text = " ".join(translated_comment.split()[:10])
        else:
            try:
                summary_text = self.summarizer(
                    translated_comment, 
                    max_length=30, 
                    min_length=5, 
                    do_sample=False
                )[0]['summary_text']
            except:
                summary_text = translated_comment

        # Keywords
        words = [w for w in translated_comment.lower().split() if w.isalpha() and w not in self.stop_words]
        keywords = list(Counter(words).keys())
        top_keywords = ", ".join(keywords[:3])

        return sentiment, summary_text, keywords, top_keywords

    def process_comments(self, comments_list):
        sentiments, summaries, all_keywords, top_keywords_list, timestamps = [], [], [], [], []
        start_date = datetime.now() - timedelta(days=30)

        for comment in comments_list:
            sentiment, summary, keywords, top_kw = self.process_comment(comment)
            sentiments.append(sentiment)
            summaries.append(summary)
            all_keywords.extend(keywords)
            top_keywords_list.append(top_kw)
            timestamps.append(start_date + timedelta(days=random.randint(0, 30)))

        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Comment": comments_list,
            "Summary": summaries,
            "Sentiment": sentiments,
            "Top Keywords": top_keywords_list
        })

        df.sort_values(by='Timestamp', inplace=True, ascending=True)

        keyword_freq = pd.DataFrame(
            Counter(all_keywords).items(),
            columns=['Keyword', 'Frequency']
        ).sort_values(by='Frequency', ascending=False)

        return df, keyword_freq

    def generate_wordcloud(self, keyword_freq, filename=None):
        wc_dict = dict(zip(keyword_freq['Keyword'], keyword_freq['Frequency']))
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(wc_dict)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        return plt

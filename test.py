# List of mixed-language comments
comments = [
    "The draft is very clear and helpful for companies.",          # English
    "Section 5 is confusing and needs clarification.",             # English
    "It would be better if SMEs get some relief.",                 # English
    "I recommend including more examples for clarity.",            # English
    "Section 12 violates the Companies Act rules.",                # English
    "यह टिप्पणी हिंदी में है।",                                     # Hindi
    "இது தமிழ் மொழியில் ஒரு கருத்து ஆகும்.",                        # Tamil
    "এটি বাংলা ভাষায় একটি মন্তব্য।",                               # Bengali
    "ഈ ഒരു മലയാളം അഭിപ്രായമാണ്."                                   # Malayalam
]

# Initialize analyzer
analyzer = MCACommentAnalyzer()

# Process comments
df, keyword_freq = analyzer.process_comments(comments)

# Show results
print(df[['Comment', 'Sentiment', 'Summary', 'Top Keywords']])

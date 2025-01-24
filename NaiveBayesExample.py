import math
from collections import defaultdict

# step 1 create a small dataset
dataset = [
    {"text": "free money now", "label": "Spam"},
	{"text": "urgent meeting today", "label": "Not Spam"},
	{"text": "win lottery money", "label": "Spam"},
	{"text": "project deadline tomorrow", "label": "Not Spam"},
	{"text": "free lottery ticket", "label": "Spam"},
]

# step 2: Preprocess and tokenize the data
def tokenize(text):
    return text.lower().split()

# Step 3: Train the Naive Bayes Classifier
def train_naive_bayes(dataset):
    word_counts = {"Spam": defaultdict(int), "Not Spam": defaultdict(int)}
    class_counts = {"Spam": 0, "Not Spam": 0}
    total_words = {"Spam": 0, "Not Spam": 0}

    for data in dataset:
        label = data["label"]
        class_counts[label] += 1
        for word in tokenize(data["text"]):
            word_counts[label][word] += 1
            total_words[label] += 1

    return word_counts, class_counts, total_words

# step 4: Calculate class probabilities
def calculate_class_probabilites(word_counts, class_counts, total_words, new_text):
    probs = {}
    total_emails = sum(class_counts.values())
    for label in class_counts:
        # Prior probability
        probs[label] = math.log(class_counts[label] / total_emails)
        for word in tokenize(new_text):
            word_freq = word_counts[label][word]
            # add laplace smoothing
            word_prob = (word_freq + 1) / (total_words[label] + len(word_counts[label]))
            probs[label] += math.log(word_prob)
    return probs

# step 5: Make a prediction
def predict(word_counts, class_counts, total_words, text):
    probs = calculate_class_probabilites(word_counts, class_counts, total_words, text)
    return max(probs, key=probs.get)

# Train the Classifier
word_counts, class_counts, total_words = train_naive_bayes(dataset)

# Step 6: Test the Classifier
new_email = "free lottery"
prediction = predict(word_counts, class_counts, total_words, new_email)
print(f"The email '{new_email}' is clasified as {prediction}")

# Additional test cases
test_emails = ["urgent meeting", "win free money", "project deadline", "free lottery ticket"]
for email in test_emails:
    prediction = predict(word_counts, class_counts, total_words, email)
    print(f"The email '{email}' is classified as {prediction}")
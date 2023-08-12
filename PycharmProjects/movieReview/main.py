import tensorflow as tf
from transformers import DistilBertTokenizer, TFAutoModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from datasets import load_dataset

# Load IMDb movie reviews dataset from Hugging Face Datasets library
dataset = load_dataset("imdb")

# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize and pad the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize and pad the entire dataset
train_dataset = dataset["train"].map(tokenize_function, batched=True)
test_dataset = dataset["test"].map(tokenize_function, batched=True)

# Build the model
input_layer = Input(shape=(None,), dtype=tf.int32, name="input_ids")
distilbert = TFAutoModel.from_pretrained("distilbert-base-uncased")(input_layer).last_hidden_state
output = Dense(1, activation="sigmoid")(distilbert[:, 0, :])

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
train_input_ids = train_dataset["input_ids"]
train_labels = train_dataset["label"]
model.fit(train_input_ids, train_labels, epochs=3, batch_size=16)

# Interactive loop for user queries
while True:
    query = input("Enter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Preprocess the query using the tokenizer
    input_ids = tokenizer.encode(query, add_special_tokens=True, return_tensors="tf")

    # Get model predictions
    prediction = model.predict(input_ids)

    # Print the sentiment prediction
    if prediction > 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    print(f"Predicted sentiment: {sentiment}")
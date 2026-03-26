import streamlit as st
import torch
import torch.nn as nn
import pickle

# ------------------ Model Definition ------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------ Load Model ------------------
input_size = 5000
model = RNN(input_size)
model.load_state_dict(torch.load("prepare.pth"))
model.eval()

# ------------------ Load Vectorizer ------------------
with open("TfidfVectorizer.pkl", "rb") as f:
    tf = pickle.load(f)

# ------------------ Prediction Function ------------------
def predict_sentiment(text):
    text = text.lower()
    vectorized = tf.transform([text]).toarray()
    X = torch.tensor(vectorized, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        output = model(X)
        prob = torch.sigmoid(output.squeeze()).item()

    sentiment = "Positive 😊" if prob > 0.6 else "Negative 😞"
    return sentiment, prob

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.write("Enter a sentence and check its sentiment")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        result, prob = predict_sentiment(user_input)

        st.subheader("Result:")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {prob:.4f}")
    else:
        st.warning("Please enter some text!")
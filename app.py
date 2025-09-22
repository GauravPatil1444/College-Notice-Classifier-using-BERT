import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

st.set_page_config(
    page_title="Notice Classifier",
    page_icon="🗒️",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "./DistilBERT-finetuned/"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def predict_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    probabilities = torch.softmax(logits, dim=-1)
    confidence = probabilities[0][predicted_class_id].item()
    
    return predicted_class_id, confidence

st.title("🗒️College Notice Classifier using BERT")
st.write("Enter text to classify into categories: Exam, Admin, Academic, or Event")

tokenizer, model = load_model()

text_input = st.text_area(
    "Enter your text:",
    placeholder="Type or paste your text here...",
    height=100
)

if st.button("Classify", type="primary"):
    if text_input.strip():

        predicted_class_id, confidence = predict_text(text_input, tokenizer, model)

        id2label = {0: 'Exam', 1: 'Admin', 2: 'Academic', 3: 'Event'}
        predicted_label = id2label[predicted_class_id]

        st.success(f"**Category:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.1%}")
    else:
        st.warning("Please enter some text to classify.")
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.config import load_config

config = load_config('configs/config.yml')
file_paths = config['file_paths']

model_en_vi_checkpoint = file_paths['model_en_vi']
model_vi_en_checkpoint = file_paths['model_vi_en']
model_en_vi_back_checkpoint = file_paths['model_en_vi_back']

# Load the model and tokenizer with caching
@st.cache_data
def load_model(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model, tokenizer

# Function to perform inference
def inference(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=75,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    model.to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=75,
        early_stopping=True,
        num_beams=5,
        length_penalty=2.0
    )

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str

# Main Streamlit app
def main():
    st.title("Text Translation")
    print('load')
    # Load the models
    model_vi_en, tokenizer_vi_en = load_model(model_vi_en_checkpoint)
    model_en_vi, tokenizer_en_vi = load_model(model_en_vi_checkpoint)
    print('end')

    # Model selection dropdown
    model_option = st.selectbox("Select Model", ["English to Vietnamese", "Vietnamese to English"])

    # Input text
    input_text = st.text_input("Enter text:", "")

    # Translate button
    if st.button("Translate"):
        if model_option == "English to Vietnamese":
            # Translate English to Vietnamese
            translated_text = inference(input_text, tokenizer_en_vi, model_en_vi, "cuda")[0]
            st.write("Translated Text (EN to VI):", translated_text)
        elif model_option == "Vietnamese to English":
            # Translate Vietnamese to English
            translated_text = inference(input_text, tokenizer_vi_en, model_vi_en, "cuda")[0]
            st.write("Translated Text (VI to EN):", translated_text)

if __name__ == "__main__":
    main()

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast

def load_model(model_checkpoint):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model, tokenizer

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
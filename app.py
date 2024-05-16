from flask import Flask, render_template, request
from concurrent.futures import ThreadPoolExecutor
from utils.config import load_config
from utils.process import load_model, inference

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=2)  # Số lượng worker trong ThreadPool

# Global variables to store loaded models
model_vi_en = None
tokenizer_vi_en = None
model_en_vi = None
tokenizer_en_vi = None
models_loaded = False

# Load config file
config = load_config('configs/config.yml')
file_paths = config['file_paths']

model_en_vi_checkpoint = file_paths['model_en_vi']
model_vi_en_checkpoint = file_paths['model_vi_en']
model_en_vi_back_checkpoint = file_paths['model_en_vi_back']

# Define a function to load models
def load_models(model_checkpoint, tokenizer_checkpoint):
    model, tokenizer = load_model(model_checkpoint)
    return model, tokenizer

# Load models asynchronously on application start
def load_all_models():
    global model_vi_en, tokenizer_vi_en, model_en_vi, tokenizer_en_vi, models_loaded
    future_vi_en = executor.submit(load_models, model_vi_en_checkpoint, None)
    future_en_vi = executor.submit(load_models, model_en_vi_checkpoint, None)
    model_vi_en, tokenizer_vi_en = future_vi_en.result()
    model_en_vi, tokenizer_en_vi = future_en_vi.result()
    models_loaded = True

# Define endpoint for translation
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if not models_loaded:
        return render_template('index.html', translated_text="Model not loaded yet. Please try again later.", model_name="")

    data = request.form['text']
    selected_model = request.form['model']
    
    if selected_model == "en_vi":
        model, tokenizer = model_en_vi, tokenizer_en_vi
        model_name = "English to Vietnamese"
    elif selected_model == "vi_en":
        model, tokenizer = model_vi_en, tokenizer_vi_en
        model_name = "Vietnamese to English"
    else:
        return render_template('index.html', translated_text="Invalid model selected", model_name="")
    
    translated_text = inference(data, tokenizer, model, "cpu")
    return render_template('index.html', translated_text=translated_text, model_name=model_name)

# Load models asynchronously when the first request is made
@app.before_request
def load_models_before_request():
    global models_loaded
    if not models_loaded:
        print('load')
        load_all_models()

if __name__ == '__main__':
    app.run(debug=True)

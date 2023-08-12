import os
import torch
import fitz
import spacy
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, AutoModelForQuestionAnswering
from werkzeug.utils import secure_filename

MAX_SEQ_LENGTH = 512  # Set this to the maximum sequence length supported by the BERT model

app = Flask(__name__)
templates_folder = os.path.abspath('C:\\Users\\nisha\\PycharmProjects\\PDFReader\\Templates')
app.config['template_folder'] = templates_folder
app.config['UPLOAD_FOLDER'] = 'C:\\Users\\nisha\\PycharmProjects\\PDFReader'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

nlp = spacy.load("en_core_web_sm")

def read_pdf(file_path):
    with fitz.open(file_path) as pdf_doc:
        text = ""
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            text += page.get_text()
    return text


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return 'No file part'

    file = request.files['pdf_file']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename('uploaded_pdf.pdf')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify(success=True, message='File uploaded successfully')
    else:
        return jsonify(success=False, message='Invalid file format. Only PDF files are allowed.')


@app.route('/chat', methods=['POST'])
def chat():
    if 'question' in request.form:
        question = request.form['question']

        pdf_text = read_pdf(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_pdf.pdf'))

        # Concatenate the question and pdf_text
        concatenated_text = f"{question} {pdf_text}"

        # Truncate or split the concatenated_text if it exceeds MAX_SEQ_LENGTH
        inputs = tokenizer(
            concatenated_text,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        # Get the model output
        with torch.no_grad():
            outputs = model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the index of the [CLS] token
        cls_index = torch.argmax(start_scores)

        # Get the tokens that represent the answer (those after the [CLS] token)
        answer_tokens = inputs["input_ids"][0][cls_index+1 : cls_index+1+MAX_SEQ_LENGTH]

        # Convert the tokens back to text
        answer = tokenizer.decode(answer_tokens)

        if not answer.strip():
            # If the BERT-based answer is not specific, use Named Entity Recognition (NER) to identify entities
            entities = []
            doc = nlp(pdf_text)
            for ent in doc.ents:
                entities.append(ent.text)

            # If NER identified entities, use the first entity as the answer
            if entities:
                answer = entities[0]
            else:
                answer = "Sorry, I couldn't find an answer to your question in the uploaded PDF."

        return jsonify(answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
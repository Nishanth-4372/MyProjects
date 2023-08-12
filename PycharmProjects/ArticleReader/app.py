from flask import Flask, render_template, request
from newspaper import Article
from transformers import pipeline

app = Flask(__name__)

nlp_bert = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')


def extract_article_content(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def answer_question(article_content, question):
    result = nlp_bert(question=question, context=article_content)
    answer = result['answer']

    if answer:
        return answer
    else:
        return "The information could not be located in the article."


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article_url = request.form['article_url']
        question = request.form['question']

        try:
            article_content = extract_article_content(article_url)
        except Exception as e:
            print("Error extracting article content:", e)
            return render_template('index.html', error="Failed to extract article content")

        answer = answer_question(article_content, question)
        return render_template('index.html', answer=answer)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

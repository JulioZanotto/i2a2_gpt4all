from flask import Flask, render_template, request
from gpt4all import GPT4All

app = Flask(__name__)

model = GPT4All("ggml-stable-vicuna-13B.q4_2.bin")

system_template = 'You are an artificial intelligence psychiatrist assistant called ElizaGPT.'
prompt_template = 'USER: {0}\ELizaGPT: '

# prompts = ['Hello my terapist']
# first_input = system_template + prompt_template.format(prompts[0])
# response = model.generate(first_input, temp=0)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_response():
    user_input = request.form['user_input']

    # Chame a função do GPT4All para gerar a resposta com base no input do usuário
    with model.chat_session(system_template):
        response = model.generate(prompt=user_input, temp=0.8, max_tokens=150)

    return render_template('index.html', response=response)


if __name__ == '__main__':
    app.run(debug=True)

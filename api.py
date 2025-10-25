from flask import Flask, request, jsonify
from llm import assistant

app = Flask(__name__)

@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')
    return f"<h1>Hello, {name}!</h1>"

@app.route('/input', methods=['POST'])
def handle_input():
    data = request.get_json()
    prompt = data.get('prompt', '')

    # if not prompt:
    #     return jsonify({'error': 'Prompt is required'}), 400
    # response ="asdas"
    response = assistant(prompt)
    return jsonify({
        'prompt': prompt,
        'response': response,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True)

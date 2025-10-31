from flask import Flask, request, jsonify
from llm import assistant
import logging

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>LLM API is running. Use /greet or /input endpoints.</h1>"


@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')
    return f"<h1>Hello, {name}!</h1>"

@app.route('/input', methods=['POST'])
def handle_input():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    logging.info(f"Received prompt: {prompt[:50]}...")  # Log first 50 chars

    try:
        response = assistant(prompt)
    except Exception as e:
        logging.error(f"Error during assistant response generation: {e}")
        return jsonify({'error': 'Internal server error during generation'}), 500

    return jsonify({
        'prompt': prompt,
        'response': response,
        'status': 'success'
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

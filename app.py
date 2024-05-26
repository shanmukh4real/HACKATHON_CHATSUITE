from flask import Flask, request, render_template, jsonify
from bot.response_generator import get_response
from bot.intent_recognizer import recognize_intent
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.form['message']
        if not user_input:
            raise ValueError("No input provided")

        intent = recognize_intent(user_input)
        response = get_response(user_input)
        logging.info(f"User input: {user_input}, Intent: {intent}, Response: {response}")
        return jsonify({'response': response, 'intent': intent})
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

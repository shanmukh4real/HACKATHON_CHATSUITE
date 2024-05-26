import requests

def get_response(user_input):
    api_url = 'YOUR_CHATBOT_API_URL'
    response = requests.post(api_url, json={'message': user_input})
    return response.json().get('reply')

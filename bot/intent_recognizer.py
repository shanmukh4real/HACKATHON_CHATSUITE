def recognize_intent(user_input):
    # Basic example of intent recognition
    if 'weather' in user_input.lower():
        return 'weather'
    return 'general'

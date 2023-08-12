import openai

# Replace 'YOUR_OPENAI_API_KEY' with your actual API key
openai.api_key = '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'

def generate_response(prompt, max_tokens=100):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response['choices'][0]['text'].strip()

def main():
    print("Chatbot: Hi there! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input, max_tokens=100)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()

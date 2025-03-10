from openai import OpenAI



client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='ajindal/llama3.1-storm:8b-Q8_0',
)

print(chat_completion.choices[0].message.content)
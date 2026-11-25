import os
from dotenv import load_dotenv
from groq import Groq, GroqError

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY bulunamadı! .env dosyasını kontrol et.")

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-20b"

messages = [
    {
        "role": "system",
        "content": "Sen yardımcı, açıklayıcı ve Türkçe konuşan bir AI mentorsun."
    }
]


def chat(user_message: str) -> None:
    messages.append({
        "role": "user",
        "content": user_message
    })

    print("\nAI: ", end="", flush=True)

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
            stream=True
        )

        assistant_response = ""

        for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                print(content, end="", flush=True)
                assistant_response += content

        print()

        messages.append({
            "role": "assistant",
            "content": assistant_response
        })

    except GroqError as error:
        print(f"\nGroq API hatası: {error}")
        messages.pop()

    if len(messages) > 11:
        messages[:] = [messages[0]] + messages[-10:]


def main():
    print("=" * 45)
    print("       AI MENTOR - GROQ STREAMING CHAT")
    print("=" * 45)
    print("Çıkmak için 'exit' yaz.\n")

    while True:
        user_message = input("Sen: ").strip()

        if user_message.lower() in ["exit", "quit", "çık"]:
            print("\nChatbot kapatılıyor. Görüşürüz!")
            break

        if not user_message:
            continue

        chat(user_message)


if __name__ == "__main__":
    main()

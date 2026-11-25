
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# Bilgisayarda çalışan yerel Ollama modeline bağlanıyoruz.
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

messages = [
    SystemMessage(
        content="Sen yardımcı, açıklayıcı ve Türkçe konuşan bir AI asistansın."
    )
]


def chat(question: str):
    messages.append(HumanMessage(content=question))

    try:
        response = model.invoke(messages)

        print("\nYEREL AI:")
        print(response.content)

        messages.append(response)

        # Son konuşmaları hafızada tut.
        if len(messages) > 11:
            messages[:] = [messages[0]] + messages[-10:]

    except Exception as error:
        print(f"\nOllama bağlantı hatası: {error}")
        messages.pop()


def main():
    print("=" * 45)
    print("       OLLAMA - YEREL AI CHAT")
    print("=" * 45)
    print("Model: qwen2.5:3b")
    print("Çıkmak için 'exit' yaz.\n")

    while True:
        question = input("Sen: ").strip()

        if question.lower() in ["exit", "quit", "çık"]:
            print("\nYerel AI kapatılıyor.")
            break

        if question:
            chat(question)


if __name__ == "__main__":
    main()


import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY bulunamadı! .env dosyasını kontrol et.")


@tool
def add_numbers(a: int, b: int) -> int:
    """İki tam sayıyı toplar ve sonucu döndürür."""
    return a + b


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=API_KEY
)

model_with_tools = model.bind_tools([add_numbers])


def ask_question(question: str):
    messages = [HumanMessage(content=question)]

    response = model_with_tools.invoke(messages)

    if not response.tool_calls:
        print("\nAI:", response.content)
        return

    messages.append(response)

    for tool_call in response.tool_calls:
        if tool_call["name"] == "add_numbers":
            result = add_numbers.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

    final_response = model_with_tools.invoke(messages)

    print("\nAI:", final_response.content)


def main():
    print("=" * 45)
    print("       LANGCHAIN TOOL CALLING")
    print("=" * 45)
    print("Çıkmak için 'exit' yaz.\n")

    while True:
        question = input("Sen: ").strip()

        if question.lower() in ["exit", "quit", "çık"]:
            print("Program kapatıldı.")
            break

        if question:
            try:
                ask_question(question)
            except Exception as error:
                print(f"\nHata: {error}")


if __name__ == "__main__":
    main()

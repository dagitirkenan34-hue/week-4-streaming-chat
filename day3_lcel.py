
import os
import json

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY bulunamadı! .env dosyasını kontrol et.")

# Groq modelini LangChain'e bağlıyoruz.
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=API_KEY
)

# Model yanıtını JSON formatında ayrıştıracak parser.
parser = JsonOutputParser()

# Prompt şablonu.
prompt = PromptTemplate(
    template="""
Sen yardımcı bir Python eğitmenisin.

Kullanıcının sorusunu cevapla.
Yanıtını aşağıdaki JSON formatında oluştur:

{format_instructions}

Kullanıcı sorusu: {question}
""",
    input_variables=["question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# LCEL zinciri: Prompt -> Model -> JSON Parser
chain = prompt | model | parser


def ask_question(question: str):
    result = chain.invoke({
        "question": question
    })

    print("\nAI YANITI:")
    print(json.dumps(result, ensure_ascii=False, indent=4))


def main():
    print("=" * 45)
    print("       LANGCHAIN LCEL - JSON CHAT")
    print("=" * 45)
    print("Çıkmak için 'exit' yaz.\n")

    while True:
        question = input("Sorun: ").strip()

        if question.lower() in ["exit", "quit", "çık"]:
            print("Program kapatıldı.")
            break

        if not question:
            continue

        try:
            ask_question(question)
        except Exception as error:
            print(f"\nHata oluştu: {error}")


if __name__ == "__main__":
    main()

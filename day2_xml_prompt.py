from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY bulunamadı!")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


def ask_sql_expert(question: str):
    messages = [
        {
            "role": "system",
            "content": """
<instructions>
    <role>Sen deneyimli bir SQL uzmanısın.</role>
    <task>
        Kullanıcının veritabanı sorusunu analiz et.
        Uygun SQL sorgusunu üret ve açıkla.
    </task>
    <rules>
        <rule>Yanıtlarını Türkçe ver.</rule>
        <rule>SQL kodlarını markdown kod bloğunda göster.</rule>
        <rule>Tablo ve kolon isimlerini açıkça belirt.</rule>
        <rule>Bilgi eksikse varsayım yaptığını belirt.</rule>
        <rule>Veri değiştiren sorgularda dikkatli ol ve önce açıklama yap.</rule>
    </rules>
    <output_format>
        <section>Açıklama</section>
        <section>SQL Sorgusu</section>
        <section>Sorgunun Çalışma Mantığı</section>
    </output_format>
</instructions>
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024
    )

    print("\nSQL UZMANI:")
    print(response.choices[0].message.content)


def main():
    print("=" * 45)
    print("       XML PROMPT - SQL UZMANI")
    print("=" * 45)
    print("Çıkmak için 'exit' yaz.\n")

    while True:
        question = input("Sorun: ").strip()

        if question.lower() in ["exit", "quit", "çık"]:
            print("Program kapatıldı.")
            break

        if question:
            try:
                ask_sql_expert(question)
            except Exception as error:
                print(f"API hatası: {error}")


if __name__ == "__main__":
    main()
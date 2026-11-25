
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY bulunamadı!")

app = FastAPI(title="AI Streaming Chat API")

client = AsyncGroq(api_key=API_KEY)

MODEL = "openai/gpt-oss-20b"

# Konuşma geçmişini kullanıcı bazında tutar.
chat_history = {}


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    provider: str = "cloud"


@app.get("/")
def home():
    return {
        "message": "AI Streaming Chat API çalışıyor!"
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    if request.provider != "cloud":
        return {
            "error": "Yerel Ollama şu anda devre dışı."
        }

    history = chat_history.setdefault(
        request.user_id,
        [
            {
                "role": "system",
                "content": "Sen yardımcı ve Türkçe konuşan bir AI asistansın."
            }
        ]
    )

    history.append({
        "role": "user",
        "content": request.message
    })

    # Sistem mesajı hariç son 5 mesajı tut.
    history[:] = [history[0]] + history[-5:]

    async def generate():
        assistant_response = ""

        try:
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=history,
                temperature=0.7,
                max_completion_tokens=1024,
                stream=True
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content

                if content:
                    assistant_response += content
                    yield content

            history.append({
                "role": "assistant",
                "content": assistant_response
            })

            history[:] = [history[0]] + history[-5:]

        except Exception as error:
            yield f"\nAPI hatası: {error}"

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

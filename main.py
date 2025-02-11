import os
from fastapi import FastAPI, UploadFile
from bhashini_translator import Bhashini
import base64

app = FastAPI()

@app.post("/translate")
async def translate(text: str, source: str, target: str):
    bhashini = Bhashini(source, target)
    translated_text = bhashini.translate(text)
    return {"translated": translated_text}

@app.post("/asr-translate")
async def asr_translate(file: UploadFile):
    bhashini = Bhashini("hin_Deva", "eng_Latn")
    audio_data = await file.read()
    base64_audio = base64.b64encode(audio_data).decode("utf-8")
    translated_text = bhashini.asr_nmt(base64_audio)
    return {"translated": translated_text}

@app.post("/tts")
async def tts(text: str, source: str):
    bhashini = Bhashini(source)
    audio_base64 = bhashini.tts(text)
    return {"audio": audio_base64}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

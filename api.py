from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import subprocess
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="front/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = Path("front/index.html")
    return index.read_text()

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):

    uid = str(uuid.uuid4())

    input_path = UPLOAD_DIR / f"{uid}_{file.filename}"
    output_csv = RESULT_DIR / f"{uid}.csv"

    # сохраняем видео
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # запускаем video_detect.py
    command = [
        "python",
        "video_detect.py",
        str(input_path),
        "-o",
        str(output_csv)
    ]

    process = subprocess.run(command)

    # проверка ошибок
    if process.returncode != 0:
        return {
            "status": "error",
            "message": "Ошибка обработки видео"
        }

    # отдаем CSV
    return FileResponse(
        path=output_csv,
        filename="result.csv",
        media_type="text/csv"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
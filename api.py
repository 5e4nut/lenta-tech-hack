from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import subprocess
import uuid

app = FastAPI()

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


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

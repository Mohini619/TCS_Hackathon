from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import subprocess
import os
import uuid

app = FastAPI(title="Pfizer Image Enhancement API")

UPLOAD_DIR="uploads"
RESULT_DIR="results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
        return """
        <!DOCTYPE html>
        <html>
        <head>
                <title>Pfizer Image Enhancement</title>
                <style>
                        body {
                                font-family: Arial, sans-serif;
                                margin: 50px;
                                text-align: center;
                        }
                        .container {
                                width: 500px;
                                margin: auto;
                                padding: 20px;
                                border: 1px solid #ccc;
                                border-radius: 10px;
                                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                        }

                        h2 {
                                color: #0066cc;
                        }
                        input[type=file] {
                                margin: 20px;
                        }
                        input[type=submit] {
                                background-color: #0066cc;
                                color: white;
                                border: none;
                                padding: 10px 20px;
                                cursor: pointer;
                                border-radius: 5px;
                        }
                        input[type=submit]:hover {
                                background-color: #004c99;
                        }
                        </style>
                </head>
        <body>
        <div class="container">
        <h2>Pfizer Image Enhancement</h2>
        <form action="/enhance" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <br>
                <input type="submit" value="Enhance Image">
        </form>

        <br>
        <a href="/health">Health Check</a>
        </div>
        </body>
        </html>
        """

@app.get("/health")
def health():
        return {
                "status": "healthy"
        }

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
        uid = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR,f"{uid}_{file.filename}")
        with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        cmd = [
                "python3",
                "inference_realesrgan.py",
                "-n",
                "RealESRGAN_x4plus",
                "--face_enhance",
                "-i",
                input_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode!= 0:
                return {
                        "status": "error",
                        "message": result.stderr
                }

        filename = os.path.splitext(os.path.basename(input_path))[0]
        name= filename
        png_file = f"results/{name}_out.png"
        jpg_file = f"results/{name}_out.jpg"
        jpeg_file = f"results/{name}_out.jpeg"

        if os.path.exists(png_file):
                output_file = png_file
                media_type = "image/png"
                download_name = "enhanced.png"

        elif os.path.exists(jpg_file):
                output_file = jpg_file
                media_type = "image/jpeg"
                download_name = "enhanced.jpg"

        elif os.path.exists(jpeg_file):
                output_file = jpeg_file
                media_type = "image/jpeg"
                download_name = "enhanced.jpeg"
        else:
                return {
                        "status": "error",
                        "message": f"Output file not found. Checked: {png_file}, {jpg_file}, {jpeg_file}"
                }

        return FileResponse(output_file, media_type=media_type, filename=download_name)
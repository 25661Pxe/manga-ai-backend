
import os, io, re, json, base64
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

app = FastAPI(title="Manga AI v5 FULL Cloud Scanlation", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class ScanRequest(BaseModel):
    image_base64: str
    target_lang: str = "Thai"


def strip_base64(s: str) -> str:
    if not s:
        raise HTTPException(status_code=400, detail="image_base64 is empty")
    if "," in s:
        s = s.split(",", 1)[1]
    return s.strip()


def decode_image(image_base64: str) -> Image.Image:
    try:
        raw = base64.b64decode(strip_base64(image_base64))
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def get_font(size: int):
    for p in [
        "C:/Windows/Fonts/LeelawUI.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    words = text.replace("\n", " ").split()
    lines, cur = [], ""
    for w in words:
        t = w if not cur else cur + " " + w
        box = draw.textbbox((0, 0), t, font=font)
        if box[2] - box[0] <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_text(draw, text, w, h):
    for size in range(34, 10, -1):
        font = get_font(size)
        lines = wrap(draw, text, font, max(20, w - 18))
        lh = int(size * 1.22)
        total_h = lh * len(lines)
        max_line = 0
        for line in lines:
            b = draw.textbbox((0, 0), line, font=font)
            max_line = max(max_line, b[2] - b[0])
        if total_h <= h - 12 and max_line <= w - 12:
            return font, lines, lh
    font = get_font(11)
    return font, wrap(draw, text, font, max(20, w - 12)), 15


def draw_scanlation(img: Image.Image, items: List[Dict[str, Any]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)

    for item in items:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
        text = item["translated"]

        pad = 6
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(out.width, x + w + pad)
        y1 = min(out.height, y + h + pad)

        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=max(10, min(w, h) // 5),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=1,
        )

        font, lines, lh = fit_text(draw, text, x1 - x0, y1 - y0)
        total_h = lh * len(lines)
        ty = y0 + ((y1 - y0) - total_h) // 2

        for line in lines:
            b = draw.textbbox((0, 0), line, font=font)
            tw = b[2] - b[0]
            tx = x0 + ((x1 - x0) - tw) // 2
            for ox, oy in [(-1,0), (1,0), (0,-1), (0,1)]:
                draw.text((tx + ox, ty + oy), line, font=font, fill=(255,255,255))
            draw.text((tx, ty), line, font=font, fill=(0,0,0))
            ty += lh

    return out


def extract_translate(img: Image.Image, target_lang: str) -> List[Dict[str, Any]]:
    if client is None:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY in Render Environment Variables")

    prompt = f"""
You are a manga OCR + scanlation translator.

Detect all manga dialogue text visible in the image and translate to {target_lang}.
Return ONLY valid JSON. No markdown.

Rules:
- Ignore browser UI, website UI, ads, page numbers, buttons, navigation.
- Use image pixel coordinates.
- Merge words in the same speech bubble.
- Keep translation short enough to fit.
- Natural manga-style translation.

JSON format:
{{
  "items": [
    {{
      "text": "original",
      "translated": "translation",
      "x": 100,
      "y": 100,
      "w": 180,
      "h": 90
    }}
  ]
}}
"""
    data_url = image_to_data_url(img)

    res = client.responses.create(
        model=OPENAI_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url}
            ]
        }]
    )

    raw = res.output_text.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I | re.S).strip()

    try:
        data = json.loads(raw)
        items = data.get("items", [])
    except Exception:
        items = [{
            "text": "",
            "translated": raw[:400],
            "x": img.width // 4,
            "y": img.height // 4,
            "w": img.width // 2,
            "h": 120
        }]

    clean = []
    for it in items:
        try:
            translated = str(it.get("translated", "")).strip()
            if not translated:
                continue
            x = max(0, min(int(float(it.get("x", 0))), img.width - 1))
            y = max(0, min(int(float(it.get("y", 0))), img.height - 1))
            w = max(25, min(int(float(it.get("w", 100))), img.width - x))
            h = max(25, min(int(float(it.get("h", 60))), img.height - y))
            clean.append({
                "text": str(it.get("text", "")),
                "translated": translated,
                "x": x, "y": y, "w": w, "h": h
            })
        except Exception:
            continue

    return clean


@app.get("/")
def root():
    return {"ok": True, "name": "Manga AI v5 FULL Cloud Scanlation"}


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}


@app.post("/translate-overlay")
def translate_overlay(req: ScanRequest):
    img = decode_image(req.image_base64)
    items = extract_translate(img, req.target_lang)
    return {"ok": True, "items": items}


@app.post("/translate-scanlation")
def translate_scanlation(req: ScanRequest):
    img = decode_image(req.image_base64)
    items = extract_translate(img, req.target_lang)
    rendered = draw_scanlation(img, items)
    return {
        "ok": True,
        "items": items,
        "image_base64": image_to_data_url(rendered)
    }

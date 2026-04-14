"""CLIP Image Embedding Service — runs on Hugging Face Spaces (free)."""
import io
import base64
from PIL import Image
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="CLIP Embedding Service")

print("Loading CLIP model...")
model = SentenceTransformer("clip-ViT-B-32")
print("CLIP ready!")


class EmbedRequest(BaseModel):
    image_url: str = ""
    image_base64: str = ""


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_image(req: EmbedRequest):
    if not req.image_url and not req.image_base64:
        raise HTTPException(400, "Provide image_url or image_base64")

    try:
        if req.image_url:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(req.image_url)
                if resp.status_code != 200:
                    raise HTTPException(400, f"Download failed: {resp.status_code}")
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            image_bytes = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        embedding = model.encode(image).tolist()
        return EmbedResponse(embedding=embedding, dimension=len(embedding))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "clip-ViT-B-32", "dimension": 512}

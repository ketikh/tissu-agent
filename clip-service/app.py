"""CLIP Image Embedding Service — runs on Hugging Face Spaces (free).

Provides /embed endpoint that takes an image URL or base64 and returns
a 512-dim CLIP embedding vector.
"""
import io
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="CLIP Embedding Service")

# Load CLIP model once at startup
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP ready!")


class EmbedRequest(BaseModel):
    image_url: str = ""
    image_base64: str = ""


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_image(req: EmbedRequest):
    """Generate CLIP embedding for an image."""
    if not req.image_url and not req.image_base64:
        raise HTTPException(400, "Provide image_url or image_base64")

    try:
        if req.image_url:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(req.image_url)
                if resp.status_code != 200:
                    raise HTTPException(400, f"Failed to download: {resp.status_code}")
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            image_bytes = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        embedding = features[0].tolist()
        return EmbedResponse(embedding=embedding, dimension=len(embedding))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "clip-vit-base-patch32", "dimension": 512}

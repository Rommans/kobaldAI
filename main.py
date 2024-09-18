from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

model_name = "KoboldAI/GPT-J-6B-Skein"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text_async(prompt: str):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=20, do_sample=True, top_p=0.8, temperature=0.8)
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/generate/")
async def generate_text(request: PromptRequest):
    prompt = request.prompt
    if not prompt.lower().startswith("story:"):
        return {"message": "This model is for story generation only. Start your prompt with 'Story:'."}
    
    story = await asyncio.to_thread(generate_text_async, prompt)
    
    return {"story": story}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

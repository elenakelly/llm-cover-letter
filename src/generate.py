import os, sys, argparse, textwrap, json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from rich import print
from rich.panel import Panel
from typing import ClassVar, Any


# -------- Provider Abstraction --------

class LLMProvider(BaseModel):
    name: str
    model: str
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

class LocalFLANT5(LLMProvider):
    pipeline: ClassVar[Any] = None  # <— tell Pydantic this is a class-level cache, not a field

    def _load(self):
        if LocalFLANT5.pipeline is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model)
            LocalFLANT5.pipeline = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
        return LocalFLANT5.pipeline

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        pipe = self._load()
        res = pipe(
            prompt,
            max_new_tokens=max_tokens,   # allow enough space
            min_new_tokens=150,          # force more than one line
            do_sample=False,             # deterministic
            num_beams=5,                 # stronger planning
            early_stopping=True,
            no_repeat_ngram_size=3,
            clean_up_tokenization_spaces=True,
        )
        return (res[0].get("generated_text", "") if res else "").strip()




class HFInference(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import requests
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise SystemExit("Set HUGGINGFACEHUB_API_TOKEN to use --provider hf")
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j and "generated_text" in j[0]:
            return j[0]["generated_text"]
        return str(j)

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import requests
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY to use --provider openai")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise professional writing assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        r = requests.post(url, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class OllamaProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import requests
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base}/api/generate"
        data = {"model": self.model, "prompt": prompt, "options": {"num_predict": max_tokens}}
        r = requests.post(url, json=data, timeout=120)
        r.raise_for_status()
        text = ""
        for line in r.iter_lines():
            if not line: continue
            try:
                obj = json.loads(line.decode("utf-8"))
                text += obj.get("response", "")
                if obj.get("done"): break
            except: pass
        return text

def make_provider(name: str, model: Optional[str]):
    if name == "local": return LocalFLANT5(name="local", model=model or "google/flan-t5-small")
    if name == "hf": return HFInference(name="hf", model=model or "google/flan-t5-small")
    if name == "openai": return OpenAIProvider(name="openai", model=model or "gpt-4o-mini")
    if name == "ollama": return OllamaProvider(name="ollama", model=model or "llama3")
    raise SystemExit(f"Unknown provider: {name}")

# -------- App Logic --------

PROMPT_TMPL = """You are an expert technical writer.
Write TWO sections ONLY, EXACTLY this Markdown:

## Bullets
- five bullets, max 20 words each, action + impact + numbers when possible
- bullet 2
- bullet 3
- bullet 4
- bullet 5
## Cover Letter
One paragraph, 120–150 words. No greeting or closing.

Use only information consistent with the job description and the candidate skills.

[Job Description]
{jd}

[Candidate Skills / Keywords]
{skills}
"""




def read_text(path): return Path(path).read_text(encoding="utf-8")
def load_skills(path): return ", ".join([s.strip() for s in read_text(path).replace("|",",").replace("\n",",").split(",") if s.strip()][:60])

def save_outputs(bullets, cover):
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/bullets.txt").write_text(bullets, encoding="utf-8")
    Path("outputs/cover_letter.md").write_text(cover, encoding="utf-8")
    print(Panel.fit("[green]Saved outputs/bullets.txt and outputs/cover_letter.md[/]"))

def split_sections(text: str):
    """Try headers first, then bullet-markers, then sentence heuristics."""
    bullets, cover = "", ""

    # 1) header split
    parts = text.split("##")
    for p in parts:
        h = p.strip().lower()
        if h.startswith("bullets"):
            bullets = "\n".join(p.splitlines()[1:]).strip()
        if h.startswith("cover letter"):
            cover = "\n".join(p.splitlines()[1:]).strip()
    if bullets and cover:
        return bullets, cover

    # 2) bullet marker heuristic
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if ln.lstrip().startswith(("-", "*", "•"))]
    if bullet_lines:
        bullets = "\n".join(bullet_lines[:5])
        remaining = [ln for ln in lines if ln not in bullet_lines]
        cover = "\n".join(remaining).strip()
        return bullets, cover

    # 3) sentence-based heuristic: first 5 short-ish sentences as bullets, rest as cover
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if sentences:
        bullets = "\n".join("- " + s for s in sentences[:5])
        cover = " ".join(sentences[5:]).strip()
        return bullets, cover

    return "", ""

def fallback_bullets_and_cover(jd: str, skills_csv: str):
    # make 5 simple bullets from skill hits in the JD
    import re
    skills = [s.strip() for s in skills_csv.split(",") if s.strip()]
    jd_low = jd.lower()
    hits = [s for s in skills if s.lower() in jd_low]
    chosen = (hits[:5] or skills[:5])
    bullets = "\n".join(f"- Applied {s} to deliver measurable outcomes in production" for s in chosen)

    cover = (
        "I am excited to apply my experience in {skills} to your needs. I have built and deployed models, "
        "owned data preprocessing and feature engineering, and collaborated with cross-functional teams to deliver "
        "reliable systems. I focus on clarity, reproducibility, and measurable impact, and I enjoy translating complex "
        "requirements into simple, maintainable solutions. I would welcome the chance to contribute to your team."
    ).format(skills=", ".join(chosen[:4]))
    return bullets, cover


def run(jd_path, cv_path, provider_name, model):
    jd, skills = read_text(jd_path), load_skills(cv_path)
    prompt = PROMPT_TMPL.format(jd=jd, skills=skills)
    provider = make_provider(provider_name, model)
    print(Panel.fit(f"Provider: [bold]{provider.name}[/] | Model: [bold]{provider.model}[/]"))
    out = provider.generate(prompt, max_tokens=400)
    # Always save raw output for debugging/parsing
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/raw.txt").write_text(out, encoding="utf-8")

    bullets, cover = split_sections(out)
    if not bullets or not cover:
        # fallback so we always produce something useful
        fb_bullets, fb_cover = fallback_bullets_and_cover(jd, skills)
        bullets = bullets or fb_bullets
        cover = cover or fb_cover

    save_outputs(bullets or "", cover or "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--provider", default="local", choices=["local","hf","openai","ollama"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    run(args.jd, args.cv, args.provider, args.model)

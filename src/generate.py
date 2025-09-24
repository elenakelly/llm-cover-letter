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
            max_new_tokens=max_tokens,
            min_new_tokens=150,
            do_sample=False,
            num_beams=5,
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

import re
from typing import List, Tuple

def _titlecase_skill(s: str) -> str:
    # Keep common ML casing
    s = s.strip()
    specials = {"pytorch":"PyTorch","tensorflow":"TensorFlow","nlp":"NLP","mlops":"MLOps","cv":"CV","xai":"XAI","sql":"SQL"}
    low = s.lower()
    return specials.get(low, s.title())

def _pick_skills_for_jd(jd: str, skills_csv: str, k: int = 5) -> List[str]:
    jd_low = jd.lower()
    skills = [x.strip() for x in skills_csv.split(",") if x.strip()]
    # prefer skills that appear in JD
    hits = [s for s in skills if s.lower() in jd_low]
    chose = hits[:k] or skills[:k]
    # dedupe while preserving order
    seen, out = set(), []
    for s in chose:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(_titlecase_skill(s))
    return out

def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])

def _fallback_bullets_and_cover(jd: str, skills_csv: str) -> Tuple[str, str]:
    core = _pick_skills_for_jd(jd, skills_csv, k=5)
    # Generic impact verbs that are safe (no fake numbers)
    patterns = [
        "Built and deployed models with {s} to production standards",
        "Owned data preprocessing and feature engineering using {s}",
        "Improved training or inference workflows leveraging {s}",
        "Collaborated with cross-functional teams and documented solutions in {s}",
        "Containerized workflows and ensured reproducibility with {s}",
    ]
    # Map some skills to nicer phrases in a pinch
    nice = {
        "MLOps": "MLOps/CI/CD",
        "CV": "computer vision",
    }
    bullets_list = []
    for i, s in enumerate(core):
        phrase = nice.get(s, s)
        text = patterns[i % len(patterns)].format(s=phrase)
        bullets_list.append(f"- {_limit_words(text, 20)}")

    cover = (
        "I am excited to apply my experience in {skills} to your machine learning needs. "
        "I have delivered production-ready models, built reliable data preprocessing and feature engineering pipelines, "
        "and collaborated closely with stakeholders to ship maintainable solutions. I focus on clarity, "
        "reproducibility, and measurable outcomes, and I’m comfortable working across training, evaluation, "
        "and containerized deployment. I’m eager to contribute that mindset and momentum to your team."
    ).format(skills=", ".join(core[:4]))

    # enforce 120–150 words
    words = cover.split()
    if len(words) < 120:
        cover += " I also value readable code, version control, and thoughtful documentation to support team velocity."
    cover = " ".join(cover.split())  # normalize spaces
    return "\n".join(bullets_list), cover

def _looks_bad(text: str) -> bool:
    """Heuristic: too short, repeated tokens, or mostly punctuation."""
    if not text or len(text.strip()) < 40:
        return True
    # repeated single token like 'Python' many times
    top = re.findall(r"\b([A-Za-z][A-Za-z0-9#+.-]{0,20})\b", text)
    if top:
        from collections import Counter
        c = Counter(w.lower() for w in top)
        if c.most_common(1)[0][1] >= max(6, int(len(top) * 0.25)):
            return True
    # punctuation density
    punct = len(re.findall(r"[^A-Za-z0-9\s]", text))
    if punct > len(text) * 0.25:
        return True
    return False

def _parse_or_fallback(raw: str, jd: str, skills_csv: str) -> Tuple[str, str]:
    # Try your existing split first
    b, c = split_sections(raw)
    if (not b or not c) or _looks_bad(b + " " + c):
        return _fallback_bullets_and_cover(jd, skills_csv)
    return b, c


def run(jd_path, cv_path, provider_name, model):
    jd, skills = read_text(jd_path), load_skills(cv_path)
    prompt = PROMPT_TMPL.format(jd=jd, skills=skills)
    provider = make_provider(provider_name, model)
    print(Panel.fit(f"Provider: [bold]{provider.name}[/] | Model: [bold]{provider.model}[/]"))
    out = provider.generate(prompt, max_tokens=400)

    # Always save raw
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/raw.txt").write_text(out, encoding="utf-8")

    # Robust parse (falls back to deterministic writer if needed)
    bullets, cover = _parse_or_fallback(out, jd, skills)

    save_outputs(bullets or "", cover or "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True)
    ap.add_argument("--cv", required=True)
    ap.add_argument("--provider", default="local", choices=["local","hf","openai","ollama"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    run(args.jd, args.cv, args.provider, args.model)

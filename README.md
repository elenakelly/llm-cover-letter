# LLM Cover Letter 

Tiny CLI tool that turns a **job description + your skills** into:
- ✅ 5 tailored resume bullet points  
- ✅ a short, punchy cover letter  

This is a portfolio-friendly mini project that shows:
- clean **prompt engineering** grounded in user data
- a **provider-agnostic LLM wrapper** (local transformers, Hugging Face, OpenAI, Ollama)
- clear outputs in `outputs/` you can commit as demo artifacts

---

## Quick Start

```bash
# (optional) create venv
# python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# Run with local FLAN-T5 (downloads ~100MB on first run)
python src/generate.py --jd examples/job_description.txt --cv examples/cv_keywords.txt --provider local

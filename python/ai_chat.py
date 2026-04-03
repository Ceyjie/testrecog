# ai_chat.py — Local Phi-3-mini medical AI via Ollama
import requests, logging
import config

log = logging.getLogger("ai")

SYSTEM_PROMPT = """You are MedPal, a friendly medical assistant robot on wheels.
You answer medical questions about symptoms, medicines, dosages, treatments,
and general health advice. Keep answers to 2-3 short sentences since they
will be spoken aloud. Always recommend a doctor for serious issues.
Never say 'As an AI'. Start answers directly and naturally."""

def ask(question: str) -> str:
    log.info(f"Question: {question[:80]}")
    try:
        r = requests.post(config.OLLAMA_URL, json={
            "model":  config.OLLAMA_MODEL,
            "prompt": question,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"num_predict": 150, "temperature": 0.7}
        }, timeout=config.OLLAMA_TIMEOUT)
        r.raise_for_status()
        answer = r.json().get("response", "").strip()
        log.info(f"Answer: {answer[:80]}")
        return answer or "I'm not sure. Please consult a doctor."
    except requests.exceptions.ConnectionError:
        log.error("Ollama not running. Start: ollama serve")
        return "My brain is offline. Please start Ollama first."
    except requests.exceptions.Timeout:
        log.error("Ollama timed out.")
        return "That took too long. Please try again."
    except Exception as e:
        log.error(f"AI error: {e}")
        return "Sorry, I had trouble with that question."

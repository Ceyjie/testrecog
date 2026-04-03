# person_registry.py — Manage registered persons
import os, json, logging
import config

log = logging.getLogger("registry")

REGISTRY_FILE = os.path.join(config.PERSONS_DIR, "registry.json")

def load() -> dict:
    os.makedirs(config.PERSONS_DIR, exist_ok=True)
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Registry load error: {e}")
        return {}

def save(registry: dict):
    os.makedirs(config.PERSONS_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)
    log.info(f"Registry saved: {list(registry.keys())}")

def list_persons() -> list:
    return list(load().keys())

def get_person(name: str) -> dict:
    return load().get(name, {})

def delete_person(name: str) -> bool:
    reg = load()
    if name not in reg:
        return False
    # Delete image files
    for img in reg[name].get("images", []):
        try:
            os.remove(img)
        except:
            pass
    del reg[name]
    save(reg)
    log.info(f"Deleted person: {name}")
    return True

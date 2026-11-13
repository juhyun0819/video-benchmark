import re

def persona(persona_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", persona_id):
        raise ValueError("Persona ID must contain only letters, numbers, underscore or hyphen")
    return {"persona_id": persona_id, "synthetic_persona": True,
            "description": "Synthetic persona; no claim of shared real-world identity"}

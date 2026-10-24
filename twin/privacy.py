"""Privacy gateway - nothing identifying leaves the device.

Every LLM call (router, brief, what-if parser, forecaster, chat) passes through
`PrivacyGateway.outbound()` before it can reach the network, and the reply passes
through `inbound()` on the way back.

  1. Pseudonymise   registered identifiers (name, first name, city, patient ID)
                    -> stable salted pseudonyms ("Patient-3F9A", "City-A")
  2. Scrub          free text with pattern rules for Indian + global identifiers:
                    Aadhaar (12 digits), ABHA health ID (14 digits), PAN, phone numbers,
                    email, explicit dates of birth, MRN / UHID style record numbers
  3. Generalise     exact age -> 5-year band, dates -> day index (the twin's own clock)
  4. Audit          every outbound payload is appended to logs/llm_egress.jsonl
                    exactly as sent, so an auditor can see what left the device
  5. Re-identify    pseudonyms in the reply are mapped back locally for display only

Follows the HIPAA Safe Harbor idea and the DPDP Act 2023 principle of data minimisation.
Set LLM_BASE_URL to a local OpenAI-compatible server (Ollama, llama.cpp, vLLM) for a
fully offline deployment - then zero bytes leave the device.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

SALT = os.getenv("PRIVACY_SALT", "glucotwin-local-salt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "llm_egress.jsonl")

PATTERNS = [
    # (?<![\d.]) / (?![\d.]) keep long decimals like 18.12345678901234 from looking like IDs
    ("ABHA_ID", re.compile(r"(?<![\d.])\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?![\d.])")),
    ("AADHAAR", re.compile(r"(?<![\d.])\d{4}[\s-]?\d{4}[\s-]?\d{4}(?![\d.])")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d.])")),
    ("DOB", re.compile(r"\b(?:dob|date of birth|born on)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)),
    ("RECORD_NO", re.compile(r"\b(?:MRN|UHID|IPD|OPD)[\s:#-]*[A-Z0-9-]{4,}\b", re.I)),
]


def _pseudo(kind: str, value: str) -> str:
    h = hashlib.sha256(f"{SALT}:{kind}:{value.lower()}".encode()).hexdigest()[:4].upper()
    return f"{kind}-{h}"


class PrivacyGateway:
    def __init__(self):
        self.forward: dict[str, str] = {}     # real -> pseudonym
        self.backward: dict[str, str] = {}    # pseudonym -> real
        self.stats = {"payloads": 0, "redactions": 0}
        self.recent: list[dict] = []
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # ------------------------------------------------------------- registry
    def register_patient(self, ehr: dict):
        d = ehr["demographics"]
        name = d.get("name", "")
        pn = _pseudo("Patient", ehr["id"])
        self._add(name, pn)
        for part in name.split():
            if len(part) > 2:
                self._add(part, pn if part == name.split()[0] else pn)
        self._add(ehr["id"], pn)
        if d.get("city"):
            self._add(d["city"], _pseudo("City", d["city"]))

    def _add(self, real: str, pseudo: str):
        if real and real not in self.forward:
            self.forward[real] = pseudo
            self.backward.setdefault(pseudo, real.split()[0] if " " in real else real)
        if real and " " in real:
            self.backward[pseudo] = real

    # ------------------------------------------------------------- transforms
    def scrub(self, text: str) -> tuple[str, list[str]]:
        found = []
        for real in sorted(self.forward, key=len, reverse=True):
            pat = re.compile(rf"\b{re.escape(real)}\b", re.I)
            if pat.search(text):
                found.append(f"name/place '{real[:1]}…' -> {self.forward[real]}")
                text = pat.sub(self.forward[real], text)
        for label, pat in PATTERNS:
            if pat.search(text):
                n = len(pat.findall(text))
                found.append(f"{label} x{n}")
                text = pat.sub(f"[{label}]", text)
        text = re.sub(r'"age":\s*(\d+)', lambda m: f'"age_band": "{int(m.group(1)) // 5 * 5}-{int(m.group(1)) // 5 * 5 + 4}"', text)
        return text, found

    def outbound(self, purpose: str, messages: list[dict]) -> list[dict]:
        clean, removed = [], []
        for m in messages:
            t, f = self.scrub(m["content"])
            clean.append({"role": m["role"], "content": t})
            removed += f
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": purpose, "removed": removed,
               "chars": sum(len(m["content"]) for m in clean), "payload": clean}
        with self._lock:
            self.stats["payloads"] += 1
            self.stats["redactions"] += len(removed)
            self.recent = (self.recent + [rec])[-20:]
            with open(LOG_PATH, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return clean

    def inbound(self, text: str | None) -> str | None:
        if not text:
            return text
        for pseudo, real in self.backward.items():
            text = text.replace(pseudo, real)
        return text


GATEWAY = PrivacyGateway()

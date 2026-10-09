"""Integration adapters (M1): M3's ingress + Q-Gate modules -> the frozen contract (guide section 10).

M3's code is used as written; this file only translates shapes:
  process(raw, source, origin, state, enabled) -> (contracts.ContentItem, [Verdict])
  scan(text) -> (redacted_text, finding_types)          canary_hit(text) -> bool
  load_qgate().score(text) -> Verdict(layer="QGATE")     load_rbf().score(text) -> Verdict(layer="RBF")
"""
import json
import threading
import re
from pathlib import Path

import joblib
import numpy as np

from aegis.contracts import ContentItem, SessionState, Verdict
from aegis.grounding.prompt import CANARY
from aegis.ingress import process as m3_process
from aegis.ingress import scan as m3_scan

# ---------------------------------------------------------------- D2 / D3
# M3's synthetic canaries + the system-prompt canary (J2) + the planted secret-store canary (setup_data)
CANARIES = sorted(set(m3_scan.CANARIES) | {CANARY, "sk-canary-0000DEADBEEF"})

TYPE_NAMES = {"aws_access_key": "AWS_KEY", "payment_card": "CARD", "generic_secret": "API_KEY",
              "high_entropy_token": "HIGH_ENTROPY", "phone": "PHONE", "ipv4": "IPV4", "email": "EMAIL"}
# Emails are not redacted by default (as in the starter): the agent must see addresses so the
# deterministic layers (A2 taint, D4 allowlist) can block exfiltration, and legit replies like
# "Email queued to bob@ourcompany.com" are not counted as leaks.
SKIP_TYPES = {"email"}


def canary_hit(text: str) -> bool:
    return any(c in text for c in CANARIES)


def scan(text: str, redact_emails: bool = False) -> tuple[str, list[str]]:
    """M3's detectors (regex + Luhn + entropy); returns (redacted_text, finding_types)."""
    skip = set() if redact_emails else SKIP_TYPES
    hits = [m for m in m3_scan.scan(text)["matches"] if m["type"] not in skip]
    spans = sorted((m["start"], m["end"], TYPE_NAMES.get(m["type"], m["type"].upper())) for m in hits)
    merged: list[list] = []                                   # overlapping matches -> one redaction
    for start, end, kind in spans:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, kind])
    for start, end, kind in reversed(merged):
        text = text[:start] + f"[REDACTED:{kind}]" + text[end:]
    return text, [kind for _, _, kind in merged]


# ---------------------------------------------------------------- D1 + tagging
def label_for(origin: str) -> str:
    """Contract labels from provenance: files under secret/ are secret, FAQ/public docs are public."""
    if "secret" in origin:
        return "secret"
    if origin.startswith("public") or origin.endswith("faq.md"):
        return "public"
    return "internal"

M3_LABEL = {"public": "public", "internal": "internal", "secret": "confidential"}


def _role(permissions) -> str:
    """Contract permissions (labels a session may read) -> M3 role for its ACL table."""
    if "secret" in permissions:
        return "admin"
    return "agent" if "internal" in permissions else "user"


def process(raw: str, source: str, origin: str, state: SessionState, enabled: bool = True) -> tuple[ContentItem, list[Verdict]]:
    """Ingress: D1 ACL (M3) -> D3 canary check -> D2 redaction (M3 detectors) -> provenance + taint."""
    label = label_for(origin)
    item = ContentItem(text=raw, source=source, origin=origin, label=label, tainted=source != "user")
    if not enabled:
        return item, []
    v = []
    m3 = m3_process.process(raw or " ", source=origin or source, label=M3_LABEL[label],
                            user_role=_role(state.permissions), redact_sensitive=False)
    if not m3.accessible:                                                   # D1
        item.text = "[ACCESS DENIED]"
        v.append(Verdict(layer="D1", decision="block", score=1, reason="acl"))
    if canary_hit(item.text):                                               # D3, before redaction hides it
        v.append(Verdict(layer="D3", decision="flag", score=1, reason="canary_in_ingress"))
    item.text, found = scan(item.text)                                      # D2
    item.redactions = len(found)
    if found:
        v.append(Verdict(layer="D2", decision="redact", score=0.5, details={"types": found}))
    if item.tainted:
        state.seen_tainted = True
    if item.label in ("internal", "secret"):
        state.seen_private = True
    return item, v


# ---------------------------------------------------------------- Q-Gate / RBF
# M3's final pipeline (scripts/train_qgate_semantic.py) writes the *_hf files; the older Aegis pipeline
# (scripts/train_qgate_aegis.py) writes qgate.pkl / rbf.pkl. The app prefers M3's final model.
FINAL_QGATE, FINAL_RBF = Path("models/qgate_quantum_hf.joblib"), Path("models/qgate_rbf_hf.joblib")
FINAL_RBF_EMBEDDER = Path("models/qgate_rbf_hf_8d_embedder.joblib")
QGATE_MODEL, RBF_MODEL = Path("models/qgate.pkl"), Path("models/rbf.pkl")
REVIEW_AT, QUARANTINE_AT = 0.5, 0.8          # defaults; M3's validation-chosen threshold overrides them


def qgate_path() -> Path | None:
    return next((p for p in (FINAL_QGATE, QGATE_MODEL) if p.exists()), None)


def qgate_available() -> bool:
    return qgate_path() is not None


def _sigmoid(m: float) -> float:
    return float(1 / (1 + np.exp(-m)))


def _m3_thresholds(n_train: int) -> tuple[float, float] | None:
    """Review/quarantine thresholds from M3's training report. M3 picks the threshold on the VALIDATION split
    (max F1) on the SVM-margin scale; our scores are sigmoid(margin), so review = sigmoid(threshold)."""
    reports = sorted(Path("results").glob("qgate_hf_training_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in reports:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        q = r.get("qgate", {})
        if "threshold" in q and q.get("training_samples", n_train) == n_train:
            review = _sigmoid(float(q["threshold"]))
            return review, review + (1 - review) * 0.6      # 0.5 -> 0.8, same spacing as the defaults
    return None


def _m3_training_texts(model) -> None:
    """M3's model does not store its 256 training texts; rebuild them exactly as train_qgate_semantic.py
    selects them (stratified subset, random_state 42) and keep them only if they reproduce the stored
    features, so the 'closest known attacks' evidence is never shown for the wrong examples."""
    if getattr(model, "train_texts", None) is not None:
        return
    csv_path = Path("data/hf_prompt_injection/train.csv")
    if not csv_path.exists():
        return
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(csv_path)
    texts, y = df["text"].astype(str).tolist(), df["label"].astype(int).to_numpy()
    n = len(model.X_train)
    idx = np.arange(len(texts)) if n >= len(texts) else \
        train_test_split(np.arange(len(texts)), train_size=n, stratify=y, random_state=42)[0]
    sub = [texts[i] for i in idx]
    if len(sub) == n and np.allclose(model.embedder.transform(sub[:6]), model.X_train[:6], atol=1e-6):
        model.train_texts, model.train_labels = sub, [int(y[i]) for i in idx]


def sentences(text: str) -> list[str]:
    """Score each sentence and keep the max: an injected sentence hidden in a benign paragraph
    would otherwise be averaged away. Max 8 sentences per chunk."""
    sents = [s for s in re.split(r"(?<=[.!?:])\s+|\n+", text) if len(s.strip()) > 15][:8]
    return sents or [text if text.strip() else "(empty)"]


def _verdict(layer: str, p: float, ms: float, review: float = REVIEW_AT, quarantine: float = QUARANTINE_AT) -> Verdict:
    d = "quarantine" if p >= quarantine else "review" if p >= review else "pass"
    return Verdict(layer=layer, decision=d, score=round(p, 4),
                   details={"ms": round(ms), "review_at": round(review, 3), "quarantine_at": round(quarantine, 3)})


_QLOCK = threading.RLock()


def _qstate(x):
    """PennyLane records each circuit on a global queue, which is not thread-safe: the server's warm-up
    thread and a request thread simulating at once raise QuantumFunctionError. Serialize the simulations."""
    from aegis.qgate.kernel import quantum_state
    with _QLOCK:
        return quantum_state(x)


class QGateDetector:
    """M3's QGate (quantum-kernel SVM) behind the contract: score(text) -> Verdict(layer="QGATE")."""
    layer = "QGATE"

    def __init__(self, model, review: float = REVIEW_AT, quarantine: float = QUARANTINE_AT):
        self.model = model
        self.review_at, self.quarantine_at = review, quarantine
        self._train_states = None

    def _kernel(self, texts):
        """Fidelity kernel |<phi(x)|phi(train)>|^2 of each text against every training example,
        with the training states simulated once and cached (~1.7 s -> ~ms per call)."""
        with _QLOCK:
            if self._train_states is None:
                self._train_states = np.array([_qstate(x) for x in self.model.X_train])
        feats = self.model.embedder.transform(list(texts))
        states = np.array([_qstate(f) for f in feats])
        return np.clip(np.abs(states.conj() @ self._train_states.T) ** 2, 0.0, 1.0), feats

    def proba(self, texts) -> np.ndarray:
        """Same math as M3's QGate.score: sigmoid of the SVM margin on the fidelity kernel."""
        K, _ = self._kernel(texts)
        margins = np.asarray(self.model.model.decision_function(K), dtype=float).reshape(-1)
        return 1 / (1 + np.exp(-margins))

    def _explain(self, sentence: str) -> dict:
        """Evidence for the top sentence: its features (the qubit angles), the 16 most likely basis states of
        its real quantum state (M3's circuit), and the training examples whose states overlap most with it."""
        K, feats = self._kernel([sentence])
        psi = _qstate(feats[0])
        n = int(np.log2(len(psi)))
        top = np.argsort(-np.abs(psi) ** 2)[:16]
        out = {"features": [round(float(f), 3) for f in feats[0]], "n_qubits": n,
               "state": [{"b": format(int(i), f"0{n}b"), "p": round(float(abs(psi[i]) ** 2), 5),
                          "ph": round(float(np.angle(psi[i])), 4)} for i in top]}
        texts, labels = getattr(self.model, "train_texts", None), getattr(self.model, "train_labels", None)
        if texts is not None and labels is not None:
            order = np.argsort(-K[0])
            near = lambda lab, n: [{"text": texts[i][:160], "k": round(float(K[0, i]), 3)}
                                   for i in order if labels[i] == lab][:n]
            out["nearest_attacks"], out["nearest_benign"] = near(1, 3), near(0, 1)
        return out

    def score(self, text: str) -> Verdict:
        import time
        t = time.perf_counter()
        sents = sentences(text)
        ps = self.proba(sents)
        top = int(ps.argmax())
        v = _verdict(self.layer, float(ps[top]), 0, self.review_at, self.quarantine_at)
        v.details.update({"sentences": [{"text": s[:200], "score": round(float(p), 4)} for s, p in zip(sents, ps)],
                          "top": top, **self._explain(sents[top])})
        v.details["ms"] = round((time.perf_counter() - t) * 1000)
        return v


class RBFDetector(QGateDetector):
    """M3's classical RBF twin, scored on the features of the embedder it was trained with."""
    layer = "RBF"

    def __init__(self, embedder, rbf, review: float = REVIEW_AT, quarantine: float = QUARANTINE_AT):
        self.embedder, self.rbf = embedder, rbf
        self.review_at, self.quarantine_at = review, quarantine

    def _explain(self, sentence: str) -> dict:
        return {}

    def proba(self, texts) -> np.ndarray:
        margins = np.asarray(self.rbf.score(self.embedder.transform(list(texts))), dtype=float)
        return 1 / (1 + np.exp(-margins))                  # same sigmoid-of-margin as QGate.score


def load_qgate(path=None) -> QGateDetector:
    """M3's final model when present (with its validation-chosen threshold), else the older Aegis model."""
    from aegis.qgate.detector import QGate
    path = Path(path) if path else qgate_path()
    if path is None:
        raise FileNotFoundError("no Q-Gate model: run python -m scripts.train_qgate_semantic (M3) "
                                "or python -m scripts.train_qgate_aegis")
    model = QGate.load(path)
    if path == FINAL_QGATE:
        _m3_training_texts(model)
        th = _m3_thresholds(len(model.X_train))
        if th:
            return QGateDetector(model, *th)
    return QGateDetector(model)


def load_rbf(path=None) -> RBFDetector:
    """M3's classical twin. The final one needs sentence-transformers (semantic features)."""
    if path is None and FINAL_RBF.exists() and FINAL_RBF_EMBEDDER.exists():
        from aegis.qgate.baseline_rbf import RBFBaseline
        return RBFDetector(joblib.load(FINAL_RBF_EMBEDDER), RBFBaseline.load(FINAL_RBF))
    bundle = joblib.load(Path(path) if path else RBF_MODEL)
    return RBFDetector(bundle["embedder"], bundle["rbf"])

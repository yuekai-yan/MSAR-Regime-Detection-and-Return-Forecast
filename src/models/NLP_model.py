# nlp_embed_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


class SpeechEmbeddingAnalyzer:
    """
    Embedding-only, low-cost NLP feature extractor for CEO transcripts.
    - Model default: text-embedding-3-small (1536-dim)
    - Outputs: sentiment, emotions, uncertainty, topics, and fingerprint features
    """

    def __init__(
        self,
        emb_model: str = "text-embedding-3-small",
        max_chars_per_chunk: int = 4000,
        topic_topk: int = 5,
        anchor_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        self.emb_model = emb_model
        self.max_chars_per_chunk = max_chars_per_chunk
        self.topic_topk = topic_topk
        self.client = client or OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        # ------- Default lexicons & anchors -------
        self.HEDGING_WORDS = {
            "maybe","perhaps","possibly","might","could","appears","suggests","approximately","around",
            "likely","unlikely","estimate","roughly","potentially","somewhat","about","seems","presumably",
            "arguably","in our view","we believe","it seems that","we think","we assume","to some extent",
            "as far as we know","sort of","kind of","at this stage","it depends","subject to","open question",
            "under review","not sure","unclear","tentative","ongoing discussion","rough estimate","early stage"
        }

        self.SELF_REF_WORDS = {
            "i","we","our","ours","us","my","me","myself","ourselves","mine","the team","our company",
            "our management","our leadership","we believe","we think","we plan","we expect","we anticipate","we aim",
            "i believe","i think","i expect","i feel","i hope","i know"
        }

        self.JARGON_WORDS = {
            "guidance","run-rate","headwinds","tailwinds","margin","gm","ebitda","capex","opex","margins",
            "gross margin","opex leverage","buyback","liquidity","cash flow","fiscal","restructuring","synergy",
            "ltv","cac","cohort","gtm","sg&a","topline","bottom line","yoy","qoq","sequential","churn",
            "retention","unit economics","operating leverage","gross profit","operating income","net income",
            "earnings per share","free cash flow","capital allocation","dividend yield","working capital",
            "balance sheet","debt leverage","credit facility","share repurchase","pricing power","market share",
            "competitive landscape","cost structure","supply chain","demand environment","macro headwinds",
            "macro backdrop","revenue growth","operational efficiency","productivity gains","cost optimization",
            "guidance update","pipeline visibility"
        }

        SENT_POS = (
            "very positive outlook, strong growth, confident tone, robust demand, "
            "encouraging momentum, solid performance, and constructive sentiment"
        )
        SENT_NEG = (
            "very negative outlook, weak demand, disappointing performance, "
            "pessimistic tone, soft revenue, declining margins, and challenging environment"
        )
        EMOTION_PHRASES = {
            "joy": "tone of joy, satisfaction, enthusiasm, upbeat optimism, celebration of success",
            "anger": "tone of anger, frustration, irritation, or criticism toward performance or external factors",
            "fear": "tone of fear, anxiety, risk aversion, concern about uncertainty or volatility",
            "sadness": "tone of sadness, regret, disappointment about missed goals or losses",
            "trust": "tone of trust, credibility, transparency, dependable leadership, and confidence in execution",
            "surprise": "tone of surprise, unexpected events, sudden shifts, or unanticipated results",
        }
        UNCERTAINTY_POS = (
            "high uncertainty, ambiguous forecasts, conditional statements, "
            "unknown variables, risk factors, and contingency planning"
        )
        UNCERTAINTY_NEG = (
            "low uncertainty, clear guidance, detailed forecasts, and precise communication of plans"
        )
        FINGERPRINT_PHRASES = {
            "optimism_pos": "strong optimism, confident growth, upbeat guidance, belief in sustained momentum",
            "optimism_neg": "low optimism, cautious or pessimistic outlook, uncertain demand",
            "hedging_pos":  "frequent hedging or qualified statements, use of words like 'maybe' or 'possibly'",
            "hedging_neg":  "direct and assertive statements, clear and confident declarations",
            "self_ref_pos": "frequent self-reference, leadership emphasis, use of 'we' or 'our'",
            "self_ref_neg": "impersonal or detached tone, avoiding self-reference",
            "complexity_pos":"high linguistic complexity, long sentences, sophisticated wording",
            "complexity_neg":"simple direct language, short sentences, plain speech",
        }
        TOPIC_LABELS = [
            "revenue growth","profitability and margins","cost control and opex","cash flow and liquidity",
            "capital expenditure and capex","ai strategy and product roadmap","macroeconomic environment",
            "supply chain and inventory","regulation and compliance","mergers and acquisitions",
            "share buyback and dividend","customer demand and pipeline","pricing power and unit economics",
            "geographic expansion","hiring and headcount"
        ]

        # allow override via anchor_config
        self.SENT_POS = (anchor_config or {}).get("SENT_POS", SENT_POS)
        self.SENT_NEG = (anchor_config or {}).get("SENT_NEG", SENT_NEG)
        self.EMOTION_PHRASES = (anchor_config or {}).get("EMOTION_PHRASES", EMOTION_PHRASES)
        self.UNCERTAINTY_POS = (anchor_config or {}).get("UNCERTAINTY_POS", UNCERTAINTY_POS)
        self.UNCERTAINTY_NEG = (anchor_config or {}).get("UNCERTAINTY_NEG", UNCERTAINTY_NEG)
        self.FINGERPRINT_PHRASES = (anchor_config or {}).get("FINGERPRINT_PHRASES", FINGERPRINT_PHRASES)
        self.TOPIC_LABELS = (anchor_config or {}).get("TOPIC_LABELS", TOPIC_LABELS)

        # will be built on first use
        self._anchor_vecs: Dict[str, np.ndarray] = {}

    # --------------------- public API --------------------- #
    def transform(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        """Transform a list of raw transcripts into feature dicts."""
        self._ensure_anchors_built()
        out = []
        for txt in texts:
            out.append(self._analyze_text(txt))
        return out

    def transform_file(self, path: str | Path) -> Dict[str, Any]:
        self._ensure_anchors_built()
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        row = self._analyze_text(text)
        row["file"] = p.name
        return row

    def transform_folder(self, folder: str | Path, pattern: str = "*.txt") -> pd.DataFrame:
        """Batch process a folder of .txt transcripts -> DataFrame."""
        self._ensure_anchors_built()
        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        rows = []
        for f in tqdm(files, desc="Analyzing (embeddings)"):
            try:
                rows.append(self.transform_file(f))
            except Exception as e:
                rows.append({"file": f.name, "error": str(e)})
        return pd.DataFrame(rows)

    # ------------------- internals ------------------- #
    def _ensure_anchors_built(self):
        if self._anchor_vecs:
            return
        phrases = (
            [self.SENT_POS, self.SENT_NEG, self.UNCERTAINTY_POS, self.UNCERTAINTY_NEG]
            + list(self.EMOTION_PHRASES.values())
            + list(self.FINGERPRINT_PHRASES.values())
            + self.TOPIC_LABELS
        )
        embs = self._embed_phrases(phrases)
        self._anchor_vecs = {ph: embs[ph] for ph in phrases}

    def _chunk_text(self, s: str) -> List[str]:
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) <= self.max_chars_per_chunk:
            return [s]
        parts, start = [], 0
        while start < len(s):
            end = min(start + self.max_chars_per_chunk, len(s))
            cut = s.rfind(".", start, end)
            if cut == -1 or cut <= start + self.max_chars_per_chunk * 0.5:
                cut = end
            parts.append(s[start:cut].strip())
            start = cut
        return [p for p in parts if p]

    def _embed_text(self, text: str) -> np.ndarray:
        chunks = self._chunk_text(text)
        vecs = []
        for ch in chunks:
            emb = self.client.embeddings.create(model=self.emb_model, input=ch).data[0].embedding
            vecs.append(np.array(emb, dtype=np.float32))
        if not vecs:
            return np.zeros(1536, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def _embed_phrases(self, phrases: List[str]) -> Dict[str, np.ndarray]:
        out = {}
        for p in phrases:
            emb = self.client.embeddings.create(model=self.emb_model, input=p).data[0].embedding
            out[p] = np.array(emb, dtype=np.float32)
        return out

    @staticmethod
    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        du, dv = np.linalg.norm(u), np.linalg.norm(v)
        if du == 0 or dv == 0:
            return 0.0
        return float(np.dot(u, v) / (du * dv))

    @staticmethod
    def _cos_to_01(c: float) -> float:
        return max(0.0, min(1.0, 0.5 * (c + 1.0)))

    @staticmethod
    def _tokenize_words(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z']+", s.lower())

    @staticmethod
    def _fk_grade(text: str) -> float:
        sentences = [seg for seg in re.split(r"[.!?]+", text) if seg.strip()]
        words = SpeechEmbeddingAnalyzer._tokenize_words(text)

        def _syllables(w: str) -> int:
            syl = re.findall(r"[aeiouy]+", w.lower())
            return max(1, len(syl))

        syl_count = sum(_syllables(w) for w in words) or 1
        W = len(words) or 1
        S = max(1, len(sentences))
        return 0.39 * (W / S) + 11.8 * (syl_count / W) - 15.59

    def _anchor(self, phrase: str) -> np.ndarray:
        return self._anchor_vecs[phrase]

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"error": "Empty text"}

        doc_vec = self._embed_text(text)
        words = self._tokenize_words(text)
        wc = len(words)
        counts = Counter(words)

        # lexical features
        hedging_cnt = sum(counts[w] for w in self.HEDGING_WORDS)
        selfref_cnt = sum(counts[w] for w in self.SELF_REF_WORDS)

        # phrase-aware jargon
        lowered = " " + re.sub(r"\s+", " ", text.lower()) + " "
        jargon_cnt = 0
        for j in self.JARGON_WORDS:
            jargon_cnt += lowered.count(f" {j} ") if " " in j else counts[j]

        # sentiment (contrast)
        c_pos = self._cosine(doc_vec, self._anchor(self.SENT_POS))
        c_neg = self._cosine(doc_vec, self._anchor(self.SENT_NEG))
        sent_score = (c_pos - c_neg + 1.0) / 2.0
        if sent_score > 0.58:
            sent_label = "Positive"
        elif sent_score < 0.42:
            sent_label = "Negative"
        else:
            sent_label = "Neutral"
        sent_conf = float(abs(sent_score - 0.5) * 2.0)

        # emotions
        emotions = {
            k: self._cos_to_01(self._cosine(doc_vec, self._anchor(v)))
            for k, v in self.EMOTION_PHRASES.items()
        }

        # uncertainty
        u_hi = self._cosine(doc_vec, self._anchor(self.UNCERTAINTY_POS))
        u_lo = self._cosine(doc_vec, self._anchor(self.UNCERTAINTY_NEG))
        uncertainty = float((u_hi - u_lo + 1.0) / 2.0)

        # fingerprints
        optimism = self._cos_to_01(
            self._cosine(doc_vec, self._anchor(self.FINGERPRINT_PHRASES["optimism_pos"]))
            - self._cosine(doc_vec, self._anchor(self.FINGERPRINT_PHRASES["optimism_neg"]))
        )
        hedging_rate = 0.0 if wc == 0 else hedging_cnt / wc
        self_ref_rate = 0.0 if wc == 0 else selfref_cnt / wc

        # jargon density & complexity
        jargon_density = 0.0 if wc == 0 else jargon_cnt / wc
        complexity_fk_grade = float(np.clip(self._fk_grade(text), 0.0, 20.0))

        # topics
        sims = []
        for lab in self.TOPIC_LABELS:
            sims.append(max(0.0, self._cosine(doc_vec, self._anchor(lab))))
        sims = np.array(sims, dtype=np.float32)
        weights = sims / sims.sum() if sims.sum() > 0 else np.ones_like(sims) / len(sims)
        top_idx = np.argsort(-weights)[: self.topic_topk]
        topics = "; ".join(f"{self.TOPIC_LABELS[i]}({weights[i]:.2f})" for i in top_idx)

        return {
            "sentiment": sent_label,
            "sent_conf": round(float(sent_conf), 4),
            "uncertainty": round(float(uncertainty), 4),
            "emo_joy": round(float(emotions["joy"]), 4),
            "emo_anger": round(float(emotions["anger"]), 4),
            "emo_fear": round(float(emotions["fear"]), 4),
            "emo_sadness": round(float(emotions["sadness"]), 4),
            "emo_trust": round(float(emotions["trust"]), 4),
            "emo_surprise": round(float(emotions["surprise"]), 4),
            "fp_optimism": round(float(optimism), 6),
            "fp_hedging_rate": round(float(hedging_rate), 6),
            "fp_self_reference_rate": round(float(self_ref_rate), 6),
            "fp_jargon_density": round(float(jargon_density), 6),
            "fp_complexity_fk_grade": round(float(complexity_fk_grade), 6),
            "topics": topics,
        }


# -------------------- minimal usage -------------------- #
if __name__ == "__main__":
    # 1) 
    analyzer = SpeechEmbeddingAnalyzer()
    example_text = "We believe our AI roadmap and strong demand will support profitable growth, despite macro headwinds."
    print(analyzer.transform([example_text])[0])

    # 2) （.txt）
    # in_dir = r"E:\EPFLcourse\ADA\2025\Project\dataverse_files\Transcripts\2016_2020"
    # df = analyzer.transform_folder(in_dir)
    # df.to_excel(Path(in_dir).with_name("transcripts_2016_2020_analysis.xlsx"), index=False)
    # print("Saved.")

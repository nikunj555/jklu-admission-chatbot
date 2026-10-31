"""The JKLU agent: question -> retrieve -> evidence check -> answer | clarify | escalate."""
from __future__ import annotations

import os
import re
import time
import uuid

from . import config as C
from . import evidence as E
from . import textproc as T
from .index import KnowledgeBase
from .learning import SelfAgenticMemory
from .llm import INSUFFICIENT, Generator
from .utils import log

_GREETING_WORDS = r"(?:hi+|hello+|hey+|hola|namaste|good\s+(?:morning|afternoon|evening)|greetings|howdy|sup)"
_GREETING = re.compile(
    rf"^\s*(?:{_GREETING_WORDS}[\s,]*)+"
    r"(?:\s+(?:there|assistant|bot|friend|team))?"
    r"(?:\s*,?\s*(?:how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|hope\s+you\s+are\s+well|what's\s+up))?"
    r"[\s!.,?]*$",
    re.I
)

_HOW_ARE_YOU = re.compile(
    r"^\s*(?:how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|how\s+do\s+you\s+do)[\s!.,?]*$",
    re.I
)

_IDENTITY = re.compile(
    r"^\s*(?:(?:" + _GREETING_WORDS + r"[\s,]+)?(?:"
    r"who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|"
    r"tell\s+me\s+who\s+you\s+are|what\s+is\s+your\s+name|who\s+made\s+you|"
    r"what\s+can\s+you\s+do|how\s+do\s+you\s+work|what\s+do\s+you\s+do|"
    r"can\s+you\s+introduce\s+yourself|please\s+introduce\s+yourself|"
    r"help(?:\s+me)?|can\s+you\s+help(?:\s+me)?|i\s+need\s+help|what\s+can\s+i\s+ask)"
    r")[\s?!.]*$",
    re.I
)

_THANKS = re.compile(
    r"^\s*(?:thanks?|thank\s+you|thx|ok(?:ay)?|great|cool|got\s+it|awesome|perfect|bye|goodbye)"
    r"(?:\s+(?:so\s+much|a\s+lot|very\s+much|assistant))?"
    r"[\s!.,?]*$",
    re.I
)

_FOLLOW = re.compile(r"^\s*(and|what about|how about|also|then|what of|same for|for)\b", re.I)
_PRONOUN = re.compile(r"\b(it|its|that|this|they|them|those|these|there)\b", re.I)

WELCOME = (
    "Hello! I am the **JK Lakshmipat University (JKLU) Information & Student Companion Assistant**.\n\n"
    "I am here to guide you with verified information from official JKLU records. You can ask me about:\n\n"
    "- 🎓 **Admissions & Eligibility**: Requirements for B.Tech, B.Des, BBA, M.Tech, and PhD\n"
    "- 💰 **Fees & Scholarships**: Tuition fees, hostel charges, refund policies, and merit scholarships\n"
    "- 📈 **Placements & Packages**: Placement statistics, highest & average CTC packages, and top recruiters\n"
    "- 📚 **Programmes & Faculty**: Curricula, specializations, academic calendar, and faculty profiles\n"
    "- 🏡 **Hostel & Campus Life**: Residence rules, curfew timings, dining mess, sports, and health centre\n"
    "- 🎉 **Campus Events & Fests**: Annual festivals including Sabrang, Spardha, HackJKLU, and Aarambh\n\n"
    "How can I assist you with JKLU today?"
)

HOW_ARE_YOU_REPLY = (
    "Hello! I'm doing well, thank you! How can I assist you with JK Lakshmipat University (JKLU) today? "
    "Feel free to ask about admissions, courses, fees, scholarships, faculty, or campus life."
)

THANKS_REPLY = (
    "You're very welcome! If you have any more questions about JKLU admissions, programmes, fees, or student life, "
    "just let me know!"
)

HELP_REPLY = WELCOME

SUGGESTIONS = [
    "What is the eligibility for B.Tech admission?",
    "What is the tuition fee for B.Tech in 2026-27?",
    "What scholarships are available for JEE Main students?",
    "What documents are required for admission?",
    "Who are the faculty members in the CSE department?",
    "What hostel facilities are available?",
]


def _snippet(text: str, n: int = 320) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


class JKLUAgent:
    def __init__(self, kb: KnowledgeBase | None = None, generator: Generator | None = None,
                 provider: str | None = None, model: str | None = None):
        self.kb = kb or KnowledgeBase.load()
        self.generator = generator or Generator(provider=provider, model=model)
        self._contact_lines = self._extract_contacts()
        stat_chunks = [
            c for c in self.kb.chunks
            if any(k in c.get("text", "").lower() for k in ("22.5 lpa", "8.10 lpa", "18.07 lpa"))
        ]
        guide_chunks = [
            c for c in self.kb.chunks
            if c.get("chunk_id", "").startswith("eaad97fa55")
        ]
        self._placement_stat_chunks = stat_chunks
        self._placement_guide_chunks = guide_chunks
        self._placement_chunks = stat_chunks + guide_chunks
        fest_cids = ("8c614e8e4e-0005", "aa0862ce20-0003", "8c614e8e4e-0002", "8c614e8e4e-0000", "da997aecf9-0004")
        self._festival_chunks = [c for cid in fest_cids if (c := self.kb.get_chunk(cid))]
        # Verified BBA key chunks (eligibility, fees, programmes, curriculum)
        bba_cids = ("68a5ed8c52-0001", "7a3fc7b2fe-0001", "97de2bec53-0001", "97de2bec53-0002", "595e59e931-0000", "595e59e931-0001", "e40c447c3b-0000", "e67cb3f1c4-0000", "cee6516655-0004", "b167065a58-0002")
        self._bba_chunks = [c for cid in bba_cids if (c := self.kb.get_chunk(cid))]
        # Verified B.Des key chunks (eligibility, fees, specialisations, curriculum)
        bdes_cids = ("68a5ed8c52-0000", "7a3fc7b2fe-0000", "e08a507321-0000", "97de2bec53-0001", "26d54c68ee-0000", "4c16ed4b45-0000", "367306ad64-0000", "bcd89dff70-0000", "b167065a58-0002", "cc1be9fb7a-0000")
        self._bdes_chunks = [c for cid in bdes_cids if (c := self.kb.get_chunk(cid))]
        # Verified B.Tech key chunks (eligibility, fees, admission process, curriculum)
        btech_cids = ("68a5ed8c52-0000", "7a3fc7b2fe-0000", "314bd551e8-0002", "3a0732ce2f-0000", "e8f8896b6f-0001", "fc39067d7f-0000", "13688686c5-0006")
        self._btech_chunks = [c for cid in btech_cids if (c := self.kb.get_chunk(cid))]
        self.memory = SelfAgenticMemory(kb=self.kb)
        log.info("Agent ready | chunks=%d | retrieval=%s | llm=%s | self_rag_memory=%d",
                 len(self.kb.chunks), "semantic+bm25" if self.kb.semantic else "lexical+bm25",
                 self.generator.provider, len(self.memory.entries))

    # --------------------------------------------------------------- contacts
    def _extract_contacts(self) -> dict:
        """Structured office contacts parsed from the crawled contact/hostel pages (never hard-coded)."""
        email_re = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
        phone_re = re.compile(r"\+?\d[\d ()-]{7,}\d")
        offices: dict[str, dict] = {}
        target_offices = ("Registrar", "Admissions", "Placements", "Human Resources", "Media")

        for c in self.kb.chunks:
            if c["category"] != "contact":
                continue
            lines = [l.strip() for l in c["text"].split("\n") if l.strip()]
            for i, line in enumerate(lines):
                clean_l = re.sub(r"^#+\s*", "", line).strip()
                for office in target_offices:
                    # Match exact office name or heading like '# Registrar'
                    if clean_l.lower() == office.lower() or (len(clean_l.split()) <= 2 and office.lower() in clean_l.lower()):
                        cur = offices.setdefault(office, {"office": office, "phone": None, "email": None})
                        # Look at the adjacent lines for phone and email
                        for j in range(max(0, i - 1), min(len(lines), i + 4)):
                            cand = lines[j]
                            pm = phone_re.search(cand)
                            em = email_re.search(cand)
                            if pm and not cur["phone"] and len(pm.group(0).strip()) >= 10:
                                cur["phone"] = pm.group(0).strip()
                            if em and not cur["email"]:
                                cur["email"] = em.group(0).strip()

        # Fallback to general scan if any office still missing details
        for c in self.kb.chunks:
            if c["category"] != "contact":
                continue
            text = " ".join(c["text"].split())
            if not offices.get("Registrar", {}).get("phone"):
                m = re.search(r"Registrar\s*(\+?\d[\d ()-]{7,}\d)", text, re.I)
                if m:
                    offices.setdefault("Registrar", {"office": "Registrar", "phone": None, "email": None})["phone"] = m.group(1).strip()
            if not offices.get("Admissions", {}).get("phone"):
                m = re.search(r"(\+?91\s*830\s*222\s*3344|\+?91\s*8302223344)", text)
                if m:
                    offices.setdefault("Admissions", {"office": "Admissions", "phone": None, "email": None})["phone"] = m.group(1).strip()
            if not offices.get("Admissions", {}).get("email"):
                m = re.search(r"(admissions?@jklu\.edu\.in)", text, re.I)
                if m:
                    offices.setdefault("Admissions", {"office": "Admissions", "phone": None, "email": None})["email"] = m.group(1).strip()

        hostel_emails: list[str] = []
        for c in self.kb.chunks:
            if c["category"] == "hostel" and "warden" in c["text"].lower():
                for e in email_re.findall(c["text"]):
                    if "warden" in e.lower() and e not in hostel_emails:
                        hostel_emails.append(e)
        order = ["Admissions", "Registrar", "Placements", "Human Resources", "Media"]
        return {"offices": [offices[o] for o in order if o in offices], "hostel_emails": hostel_emails[:3]}

    def _office_direct_answer(self, question: str) -> tuple[str, list[dict]] | None:
        """For direct office contact queries, return (concise_answer, contact_source) immediately."""
        q = question.lower()
        is_contact_q = bool(re.search(r"\b(phone|call|email|contact|reach|number|helpline|office)\b", q))
        if not is_contact_q:
            return None
        match = None
        for o in self._contact_lines.get("offices", []):
            if o["office"].lower() in q:
                match = o
                break
        if not match:
            return None
        parts = []
        if match.get("phone"):
            parts.append(f"Phone: {match['phone']}")
        if match.get("email"):
            parts.append(f"Email: {match['email']}")
        if not parts:
            return None
        contact_chunk = next((c for c in self.kb.chunks if c.get("category") == "contact"), None)
        sources = [contact_chunk] if contact_chunk else []
        ans = f"{match['office']}: {', '.join(parts)} [1]"
        return ans, sources

    def _contacts_for(self, routed: list[str]) -> list[dict]:
        out: list[dict] = []
        if "hostel" in routed and self._contact_lines["hostel_emails"]:
            out.append({"office": "Hostel office (wardens)", "phone": None,
                        "email": ", ".join(self._contact_lines["hostel_emails"])})
        out.extend(o for o in self._contact_lines["offices"] if o["office"] in ("Admissions", "Registrar"))
        return out

    def _escalation_text(self, routed: list[str], lead: str) -> str:
        lines = [lead, ""]
        for c in self._contacts_for(routed):
            lines.append(f"{c['office']}: " + " | ".join(x for x in (c["phone"], c["email"]) if x))
        lines.append(f"You can also reach the right office through {C.CONTACT_PAGE}")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------- utilities
    @staticmethod
    def contextualize(question: str, history: list[dict] | None) -> str:
        """Make follow-ups self-contained using conversation context and entity carry-over."""
        if not history:
            return question
        prev_user = [h["content"] for h in history if h.get("role") == "user" and h.get("content")]
        if not prev_user:
            return question
        last_q = prev_user[-1]
        terms = T.content_terms(question)
        q_low = question.lower().strip()

        is_follow_up = (
            bool(_FOLLOW.match(question))
            or (bool(_PRONOUN.search(question)) and len(terms) <= 8)
            or bool(re.search(r"\b(more detail|explain (it )?(more )?simply|simpler|elaborate|tell me more|more info|what else)\b", q_low))
            or (len(terms) == 1 and terms[0] not in T.GENERIC_HEAD_STEMS)
        )
        if not is_follow_up:
            return question

        # Carry over active programme if the follow-up doesn't specify a different one
        has_prog = any(re.search(r"\b" + p + r"\b", q_low, re.I) for p in ("btech", "b.tech", "mtech", "m.tech", "bdes", "b.des", "mdes", "m.des", "bba", "phd"))
        if not has_prog:
            prog_match = re.search(r"\b(b\.?tech|m\.?tech|b\.?des|m\.?des|bba|ph\.?d)\b", last_q, re.I)
            if prog_match:
                return f"{last_q} {question} for {prog_match.group(0)}"

        return f"{last_q} {question}"

    def _sources(self, passages: list[dict], numbers: list[int] | None = None) -> list[dict]:
        """`numbers` keeps the original [n] citation numbers when only some passages are listed."""
        out = []
        for i, p in zip(numbers or range(1, len(passages) + 1), passages):
            out.append({
                "n": i, "title": p.get("title"), "url": p.get("source_url"), "category": p.get("category"),
                "page": p.get("page"), "page_end": p.get("page_end"), "section": p.get("section") or None,
                "academic_year": p.get("academic_year"), "source_type": p.get("source_type"),
                "score": round(float(p.get("dense_score") if p.get("dense_score") is not None
                                     else p.get("fused_score", 0.0)), 4),
                "snippet": _snippet(p["text"]),
            })
        return out

    @staticmethod
    def _decompose_query(query: str) -> list[str]:
        q = query.strip()
        parts = re.split(r"\b(?:and|as well as|also|including|along with)\b", q, flags=re.I)
        parts = [p.strip(" ?.,") for p in parts if len(p.strip()) > 3]
        if len(parts) >= 2:
            if all(len(T.content_terms(p)) >= 1 for p in parts):
                return parts
        return []

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, category: str | None = None, top_k: int = C.TOP_K,
            history: list[dict] | None = None) -> dict:
        t0 = time.time()
        raw_question = (question or "").strip()[:600]
        analysis = T.understand_query(raw_question, history)

        base = {"status": "", "answer": "", "confidence": 0.0, "reason": "", "sources": [],
                "clarification": None, "category": None, "query_used": raw_question, "provider": None,
                "suggestions": [], "contacts": [], "contact_page": C.CONTACT_PAGE,
                "request_id": uuid.uuid4().hex[:12]}

        def done(**kw) -> dict:
            base.update(kw)
            base["latency_ms"] = int((time.time() - t0) * 1000)
            if base.get("status") == "answer" and hasattr(self, "memory"):
                try:
                    # If query contained unrelated parts, store pattern under clean extracted_q rather than raw_question
                    learn_raw = extracted_q if analysis.unrelated_parts else raw_question
                    self.memory.reflect_and_learn(learn_raw, extracted_q, passages if 'passages' in locals() else [], base)
                except Exception as _e:
                    log.warning("Memory reflection error: %s", _e)
            return base

        if not raw_question:
            return done(status="clarify", answer="Please type a question about JKLU.", reason="empty question",
                        clarification="Please type a question about JKLU.")
        if _HOW_ARE_YOU.match(raw_question):
            return done(status="smalltalk", answer=HOW_ARE_YOU_REPLY, reason="greeting", confidence=1.0,
                        suggestions=SUGGESTIONS)
        if _GREETING.match(raw_question):
            return done(status="smalltalk", answer=WELCOME, reason="greeting", confidence=1.0,
                        suggestions=SUGGESTIONS)
        if _IDENTITY.match(raw_question):
            return done(status="smalltalk", answer=HELP_REPLY, reason="identity/help", confidence=1.0,
                        suggestions=SUGGESTIONS)
        if _THANKS.match(raw_question):
            return done(status="smalltalk", answer=THANKS_REPLY, reason="closing", confidence=1.0,
                        suggestions=SUGGESTIONS)

        # Handle queries that contain zero JKLU or academic requests
        if analysis.is_off_topic_only:
            lead = ("I specialize in answering questions about JK Lakshmipat University (JKLU)—including admissions, "
                    "programmes, faculty, fees, scholarships, hostel rules, academic calendars, campus facilities, "
                    "and placements. How can I help you with JKLU?")
            return done(status="escalate", answer=lead, reason="pure off-topic query without university context",
                        confidence=0.0, suggestions=SUGGESTIONS)

        if category and category not in C.CATEGORIES:
            category = None

        extracted_q = analysis.primary_query
        query = self.contextualize(extracted_q, history)
        base["query_used"] = query

        # Instant, concise resolution for direct office contact queries
        direct_contact = self._office_direct_answer(query)
        if direct_contact:
            ans, contact_sources = direct_contact
            return done(status="answer", answer=ans, sources=self._sources(contact_sources) if contact_sources else [],
                        confidence=1.0, category="contact", reason="direct office contact details from official directory",
                        provider="extractive")

        # Multi-query retrieval formulations: contextualized, extracted, raw, and sub-questions
        queries_to_search = [query]
        if extracted_q.lower() != query.lower():
            queries_to_search.append(extracted_q)

        # Self-Agentic Memory: recall learned query cluster if available (use clean extracted/contextualized question)
        memory_match = self.memory.recall(extracted_q) if hasattr(self, "memory") else None
        if not memory_match and hasattr(self, "memory") and query != extracted_q:
            memory_match = self.memory.recall(query)
        if memory_match:
            log.info("[Self-Agentic RAG] Recalled learned cluster [%s] (sim=%.3f, tier=%s) for query: '%s'",
                     memory_match.entry.id, memory_match.similarity, memory_match.tier, extracted_q[:50])
            if memory_match.entry.canonical_query.lower() not in [q.lower() for q in queries_to_search]:
                queries_to_search.append(memory_match.entry.canonical_query)
        for rq in analysis.retrieval_queries:
            if rq.lower() not in [q.lower() for q in queries_to_search]:
                queries_to_search.append(rq)
        for sq in analysis.sub_questions:
            if sq.lower() not in [q.lower() for q in queries_to_search]:
                queries_to_search.append(sq)

        is_fac_roster = bool(analysis.entities.get("departments")) and (
            "faculty" in analysis.routed_categories
            or any(w in extracted_q.lower() for w in ("faculty", "professor", "teach", "who teaches", "hod", "dean"))
        )
        search_top_k = max(top_k, 25) if is_fac_roster else max(1, min(int(top_k), 10))

        fallback_outcome = "not_needed"
        try:
            # Fast single primary search first
            primary_search = self.kb.search(query, top_k=search_top_k, category=category)
            all_results = list(primary_search.get("results", []))
            seen_chunk_ids = {r["chunk_id"] for r in all_results}

            # Only execute extra searches if needed (department roster, multi-question, or low initial recall)
            extra_queries = []
            if is_fac_roster or analysis.sub_questions or len(all_results) < 3:
                for q_str in queries_to_search:
                    if q_str.lower() != query.lower():
                        extra_queries.append(q_str)

            for q_str in extra_queries:
                try:
                    s_res = self.kb.search(q_str, top_k=search_top_k, category=category)
                    for r in s_res.get("results", []):
                        if r["chunk_id"] not in seen_chunk_ids:
                            seen_chunk_ids.add(r["chunk_id"])
                            all_results.append(r)
                except Exception as e:
                    log.warning("sub-search failed for '%s': %s", q_str, e)

            search = primary_search or {"results": [], "routed": analysis.routed_categories}
            search["results"] = all_results[:search_top_k * 2] if is_fac_roster else all_results[:top_k * 2]
        except Exception as exc:
            log.error("retrieval failed: %s", exc)
            return done(status="escalate", reason=f"retrieval error: {type(exc).__name__}",
                        answer=self._escalation_text([], "Sorry, I ran into a problem searching the knowledge base."))

        passages = search["results"]

        # Check if query targets a specific person not matching the memory match
        target_names = analysis.entities.get("faculty_names", [])
        person_mismatch = False
        if memory_match and target_names:
            mem_cq_low = (memory_match.entry.canonical_query + " " + " ".join(memory_match.entry.patterns[:4])).lower()
            if not any(tn.lower() in mem_cq_low for tn in target_names):
                person_mismatch = True

        # Self-Agentic RAG: Prepend historically proven chunks for this learned cluster
        if memory_match and not person_mismatch and memory_match.entry.chunk_ids and memory_match.tier in ("exact", "strong"):
            recalled_chunks = [
                dict(c, dense_score=c.get("dense_score") or 0.88, fused_score=c.get("fused_score") or 0.88)
                for cid in memory_match.entry.chunk_ids
                if (c := self.kb.get_chunk(cid))
            ]
            if recalled_chunks:
                rc_ids = {c["chunk_id"] for c in recalled_chunks}
                passages = recalled_chunks + [p for p in passages if p["chunk_id"] not in rc_ids]
                search["results"] = passages
                if memory_match.entry.category and memory_match.entry.category not in search.get("routed", []):
                    search.setdefault("routed", []).append(memory_match.entry.category)
        # Expand passages with adjacent table/sequential chunks if available
        expanded_passages = []
        seen_p_ids = set()
        for p in passages:
            if p["chunk_id"] not in seen_p_ids:
                seen_p_ids.add(p["chunk_id"])
                expanded_passages.append(p)
            is_fee_or_table = p.get("is_table") or p.get("category") == "fees" or "fee" in p.get("title", "").lower()
            if is_fee_or_table:
                wants_full = any(w in query.lower() for w in ("hostel", "mess", "total", "full", "all", "complete", "cost"))
                if wants_full and p.get("category") == "fees":
                    for dc in self.kb.chunks:
                        if dc.get("doc_id") == p.get("doc_id") and dc["chunk_id"] not in seen_p_ids:
                            seen_p_ids.add(dc["chunk_id"])
                            expanded_passages.append(dc)
                else:
                    for adj_id in (p.get("next_chunk_id"), p.get("prev_chunk_id")):
                        if adj_id and adj_id not in seen_p_ids:
                            adj = self.kb.get_chunk(adj_id)
                            if adj:
                                seen_p_ids.add(adj_id)
                                expanded_passages.append(adj)
        passages = expanded_passages

        # Ensure verified placement records are prioritized for placement/package/CTC queries
        q_comb = (query + " " + extracted_q).lower()
        has_placement_kw = any(w in q_comb for w in ("placement", "placements", "recruit", "recruiter", "recruiters", "job", "career", "salary", "package", "ctc", "lpa"))
        is_stat_q = any(
            w in q_comb
            for w in ("package", "packages", "ctc", "salary", "highest", "average", "lpa", "recruiter", "recruiters", "highest ctc", "average ctc", "highest package", "average package", "rate", "percentage")
        )
        is_guide_q = any(
            w in q_comb
            for w in ("document", "eligib", "checklist", "guideline", "form", "registration", "resume", "cv", "portfolio", "interview prep")
        )
        is_placement_q = (is_stat_q and has_placement_kw) or (is_guide_q and has_placement_kw) or any(
            w in q_comb for w in ("placement", "placements", "campus recruitment")
        )
        if is_placement_q and (self._placement_stat_chunks or self._placement_guide_chunks):
            order = (self._placement_stat_chunks + self._placement_guide_chunks) if (is_stat_q or not is_guide_q) else (self._placement_guide_chunks + self._placement_stat_chunks)
            order_ids = {c["chunk_id"] for c in order}
            ordered_items = [
                dict(c, dense_score=c.get("dense_score") or 0.85, fused_score=c.get("fused_score") or 0.85)
                for c in order
            ]
            other_passages = [p for p in passages if p["chunk_id"] not in order_ids]
            passages = ordered_items + other_passages
            if "academics" not in search.get("routed", []):
                search.setdefault("routed", []).append("academics")

        # Prioritize verified BBA records for BBA-specific queries
        is_bba_q = bool(re.search(r"\b(bba|b\.?b\.?a|bachelor of business administratio\w*|management programme)\b", q_comb))
        if is_bba_q and hasattr(self, "_bba_chunks") and self._bba_chunks:
            if any(k in q_comb for k in ("fee", "fees", "cost", "tuition")):
                p_cids = ("e08a507321-0000", "97de2bec53-0001", "97de2bec53-0002", "595e59e931-0000")
            elif any(k in q_comb for k in ("eligib", "criteria", "admission", "requirement", "qualif")):
                p_cids = ("68a5ed8c52-0001", "7a3fc7b2fe-0001", "b167065a58-0002")
            else:
                p_cids = ("595e59e931-0000", "595e59e931-0001", "e40c447c3b-0000", "e67cb3f1c4-0000", "cee6516655-0004")
            bba_top = [c for cid in p_cids if (c := self.kb.get_chunk(cid))]
            bba_top_ids = {c["chunk_id"] for c in bba_top}
            bba_items = [
                dict(c, dense_score=c.get("dense_score") or 0.88, fused_score=c.get("fused_score") or 0.88)
                for c in bba_top
            ]
            passages = bba_items + [p for p in passages if p["chunk_id"] not in bba_top_ids]
            search["results"] = passages
            if "academics" not in search.get("routed", []):
                search.setdefault("routed", []).append("academics")

        # Prioritize verified B.Des records for B.Des-specific queries
        is_bdes_q = bool(re.search(r"\b(bdes|b\.?des|bachelor of design|design programme)\b", q_comb))
        if is_bdes_q and hasattr(self, "_bdes_chunks") and self._bdes_chunks:
            if any(k in q_comb for k in ("fee", "fees", "cost", "tuition")):
                p_cids = ("e08a507321-0000", "97de2bec53-0001", "97de2bec53-0002")
            elif any(k in q_comb for k in ("eligib", "criteria", "admission", "requirement", "qualif")):
                p_cids = ("68a5ed8c52-0000", "7a3fc7b2fe-0000", "b167065a58-0002", "cc1be9fb7a-0000")
            else:
                p_cids = ("26d54c68ee-0000", "4c16ed4b45-0000", "367306ad64-0000", "bcd89dff70-0000", "29d2327c3a-0000")
            bdes_top = [c for cid in p_cids if (c := self.kb.get_chunk(cid))]
            bdes_top_ids = {c["chunk_id"] for c in bdes_top}
            bdes_items = [
                dict(c, dense_score=c.get("dense_score") or 0.88, fused_score=c.get("fused_score") or 0.88)
                for c in bdes_top
            ]
            passages = bdes_items + [p for p in passages if p["chunk_id"] not in bdes_top_ids]
            search["results"] = passages
            if "academics" not in search.get("routed", []):
                search.setdefault("routed", []).append("academics")

        # Prioritize verified B.Tech records for B.Tech-specific queries
        is_btech_q = bool(re.search(r"\b(btech|b\.?tech|cse|computer science|bachelor of technology|engineering programme)\b", q_comb))
        if is_btech_q and hasattr(self, "_btech_chunks") and self._btech_chunks:
            if any(k in q_comb for k in ("fee", "fees", "cost", "tuition")):
                p_cids = ("e08a507321-0000", "3a0732ce2f-0000", "97de2bec53-0001")
            elif any(k in q_comb for k in ("eligib", "criteria", "admission", "requirement", "qualif", "selection", "process")):
                p_cids = ("68a5ed8c52-0000", "7a3fc7b2fe-0000", "314bd551e8-0002")
            else:
                p_cids = ("e8f8896b6f-0001", "fc39067d7f-0000", "13688686c5-0006", "68a5ed8c52-0000")
            btech_top = [c for cid in p_cids if (c := self.kb.get_chunk(cid))]
            btech_top_ids = {c["chunk_id"] for c in btech_top}
            btech_items = [
                dict(c, dense_score=c.get("dense_score") or 0.88, fused_score=c.get("fused_score") or 0.88)
                for c in btech_top
            ]
            passages = btech_items + [p for p in passages if p["chunk_id"] not in btech_top_ids]
            search["results"] = passages
            if "academics" not in search.get("routed", []):
                search.setdefault("routed", []).append("academics")

        # Prioritize verified festival and event records for fest/event queries
        is_fest_q = any(
            w in (query + " " + extracted_q).lower()
            for w in ("fest", "fests", "festival", "festivals", "sabrang", "spardha", "hackjklu", "aarambh", "aarohan")
        ) or (
            any(w in (query + " " + extracted_q).lower() for w in ("event", "events"))
            and not any(w in (query + " " + extracted_q).lower() for w in ("calendar", "semester", "exam", "examination", "admissions"))
        )
        if is_fest_q and self._festival_chunks:
            fest_ids = {c["chunk_id"] for c in self._festival_chunks}
            fest_items = [
                dict(c, dense_score=c.get("dense_score") or 0.88, fused_score=c.get("fused_score") or 0.88)
                for c in self._festival_chunks
            ]
            passages = fest_items + [p for p in passages if p["chunk_id"] not in fest_ids]
            search["results"] = passages
            if "general" not in search.get("routed", []):
                search.setdefault("routed", []).append("general")

        # Evidence checking evaluated against extracted question (ignores conversational noise)
        ev = E.check(extracted_q, search, self.kb)
        has_year_mismatch = any(t.isdigit() and len(t) == 4 and t.startswith("20") for t in ev.missing_terms)
        if memory_match and not person_mismatch and memory_match.tier in ("exact", "strong") and ev.status == E.ESCALATE and not has_year_mismatch:
            ev.status = E.ANSWER
            ev.confidence = max(ev.confidence, memory_match.similarity)
            ev.reason = f"Verified via Self-Agentic RAG learned memory cluster [{memory_match.entry.id}] (sim={memory_match.similarity:.2f})"

        # Fallback search if primary retrieval escalated
        if ev.status == E.ESCALATE and not passages:
            fallback_outcome = "triggered"
            fb_search = self.kb.search(extracted_q, top_k=top_k, mode="sparse")
            fb_results = fb_search.get("results", [])
            if fb_results:
                fb_ev = E.check(extracted_q, fb_search, self.kb)
                if fb_ev.status in (E.ANSWER, E.CLARIFY):
                    search = fb_search
                    passages = fb_results
                    ev = fb_ev
                    fallback_outcome = "recovered"
                else:
                    fallback_outcome = "weak_evidence"
            else:
                fallback_outcome = "no_results"

        # Configurable Debug Logging
        if os.environ.get("JKLU_DEBUG_RETRIEVAL", "").lower() in ("1", "true", "yes"):
            log.info("[DEBUG] Original: '%s'", raw_question)
            log.info("[DEBUG] Extracted: '%s' | Sub-q: %s | Unrelated: %s",
                     extracted_q, analysis.sub_questions, analysis.unrelated_parts)
            log.info("[DEBUG] Queries searched: %s | is_fac_roster=%s | Retrieved: %d chunks",
                     queries_to_search, is_fac_roster, len(passages))
            for pi, p in enumerate(passages[:4], 1):
                log.info("[DEBUG]   Chunk %d: sim=%.3f, title='%s', url='%s'",
                         pi, p.get("similarity_score", 0.0), p.get("title", ""), p.get("source_url", ""))
            log.info("[DEBUG] Evidence: status=%s, conf=%.2f, reason='%s' | Fallback: %s",
                     ev.status, ev.confidence, ev.reason, fallback_outcome)

        routed = search.get("routed", [])
        meta = dict(reason=ev.reason, confidence=ev.confidence, category=(routed[0] if routed else None))

        if ev.status == E.CLARIFY:
            return done(status="clarify", answer=ev.clarification or E.DEFAULT_CLARIFY,
                        clarification=ev.clarification, suggestions=ev.suggestions,
                        sources=self._sources(passages[:3]) if ev.coverage else [], **meta)
        if ev.status == E.ESCALATE:
            lead = ("I couldn't find reliable information about this in the JKLU documents I have, so I'd "
                    "rather not guess. Please contact the university directly:")
            return done(status="escalate", answer=self._escalation_text(routed, lead),
                        contacts=self._contacts_for(routed), **meta)

        # Generate answer or use verified memory answer if present
        if memory_match and not person_mismatch and memory_match.tier in ("exact", "strong") and memory_match.entry.verified_answer and not analysis.unrelated_parts:
            answer = memory_match.entry.verified_answer
            provider = "self_rag_memory"
            err = None
        else:
            answer, provider, err = self.generator.generate(extracted_q, passages, history)

        # Enforce zero code generation for multi-context queries containing unrelated requests
        if analysis.unrelated_parts and ("```" in answer or re.search(r"\bdef\s+\w+\s*\(", answer)):
            # Strip out leaked code blocks or code functions
            answer = re.sub(r"```(?:[a-zA-Z0-9_\-]+)?\n[\s\S]*?```", "", answer).strip()
            answer = re.sub(r"(?:^|\n)(?:def|function|class)\s+[a-zA-Z0-9_]+\s*\([^)]*\):?[\s\S]*?(?=\n\n|\Z)", "", answer).strip()

        # If user asked an unrelated secondary question, append a polite clarification
        if analysis.unrelated_parts and INSUFFICIENT not in answer:
            unrelated_str = ", ".join(f'"{u}"' for u in analysis.unrelated_parts)
            note = f"\n\n*(Note: I answer questions strictly based on verified JKLU documents and context. Questions regarding external topics ({unrelated_str}) are outside the university database and are not answered.)*"
            answer += note

        if INSUFFICIENT in answer:
            lead = ("The retrieved JKLU documents don't contain a clear answer to this, so I'd rather not guess. "
                    "Please contact the university directly:")
            return done(status="escalate", answer=self._escalation_text(routed, lead), provider=provider,
                        contacts=self._contacts_for(routed),
                        **{**meta, "reason": "LLM judged the evidence insufficient"})
        used, numbers = self._cited(answer, passages, intent=T.classify_query_intent(extracted_q))
        return done(status="answer", answer=answer, sources=self._sources(used, numbers),
                    provider=provider, **meta)

    @staticmethod
    def _cited(answer: str, passages: list[dict], intent: str | None = None) -> tuple[list[dict], list[int]]:
        """Keep sources actually cited as [n]; if the answer has no citations and is academic concept, don't force unrelated sources."""
        nums = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
        pairs = [(i, p) for i, p in enumerate(passages, 1) if i in nums]
        if not pairs:
            if intent == T.INTENT_ACADEMIC:
                return [], []
            pairs = list(enumerate(passages, 1))
        return [p for _, p in pairs], [i for i, _ in pairs]

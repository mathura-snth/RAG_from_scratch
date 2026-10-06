"""
cove.py — Factored Chain-of-Verification (CoVe) for DentaRAG
=============================================================
Implements the Factored CoVe pipeline (Dhuliawala et al., 2023)
adapted for a dental RAG system.

Pipeline (4 steps):
    1. Draft      : generate an initial answer from RAG context
    2. Plan       : generate N independent verification questions
                    targeting specific factual claims in the draft
    3. Execute    : answer each verification question independently
                    (without seeing the draft — "factored" variant)
    4. Revise     : rewrite the draft using the verification answers

Reference:
    Dhuliawala et al. (2023). Chain-of-Verification Reduces Hallucination
    in Large Language Models. arXiv:2309.11495
"""

from __future__ import annotations
from typing import Any
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DRAFT_PROMPT = """\
You are a precise dental clinical assistant.
Use the following context extracted from dental research documents to answer
the question. Be factual and concise.

Context:
{context}

Question: {question}

Draft answer:"""


PLAN_PROMPT = """\
You are a medical fact-checker. Read the draft answer below and identify
the specific factual claims it makes that could be verified independently.

Generate exactly {n_verif} short, self-contained verification questions
(one per line, no numbering) that would confirm or refute the key claims
in the draft. Each question must be answerable from the context alone.

Draft answer:
{draft}

Verification questions (one per line):"""


EXECUTE_PROMPT = """\
You are a dental expert. Answer the following question as precisely as
possible using the context below. Do not refer to any previous answer —
answer independently.

Context:
{context}

Question: {verification_question}

Answer:"""


REVISE_PROMPT = """\
You are a precise dental clinical assistant.
You wrote an initial draft answer to a question, then verified specific
claims using the context. Now revise the draft to correct any errors
found during verification and produce a final, accurate answer.

Original question: {question}

Initial draft:
{draft}

Verification results:
{verification_summary}

Final revised answer:"""


# ---------------------------------------------------------------------------
# CoVe engine
# ---------------------------------------------------------------------------

class CoVeEngine:
    """
    Factored Chain-of-Verification engine.

    Parameters
    ----------
    llm : ChatGroq
        The LLM used for all four steps.
    n_verification_questions : int
        Number of verification questions generated at the Plan step.
    """

    def __init__(self, llm: ChatGroq, n_verification_questions: int = 3):
        self.llm = llm
        self.n_verif = n_verification_questions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, question: str, contexts: list[str]) -> dict[str, Any]:
        """
        Run the full Factored CoVe pipeline.

        Parameters
        ----------
        question : str
            The user's dental question.
        contexts : list[str]
            Retrieved document chunks from the RAG retriever.

        Returns
        -------
        dict with keys:
            draft               : str   — initial answer (step 1)
            verification_questions : list[str] — questions generated (step 2)
            verification_answers   : list[str] — independent answers (step 3)
            final_answer        : str   — revised answer (step 4)
        """
        context_text = "\n\n".join(contexts) if contexts else ""

        # Step 1 — Draft
        draft = self._draft(question, context_text)

        # Step 2 — Plan
        verif_questions = self._plan(draft)

        # Step 3 — Execute (factored: each question answered independently)
        verif_answers = self._execute(verif_questions, context_text)

        # Step 4 — Revise
        final_answer = self._revise(question, draft, verif_questions, verif_answers)

        return {
            "draft": draft,
            "verification_questions": verif_questions,
            "verification_answers": verif_answers,
            "final_answer": final_answer,
        }

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _draft(self, question: str, context_text: str) -> str:
        """Step 1 — generate the initial draft answer."""
        prompt = DRAFT_PROMPT.format(context=context_text, question=question)
        response = self.llm.invoke(prompt)
        draft = response.content.strip()
        print(f"[CoVe] Step 1 — Draft generated ({len(draft)} chars)")
        return draft

    def _plan(self, draft: str) -> list[str]:
        """Step 2 — generate verification questions from the draft."""
        prompt = PLAN_PROMPT.format(draft=draft, n_verif=self.n_verif)
        response = self.llm.invoke(prompt)
        raw = response.content.strip()

        # Parse: one question per non-empty line
        questions = [
            line.strip().lstrip("-•* ")
            for line in raw.splitlines()
            if line.strip()
        ][: self.n_verif]  # guard against the model returning too many

        print(f"[CoVe] Step 2 — {len(questions)} verification questions planned")
        for i, q in enumerate(questions, 1):
            print(f"         Q{i}: {q}")
        return questions

    def _execute(self, verif_questions: list[str], context_text: str) -> list[str]:
        """
        Step 3 — answer each verification question independently.
        This is the 'Factored' variant: the draft is NOT shown here,
        preventing the model from simply confirming its own claims.
        """
        answers = []
        for i, vq in enumerate(verif_questions, 1):
            prompt = EXECUTE_PROMPT.format(
                context=context_text,
                verification_question=vq,
            )
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            answers.append(answer)
            print(f"[CoVe] Step 3 — Verified Q{i}: {answer[:80]}...")
        return answers

    def _revise(
        self,
        question: str,
        draft: str,
        verif_questions: list[str],
        verif_answers: list[str],
    ) -> str:
        """Step 4 — revise the draft using verification results."""
        # Build a readable verification summary
        summary_lines = []
        for q, a in zip(verif_questions, verif_answers):
            summary_lines.append(f"Q: {q}\nA: {a}")
        verification_summary = "\n\n".join(summary_lines)

        prompt = REVISE_PROMPT.format(
            question=question,
            draft=draft,
            verification_summary=verification_summary,
        )
        response = self.llm.invoke(prompt)
        final = response.content.strip()
        print(f"[CoVe] Step 4 — Final answer revised ({len(final)} chars)")
        return final

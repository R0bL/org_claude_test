"""
LLM Judge for Entity Resolution Validation (Minimal)
"""

import json
from typing import Dict, Any


def parse_llm_json_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response from LLM."""
    text = response_text.strip()

    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```"))
        text = text.strip()

    if not text.startswith("{"):
        start = text.find("{")
        if start >= 0:
            text = text[start:]

    try:
        result = json.loads(text)
        is_match = bool(result.get("match"))
        return {
            "match": is_match,
            "confidence": float(result.get("confidence", 0.9 if is_match else 0.1)),
            "reason": str(result.get("reason", ""))[:500]
        }
    except json.JSONDecodeError:
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ["match: true", "yes", "same"]):
            return {"match": True, "confidence": 0.7, "reason": text[:150]}
        elif any(phrase in text_lower for phrase in ["match: false", "no", "different"]):
            return {"match": False, "confidence": 0.7, "reason": text[:150]}
        else:
            return {"match": None, "confidence": 0.0, "reason": f"parse_error: {text[:100]}"}


def judge_match_with_evidence(
    row: Dict,
    dim_org_record: Dict,
    evidence: Dict,
    client,
    model: str = "claude-sonnet-4-20250514"
) -> Dict[str, Any]:
    """LLM validation with FDA label + web search evidence."""
    evidence_sections = []

    if evidence.get("fda_labeler"):
        evidence_sections.append(f"""FDA DRUG LABEL (Official):
  Labeler: {evidence['fda_labeler']}
  Drug: {evidence.get('fda_drug', 'N/A')}""")

    if evidence.get("company_context"):
        snippets = [f"  - {ctx.get('snippet', '')[:150]}" for ctx in evidence["company_context"]]
        evidence_sections.append("COMPANY INFO (Web Search):\n" + "\n".join(snippets))

    if evidence.get("relationship_search"):
        snippets = [f"  - {rel.get('snippet', '')[:150]}" for rel in evidence["relationship_search"]]
        evidence_sections.append("RELATIONSHIP EVIDENCE:\n" + "\n".join(snippets))

    evidence_text = "\n\n".join(evidence_sections) if evidence_sections else "No additional evidence found."
    fda_labeler = evidence.get('fda_labeler', 'Unknown')

    prompt = f"""You are an expert at pharmaceutical company entity resolution.

TASK: Determine if the SOURCE ORGANIZATION and CANDIDATE MATCH refer to the same company.

SOURCE ORGANIZATION (from OpenFDA drug label):
  Name in database: {row.get('name_r', 'N/A')}
  FDA Official Labeler: {fda_labeler}

CANDIDATE MATCH (from reference database):
  Name: {dim_org_record.get('name', 'N/A')}
  Country: {dim_org_record.get('country_code', 'N/A')}

EVIDENCE:
{evidence_text}

ANALYSIS STEPS:
1. Compare the FDA Official Labeler name to the Candidate Match name
2. Consider if they could be:
   - Same company with different legal suffix (Inc vs Ltd)
   - Parent/subsidiary relationship (match if same corporate family)
   - Regional variations (e.g., "Pfizer" vs "Pfizer (Germany)" = MATCH)
   - Completely different companies (e.g., different industries) = NO MATCH

3. Use the web search evidence to verify relationships

RESPOND WITH JSON ONLY:
{{"match": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return parse_llm_json_response(response.content[0].text)
    except Exception as e:
        return {"match": None, "confidence": 0.0, "reason": f"error: {str(e)[:100]}"}


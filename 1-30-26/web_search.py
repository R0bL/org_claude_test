"""
Web Search and FDA Label Lookup (Minimal)
"""

import requests
import os
import re
from typing import Optional, Dict, List


def get_labeler_from_dailymed(ndc_code: str) -> Optional[Dict]:
    """Get labeler info from DailyMed using NDC search."""
    if not ndc_code:
        return None
    ndc_clean = ndc_code.split('_')[0] if '_' in ndc_code else ndc_code
    try:
        url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?ndc={ndc_clean}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', [])
            if results:
                spl = results[0]
                title = spl.get('title', '')
                match = re.search(r'\[([^\]]+)\]', title)
                labeler = match.group(1) if match else None
                drug_name = title.split('[')[0].strip() if '[' in title else title
                return {
                    "setid": spl.get('setid'),
                    "title": title,
                    "labeler": labeler,
                    "drug_name": drug_name,
                    "ndc": ndc_clean,
                    "source": "dailymed_ndc_search"
                }
    except Exception:
        pass
    return None


def serper_search(query: str, num_results: int = 5) -> List[Dict]:
    """Execute Google search via Serper API."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return []
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        return response.json().get("organic", [])
    except Exception:
        return []


def search_company_context(company_name: str) -> List[Dict]:
    """Search for context about a pharmaceutical company."""
    if not company_name:
        return []
    return serper_search(
        f'"{company_name}" pharmaceutical company headquarters subsidiaries products',
        num_results=3
    )


def search_company_relationship(org1: str, org2: str) -> List[Dict]:
    """Search for relationship between two organizations."""
    if not org1 or not org2:
        return []
    return serper_search(
        f'"{org1}" "{org2}" same company subsidiary parent',
        num_results=2
    )


def get_match_evidence(
    source_org: str,
    matched_org: str,
    ndc_code: Optional[str] = None,
    spl_id: Optional[str] = None,
    drug_name: Optional[str] = None
) -> Dict:
    """Gather comprehensive evidence for match validation."""
    evidence = {
        "fda_labeler": None,
        "fda_drug": None,
        "drug_label_source": None,
        "company_context": None,
        "relationship_search": None
    }

    if ndc_code:
        dailymed_result = get_labeler_from_dailymed(ndc_code)
        if dailymed_result:
            evidence["fda_labeler"] = dailymed_result.get("labeler")
            evidence["fda_drug"] = dailymed_result.get("drug_name")
            evidence["drug_label_source"] = "dailymed_ndc"
            evidence["dailymed_full"] = dailymed_result

    labeler_to_search = evidence.get("fda_labeler") or source_org
    if labeler_to_search:
        context_results = search_company_context(labeler_to_search)
        if context_results:
            evidence["company_context"] = [
                {"title": r.get("title", ""), "snippet": r.get("snippet", "")[:200]}
                for r in context_results[:3]
            ]

    relationship_results = search_company_relationship(source_org, matched_org)
    if relationship_results:
        evidence["relationship_search"] = [
            {"title": r.get("title", ""), "snippet": r.get("snippet", "")[:200]}
            for r in relationship_results[:2]
        ]

    return evidence


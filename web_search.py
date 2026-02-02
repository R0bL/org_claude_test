"""
Web Search and FDA Label Lookup Module
=======================================

Provides evidence gathering for LLM validation:
1. DailyMed API - Search by NDC to get FDA labeler info
2. Serper web search - Company context and relationship verification
"""

import requests
import os
import re
from typing import Optional, Dict, List
from utils import log_step


# =============================================================================
# DAILYMED API - FDA Drug Label Lookup by NDC
# =============================================================================

def get_labeler_from_dailymed(ndc_code: str) -> Optional[Dict]:
    """
    Get labeler info from DailyMed using NDC search.
    
    Uses the /spls.json?ndc= endpoint which returns title containing labeler.
    Title format: "DRUG NAME [LABELER NAME]"
    
    DailyMed API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
    """
    if not ndc_code:
        return None
    
    # Clean NDC (take first part before underscore if composite ID)
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
                
                # Extract labeler from title format: "DRUG NAME [LABELER NAME]"
                match = re.search(r'\[([^\]]+)\]', title)
                labeler = match.group(1) if match else None
                
                # Extract drug name (everything before the bracket)
                drug_name = title.split('[')[0].strip() if '[' in title else title
                
                return {
                    "setid": spl.get('setid'),
                    "title": title,
                    "labeler": labeler,
                    "drug_name": drug_name,
                    "ndc": ndc_clean,
                    "source": "dailymed_ndc_search"
                }
    except Exception as e:
        # Silently fail - will use Serper fallback
        pass
    
    return None


# =============================================================================
# SERPER WEB SEARCH - Company Context and Verification
# =============================================================================

def serper_search(query: str, num_results: int = 5) -> List[Dict]:
    """Execute Google search via Serper API"""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": num_results}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        return response.json().get("organic", [])
    except Exception as e:
        return []


def search_company_context(company_name: str) -> List[Dict]:
    """Search for context about a pharmaceutical company"""
    if not company_name:
        return []
    return serper_search(
        f'"{company_name}" pharmaceutical company headquarters subsidiaries products',
        num_results=3
    )


def search_company_relationship(org1: str, org2: str) -> List[Dict]:
    """Search for relationship between two organizations"""
    if not org1 or not org2:
        return []
    return serper_search(
        f'"{org1}" "{org2}" same company subsidiary parent',
        num_results=2
    )


# =============================================================================
# RICH EVIDENCE GATHERING
# =============================================================================

def get_match_evidence(
    source_org: str,
    matched_org: str,
    ndc_code: Optional[str] = None,
    spl_id: Optional[str] = None,
    drug_name: Optional[str] = None
) -> Dict:
    """
    Gather comprehensive evidence for match validation:
    
    1. DailyMed lookup - Get official FDA labeler from NDC
    2. Company context - Search for info about the FDA labeler
    3. Relationship search - Look for evidence connecting source and matched orgs
    
    Returns structured evidence dict for LLM prompt.
    """
    evidence = {
        "fda_labeler": None,
        "fda_drug": None,
        "drug_label_source": None,
        "company_context": None,
        "relationship_search": None
    }
    
    # Step 1: Get FDA labeler from DailyMed
    if ndc_code:
        dailymed_result = get_labeler_from_dailymed(ndc_code)
        if dailymed_result:
            evidence["fda_labeler"] = dailymed_result.get("labeler")
            evidence["fda_drug"] = dailymed_result.get("drug_name")
            evidence["drug_label_source"] = "dailymed_ndc"
            evidence["dailymed_full"] = dailymed_result
    
    # Step 2: Search for company context about the FDA labeler
    labeler_to_search = evidence.get("fda_labeler") or source_org
    if labeler_to_search:
        context_results = search_company_context(labeler_to_search)
        if context_results:
            # Extract snippets for LLM
            evidence["company_context"] = [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", "")[:200]
                }
                for r in context_results[:3]
            ]
    
    # Step 3: Search for relationship between orgs
    relationship_results = search_company_relationship(source_org, matched_org)
    if relationship_results:
        evidence["relationship_search"] = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", "")[:200]
            }
            for r in relationship_results[:2]
        ]
    
    return evidence


def format_evidence_for_prompt(evidence: Dict) -> str:
    """Format evidence dict into readable text for LLM prompt"""
    sections = []
    
    # FDA Labeler info
    if evidence.get("fda_labeler"):
        sections.append(f"""FDA DRUG LABEL (from DailyMed):
  Official Labeler: {evidence['fda_labeler']}
  Drug Product: {evidence.get('fda_drug', 'N/A')}""")
    
    # Company context
    if evidence.get("company_context"):
        snippets = []
        for ctx in evidence["company_context"]:
            snippets.append(f"  - {ctx['snippet'][:150]}")
        sections.append("COMPANY INFORMATION (from web search):\n" + "\n".join(snippets))
    
    # Relationship search
    if evidence.get("relationship_search"):
        snippets = []
        for rel in evidence["relationship_search"]:
            snippets.append(f"  - {rel['snippet'][:150]}")
        sections.append("RELATIONSHIP EVIDENCE:\n" + "\n".join(snippets))
    
    if not sections:
        return "No additional evidence found."
    
    return "\n\n".join(sections)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing DailyMed NDC Search...")
    
    # Test with known NDC
    test_ndc = "29300-129"
    result = get_labeler_from_dailymed(test_ndc)
    if result:
        print(f"  NDC: {test_ndc}")
        print(f"  Labeler: {result.get('labeler')}")
        print(f"  Drug: {result.get('drug_name')}")
    else:
        print(f"  NDC {test_ndc}: No result")
    
    # Test full evidence gathering
    print("\nTesting get_match_evidence...")
    evidence = get_match_evidence(
        source_org="Unichem Pharmaceuticals",
        matched_org="Unichem Laboratories (India)",
        ndc_code="29300-129",
        spl_id=None,
        drug_name=None
    )
    print(f"  FDA Labeler: {evidence.get('fda_labeler')}")
    print(f"  Drug: {evidence.get('fda_drug')}")
    print(f"  Source: {evidence.get('drug_label_source')}")
    
    if evidence.get('company_context'):
        print(f"  Company context: {len(evidence['company_context'])} results")
    
    print("\nFormatted for LLM:")
    print(format_evidence_for_prompt(evidence))

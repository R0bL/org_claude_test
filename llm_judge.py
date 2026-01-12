"""
LLM Judge for Entity Resolution Validation

Source-aware post-Splink validation using Claude to review matches.
"""

import json
from typing import Dict, Optional, Any
from anthropic import Anthropic

from data_prep import haversine_distance


# Source-specific context for prompt construction
SOURCE_CONTEXT = {
    "who_clinical_trials_silver.studies_metadata": {
        "description": "WHO clinical trial registry. Contains global trial metadata.",
        "key_fields": ["contact_address", "contact_email", "contact_affiliation"],
        "notes": "May have contact_address and contact_email for additional context."
    },
    "nih_clinical_trials_gov_silver.cl_trial_locations": {
        "description": "Clinical trial site/facility location from ClinicalTrials.gov.",
        "key_fields": ["city", "state", "zip", "latitude", "longitude"],
        "notes": "Match to institutional location. Geographic coordinates available."
    },
    "nih_clinical_trials_gov_silver.cl_trial_sponsors": {
        "description": "Trial sponsor from ClinicalTrials.gov.",
        "key_fields": ["class"],
        "notes": "'class' indicates NIH/INDUSTRY/OTHER. Short names may be individuals."
    },
    "nih_clinical_trials_gov_silver.cl_trial_collaborators": {
        "description": "Trial collaborating organization from ClinicalTrials.gov.",
        "key_fields": ["class"],
        "notes": "Similar to sponsors. May be research institutions or companies."
    },
    "nih_clinical_trials_gov_silver.cl_trial_organizations": {
        "description": "Organization linked to clinical trial from ClinicalTrials.gov.",
        "key_fields": ["city", "state", "country"],
        "notes": "General organization record. May be sponsor, site, or collaborator."
    },
    "open_fda_silver.ndc_drugs": {
        "description": "FDA drug labeler/manufacturer from NDC directory.",
        "key_fields": ["drug_generic_name", "drug_brand_name", "labeler_name"],
        "notes": "Same company may have many drug products - that's normal. Match to parent company."
    },
    "legacy_alpha_silver._organizations_consolidated": {
        "description": "Legacy organization record with Chinese aliases and geo_locations.",
        "key_fields": ["name_aliases", "geo_locations", "website_url"],
        "notes": "Check alias list carefully. May have Chinese/English names. Has website URLs."
    },
    "chinese_clinical_trials_silver.trials": {
        "description": "Clinical trial from Chinese Clinical Trial Registry (ChiCTR).",
        "key_fields": ["sponsor", "institution"],
        "notes": "Name may be romanized Chinese. Limited metadata available."
    },
    "chinese_clinical_trials_silver.trial_contacts": {
        "description": "Contact organization from Chinese trial registry.",
        "key_fields": ["affiliation", "email"],
        "notes": "Affiliation may be in Chinese or romanized. Email domain may help."
    },
    "uspto_silver.patents": {
        "description": "Patent assignee from USPTO.",
        "key_fields": ["publication_number", "application_number", "assignee_type"],
        "notes": "May be subsidiary vs parent company. Patent assignees can change over time."
    },
}


def get_source_context(source_table: str) -> Dict[str, Any]:
    """Get context dict for a source table, with fallback for unknown sources."""
    # Try exact match first
    if source_table in SOURCE_CONTEXT:
        return SOURCE_CONTEXT[source_table]
    
    # Try partial match (for sources with wildcards in the lookup)
    for key, value in SOURCE_CONTEXT.items():
        if key.endswith(".*"):
            prefix = key[:-2]
            if source_table.startswith(prefix):
                return value
    
    # Fallback for unknown sources
    return {
        "description": "Unknown source table.",
        "key_fields": [],
        "notes": "No specific context available for this source."
    }


def extract_metadata_fields(metadata: Dict) -> Dict[str, Any]:
    """Extract useful metadata fields for prompt context."""
    fields = {}
    
    # Contact info
    for key in ['contact_address', 'contact_email', 'contact_affiliation', 'email']:
        if metadata.get(key):
            fields[key] = metadata[key]
    
    # Drug info
    for key in ['drug_generic_name', 'drug_brand_name', 'labeler_name']:
        if metadata.get(key):
            fields[key] = metadata[key]
    
    # Aliases and names
    if metadata.get('name_aliases'):
        aliases = metadata['name_aliases']
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(',') if a.strip()]
        fields['aliases'] = aliases[:5]  # Limit to 5 aliases
    
    # Geographic
    for key in ['geo_locations', 'website_url']:
        if metadata.get(key):
            fields[key] = metadata[key]
    
    # Patent info
    for key in ['publication_number', 'application_number', 'assignee_type']:
        if metadata.get(key):
            fields[key] = metadata[key]
    
    # Trial info
    for key in ['class', 'sponsor', 'institution', 'affiliation']:
        if metadata.get(key):
            fields[key] = metadata[key]
    
    return fields


def build_prompt(
    row: Dict,
    dim_org_record: Dict,
    metadata: Dict,
    geo_distance_km: Optional[float] = None
) -> str:
    """
    Build validation prompt with chain-of-thought reasoning.
    
    Research-based design:
    - Chain-of-thought prompting improves accuracy by 15-36%
    - Contrastive examples reduce false positives
    - Multi-step reasoning outperforms single-step
    """
    source_name = row.get('name_r', 'N/A')
    dim_name = dim_org_record.get('name', 'N/A')
    dim_country = dim_org_record.get('country_code', '')
    dim_city = dim_org_record.get('city', '')
    source_table = row.get('source_table', 'unknown')
    
    # Get aliases safely (fixes numpy array bug)
    aliases = dim_org_record.get('all_names', [])
    if aliases is None:
        aliases = []
    elif isinstance(aliases, str):
        aliases = [aliases]
    elif hasattr(aliases, 'tolist'):
        aliases = aliases.tolist()
    aliases_str = ", ".join(str(a) for a in list(aliases)[:3]) if aliases else "None"

    return f"""You are an expert at organization entity resolution. Determine if these two records refer to the SAME real-world organization.

## RECORD A (from {source_table}):
Name: "{source_name}"

## RECORD B (reference database):
Name: "{dim_name}"
Aliases: {aliases_str}
Location: {dim_city}, {dim_country}

## STEP-BY-STEP ANALYSIS:

1. **Core Identity Check**: Do both names refer to the same core organization?
   - Ignore legal suffixes (Inc, LLC, Ltd, GmbH, AG)
   - Ignore location qualifiers in parentheses like "(United States)"
   - Consider abbreviations (MIT = Massachusetts Institute of Technology)

2. **Subsidiary vs Parent Check**: Are these DIFFERENT legal entities?
   - "Pfizer Inc" and "Pfizer Canada" = DIFFERENT (regional subsidiary)
   - "AbbVie" and "AbbVie Inc." = SAME (just legal suffix difference)
   
3. **Department Check**: Is one a department/division of the other?
   - "Johns Hopkins Cardiology Dept" and "Johns Hopkins University" = SAME
   - Departments belong to their parent organization

4. **Person Name Check**: Is Record A a person's name, not an organization?
   - 2-3 word names like "John Smith" or "Dr. Maria Garcia" = NOT an organization
   - If Record A is a person, answer NO MATCH

## EXAMPLES:

Record A: "Novo Nordisk (Japan)" | Record B: "Novo Nordisk" -> YES (same company, location qualifier)
Record A: "AbbVie Inc." | Record B: "AbbVie" -> YES (same company, legal suffix)
Record A: "MIT" | Record B: "Massachusetts Institute of Technology" -> YES (abbreviation)
Record A: "Stanford Medicine" | Record B: "Stanford University" -> YES (department)
Record A: "Novartis Canada" | Record B: "Novartis AG" -> NO (different subsidiary)
Record A: "Dr. Philip Chen" | Record B: "Chen Medical Group" -> NO (person vs org)
Record A: "Apple" | Record B: "Apple Inc." -> YES (same company)
Record A: "Apple Records" | Record B: "Apple Inc." -> NO (different companies)

## YOUR TASK:

Analyze the records above and respond with ONLY this JSON:
{{"match": true, "reason": "brief explanation"}} or {{"match": false, "reason": "brief explanation"}}"""


def judge_match(
    row: Dict,
    dim_org_record: Dict,
    metadata: Dict,
    client: Anthropic,
    model: str = "claude-sonnet-4-20250514"
) -> Dict[str, Any]:
    """
    Use LLM to judge if a match is correct.
    
    Args:
        row: Prediction row with match details
        dim_org_record: dim_org record being matched to
        metadata: Parsed metadata from source record
        client: Anthropic client
        model: Model to use (default: claude-sonnet-4-20250514)
        
    Returns:
        Dict with match, confidence, reason keys
    """
    # Calculate geographic distance if coordinates available
    geo_distance = None
    try:
        lat1 = dim_org_record.get('latitude')
        lon1 = dim_org_record.get('longitude')
        lat2 = row.get('latitude_r') or metadata.get('latitude')
        lon2 = row.get('longitude_r') or metadata.get('longitude')
        geo_distance = haversine_distance(lat1, lon1, lat2, lon2)
    except (TypeError, ValueError):
        pass
    
    prompt = build_prompt(row, dim_org_record, metadata, geo_distance)
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Try to parse JSON response
        # Handle cases where model wraps JSON in markdown
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines 
                if not line.startswith("```")
            )
        
        result = json.loads(response_text)
        
        # Normalize response (confidence defaults to 0.9 for matches, 0.1 for non-matches)
        is_match = bool(result.get("match"))
        return {
            "llm_match": is_match,
            "llm_confidence": float(result.get("confidence", 0.9 if is_match else 0.1)),
            "llm_reason": str(result.get("reason", ""))[:200]
        }
        
    except json.JSONDecodeError:
        return {
            "llm_match": None,
            "llm_confidence": 0,
            "llm_reason": f"parse_error: {response_text[:100]}"
        }
    except Exception as e:
        return {
            "llm_match": None,
            "llm_confidence": 0,
            "llm_reason": f"error: {str(e)[:100]}"
        }


def batch_judge_matches(
    predictions_df,
    dim_org_df,
    client: Anthropic,
    sample_size: Optional[int] = None,
    model: str = "claude-sonnet-4-20250514",
    progress_callback=None,
    max_workers: int = 20
) -> list:
    """
    Batch process predictions through LLM judge with parallel processing.
    
    Args:
        predictions_df: DataFrame of predictions to validate
        dim_org_df: dim_org DataFrame for lookup
        client: Anthropic client
        sample_size: Optional limit on records to process (None = all)
        model: Model to use
        progress_callback: Optional callback(idx, total) for progress
        max_workers: Number of parallel API calls (default 20)
        
    Returns:
        List of result dicts with prediction_idx and LLM judgment
    """
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Create dim_org lookup
    dim_org_lookup = dim_org_df.set_index('unique_id').to_dict('index')
    
    # Sample if requested
    if sample_size is not None and sample_size < len(predictions_df):
        sample = predictions_df.sample(sample_size, random_state=42)
    else:
        sample = predictions_df
    
    total = len(sample)
    
    def process_single(idx, row):
        """Process a single prediction with retry logic."""
        # Get dim_org record
        dim_record = dim_org_lookup.get(row.get('unique_id_l'), {})
        
        # Parse metadata
        raw_meta = row.get('raw_metadata', '{}')
        try:
            if isinstance(raw_meta, str):
                metadata = json.loads(raw_meta) if raw_meta else {}
            else:
                metadata = raw_meta or {}
        except json.JSONDecodeError:
            # Try parsing as Java map format
            from utils import parse_java_map
            metadata = parse_java_map(raw_meta) if raw_meta else {}
        
        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = judge_match(row.to_dict(), dim_record, metadata, client, model)
                result['prediction_idx'] = idx
                return result
            except Exception as e:
                if "rate" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return {
                    'llm_match': None,
                    'llm_confidence': 0,
                    'llm_reason': f'error after {max_retries} retries: {str(e)[:50]}',
                    'prediction_idx': idx
                }
    
    results = []
    completed = 0
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single, idx, row.to_dict()): idx 
            for idx, row in sample.iterrows()
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                results.append({
                    'llm_match': None,
                    'llm_confidence': 0,
                    'llm_reason': f'future error: {str(e)[:50]}',
                    'prediction_idx': idx
                })
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    
    return results


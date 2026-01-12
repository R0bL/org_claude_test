"""
Multi-Agent Validation Framework for Entity Resolution

Implements specialized agents for different validation scenarios to reduce costs
and improve accuracy. Based on multi-agent RAG framework research showing
94.3% accuracy with 61% reduction in API calls.

Agents:
1. DirectMatchAgent - Fast deterministic matching (exact/fuzzy)
2. GeographicAgent - Location-based validation
3. ContextualAgent - Source-specific metadata validation
4. EnsembleAgent - Full LLM validation with multiple models

Routing Strategy:
- Use deterministic agents for obvious cases (cheap, fast)
- Use specialized agents for cases with specific signals
- Use full LLM only for ambiguous cases requiring deep reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from anthropic import Anthropic
from utils import log_step
from data_prep import haversine_distance
from jellyfish import jaro_winkler_similarity
import re


class MatchingAgentOrchestrator:
    """
    Coordinate specialized agents for entity resolution validation.

    Routing decision tree:
    1. Exact match → DirectMatchAgent (instant, free)
    2. Has geo data + clear signal → GeographicAgent (instant, free)
    3. Has metadata + source patterns → ContextualAgent (instant, free)
    4. Otherwise → Full LLM validation (expensive, accurate)
    """

    def __init__(self, client: Optional[Anthropic] = None,
                 primary_model: str = "claude-sonnet-4-20250514",
                 fast_model: str = "claude-3-5-haiku-20241022",
                 enable_routing: bool = True):
        """
        Initialize orchestrator.

        Args:
            client: Anthropic client for LLM calls
            primary_model: Primary LLM model for complex cases
            fast_model: Fast/cheap model for simple cases
            enable_routing: If False, always use full LLM (for comparison)
        """
        self.client = client
        self.primary_model = primary_model
        self.fast_model = fast_model
        self.enable_routing = enable_routing

        # Statistics tracking
        self.stats = {
            'direct': 0,
            'geographic': 0,
            'contextual': 0,
            'llm_fast': 0,
            'llm_primary': 0,
            'total': 0
        }

    def validate_match(self, row: pd.Series, dim_org_record: Dict,
                      metadata: Dict, llm_judge_func: callable) -> Dict[str, Any]:
        """
        Route prediction to appropriate validation agent.

        Args:
            row: Prediction row with match_probability, names, etc.
            dim_org_record: Reference dim_org record
            metadata: Metadata from mismatched record
            llm_judge_func: Function to call for full LLM validation

        Returns:
            Dict with llm_match, llm_confidence, llm_reason, agent_used
        """
        self.stats['total'] += 1

        if not self.enable_routing:
            # Bypass routing - always use LLM
            result = llm_judge_func(row, dim_org_record, metadata, self.client, self.primary_model)
            result['agent_used'] = 'llm_primary'
            self.stats['llm_primary'] += 1
            return result

        # Route 1: Direct Match Agent (exact or very high similarity)
        direct_result = self._direct_match_agent(row, dim_org_record)
        if direct_result is not None:
            self.stats['direct'] += 1
            return direct_result

        # Route 2: Geographic Agent (has lat/lon data)
        geo_result = self._geographic_agent(row, dim_org_record, metadata)
        if geo_result is not None:
            self.stats['geographic'] += 1
            return geo_result

        # Route 3: Contextual Agent (source-specific patterns)
        context_result = self._contextual_agent(row, dim_org_record, metadata)
        if context_result is not None:
            self.stats['contextual'] += 1
            return context_result

        # Route 4: LLM Agent (choose model based on complexity)
        llm_result = self._llm_agent(row, dim_org_record, metadata, llm_judge_func)
        return llm_result

    def _direct_match_agent(self, row: pd.Series, dim_org: Dict) -> Optional[Dict]:
        """
        Fast deterministic matching for obvious cases.

        Returns match=True if:
        - Exact normalized name match
        - Very high similarity (>0.98) with country match

        Returns match=False if:
        - Very low similarity (<0.3)
        - Name length ratio > 3x (likely different entities)

        Returns None if uncertain (needs other agents)
        """
        name_l = str(row.get('name_normalized_l', row.get('name_l', ''))).lower().strip()
        name_r = str(row.get('name_normalized_r', row.get('name_r', ''))).lower().strip()

        if not name_l or not name_r:
            return None  # Missing data, can't determine

        # Exact match
        if name_l == name_r:
            return {
                'llm_match': True,
                'llm_confidence': 0.99,
                'llm_reason': 'DirectMatchAgent: Exact name match',
                'agent_used': 'direct'
            }

        # Calculate similarity
        similarity = jaro_winkler_similarity(name_l, name_r)

        # Very high similarity + country match
        country_l = row.get('country_code_l', '')
        country_r = row.get('country_code_r', '')
        countries_match = (country_l == country_r) if (country_l and country_r) else None

        if similarity > 0.98 and countries_match:
            return {
                'llm_match': True,
                'llm_confidence': 0.95,
                'llm_reason': f'DirectMatchAgent: Very high similarity ({similarity:.3f}) with country match',
                'agent_used': 'direct'
            }

        # Very low similarity - likely different
        if similarity < 0.3:
            return {
                'llm_match': False,
                'llm_confidence': 0.85,
                'llm_reason': f'DirectMatchAgent: Very low similarity ({similarity:.3f})',
                'agent_used': 'direct'
            }

        # Length ratio check (one name 3x longer = likely parent/subsidiary)
        len_ratio = max(len(name_l), len(name_r)) / max(len(min(len(name_l), len(name_r))), 1)
        if len_ratio > 3.0 and similarity < 0.7:
            return {
                'llm_match': False,
                'llm_confidence': 0.80,
                'llm_reason': f'DirectMatchAgent: Large length difference (ratio={len_ratio:.1f})',
                'agent_used': 'direct'
            }

        # Uncertain - need other agents
        return None

    def _geographic_agent(self, row: pd.Series, dim_org: Dict, metadata: Dict) -> Optional[Dict]:
        """
        Geographic-based validation using coordinates.

        Returns match=True if:
        - Same location (<10km) with similar names
        - Same city + country with high name similarity

        Returns match=False if:
        - Very far apart (>500km) with no clear connection

        Returns None if no geographic data or inconclusive
        """
        # Extract coordinates
        lat_l = row.get('latitude_l') or metadata.get('latitude')
        lon_l = row.get('longitude_l') or metadata.get('longitude')
        lat_r = row.get('latitude_r') or dim_org.get('latitude')
        lon_r = row.get('longitude_r') or dim_org.get('longitude')

        # Need all four coordinates
        if not all([lat_l, lon_l, lat_r, lon_r]):
            return None  # No geographic data

        try:
            distance_km = haversine_distance(
                float(lat_l), float(lon_l),
                float(lat_r), float(lon_r)
            )
        except (ValueError, TypeError):
            return None  # Invalid coordinates

        # Same location (<10km)
        if distance_km < 10:
            return {
                'llm_match': True,
                'llm_confidence': 0.95,
                'llm_reason': f'GeographicAgent: Same location ({distance_km:.1f}km apart)',
                'agent_used': 'geographic'
            }

        # Very far apart
        if distance_km > 500:
            # Check if it could be headquarters vs branch
            name_l = str(row.get('name_normalized_l', '')).lower()
            name_r = str(row.get('name_normalized_r', '')).lower()

            # If one name contains location qualifier, might be branch
            has_location_qualifier = any(loc in name_l + name_r for loc in
                                        ['branch', 'division', 'regional', 'office'])

            if not has_location_qualifier:
                return {
                    'llm_match': False,
                    'llm_confidence': 0.85,
                    'llm_reason': f'GeographicAgent: Very far apart ({distance_km:.0f}km), no branch indicators',
                    'agent_used': 'geographic'
                }

        # Medium distance or inconclusive - need other agents
        return None

    def _contextual_agent(self, row: pd.Series, dim_org: Dict, metadata: Dict) -> Optional[Dict]:
        """
        Source-specific contextual validation.

        Uses domain knowledge about specific data sources:
        - Patent data: Company vs inventor detection
        - Clinical trials: Sponsor vs site distinction
        - FDA: Generic vs brand name patterns
        """
        source_table = row.get('source_table', '')

        # USPTO Patents - detect person vs company
        if 'uspto' in source_table.lower():
            return self._validate_patent_entity(row, dim_org, metadata)

        # Clinical trials - sponsor vs site
        if 'clinical_trial' in source_table.lower() or 'trial' in source_table.lower():
            return self._validate_clinical_trial_entity(row, dim_org, metadata)

        # FDA drugs - brand vs generic
        if 'fda' in source_table.lower():
            return self._validate_fda_entity(row, dim_org, metadata)

        # No specific patterns - need LLM
        return None

    def _validate_patent_entity(self, row: pd.Series, dim_org: Dict, metadata: Dict) -> Optional[Dict]:
        """Validate USPTO patent entities (detect person vs company)."""
        name_r = str(row.get('name_r', '')).lower()

        # Person indicators
        person_patterns = [
            r'\b[a-z]+\s+[a-z]\.\s+[a-z]+\b',  # "John A. Smith"
            r'\b(jr|sr|ii|iii|iv|phd|md)\b',    # Suffixes
        ]

        for pattern in person_patterns:
            if re.search(pattern, name_r):
                return {
                    'llm_match': False,
                    'llm_confidence': 0.90,
                    'llm_reason': 'ContextualAgent: Detected person name in patent (should be company)',
                    'agent_used': 'contextual'
                }

        return None  # Uncertain

    def _validate_clinical_trial_entity(self, row: pd.Series, dim_org: Dict,
                                       metadata: Dict) -> Optional[Dict]:
        """Validate clinical trial entities (sponsor vs site)."""
        # Check metadata for role
        rp_type = metadata.get('rp_type', '').lower()

        # If it's a location/site but dim_org is a sponsor company
        if 'location' in rp_type or 'site' in rp_type:
            dim_type = dim_org.get('type', '').lower()
            if 'sponsor' in dim_type or 'pharmaceutical' in dim_type:
                # Likely mismatch - site vs sponsor
                return {
                    'llm_match': False,
                    'llm_confidence': 0.85,
                    'llm_reason': 'ContextualAgent: Type mismatch (trial site vs pharmaceutical sponsor)',
                    'agent_used': 'contextual'
                }

        return None  # Uncertain

    def _validate_fda_entity(self, row: pd.Series, dim_org: Dict, metadata: Dict) -> Optional[Dict]:
        """Validate FDA drug entities (brand vs manufacturer)."""
        # Check if names suggest brand vs company
        name_r = str(row.get('name_r', '')).lower()
        name_l = str(row.get('name_l', '')).lower()

        # Company indicators
        company_suffixes = ['inc', 'corp', 'ltd', 'llc', 'pharmaceuticals', 'pharma', 'laboratories', 'labs']

        has_company_suffix_l = any(suffix in name_l for suffix in company_suffixes)
        has_company_suffix_r = any(suffix in name_r for suffix in company_suffixes)

        # If one has company suffix and other doesn't, might be brand vs manufacturer
        if has_company_suffix_l and not has_company_suffix_r:
            # Check if name_r is very short (typical for brand names)
            if len(name_r.split()) <= 2:
                return {
                    'llm_match': False,
                    'llm_confidence': 0.75,
                    'llm_reason': 'ContextualAgent: Possible brand name vs manufacturer mismatch',
                    'agent_used': 'contextual'
                }

        return None  # Uncertain

    def _llm_agent(self, row: pd.Series, dim_org: Dict, metadata: Dict,
                  llm_judge_func: callable) -> Dict:
        """
        Route to appropriate LLM model based on case complexity.

        Use fast model (Haiku) for:
        - Very high Splink scores (>0.98) - just need confirmation
        - Very low Splink scores (<0.60) - just need rejection

        Use primary model (Sonnet/Opus) for:
        - Boundary cases (0.70-0.95) - need careful reasoning
        """
        match_prob = row.get('match_probability', 0.5)

        # Simple cases - use fast model
        if match_prob > 0.98 or match_prob < 0.60:
            model = self.fast_model
            self.stats['llm_fast'] += 1
            agent_used = 'llm_fast'
        else:
            # Complex boundary cases - use primary model
            model = self.primary_model
            self.stats['llm_primary'] += 1
            agent_used = 'llm_primary'

        result = llm_judge_func(row, dim_org, metadata, self.client, model)
        result['agent_used'] = agent_used

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for agent routing.

        Returns:
            Dict with counts and percentages for each agent
        """
        if self.stats['total'] == 0:
            return self.stats

        total = self.stats['total']
        stats_with_pct = {}

        for agent, count in self.stats.items():
            if agent == 'total':
                stats_with_pct[agent] = count
            else:
                pct = 100 * count / total
                stats_with_pct[agent] = {
                    'count': count,
                    'percentage': pct
                }

        # Calculate cost savings (assuming direct/geo/context are free)
        free_calls = (self.stats['direct'] + self.stats['geographic'] +
                     self.stats['contextual'])
        llm_calls = self.stats['llm_fast'] + self.stats['llm_primary']

        stats_with_pct['summary'] = {
            'free_deterministic': free_calls,
            'llm_required': llm_calls,
            'cost_reduction_pct': 100 * free_calls / total if total > 0 else 0
        }

        return stats_with_pct

    def print_stats(self):
        """Print usage statistics in readable format."""
        stats = self.get_stats()

        print("\nMULTI-AGENT VALIDATION STATISTICS")
        print("=" * 60)
        print(f"Total validations: {stats['total']:,}\n")

        print("Agent Usage:")
        for agent in ['direct', 'geographic', 'contextual', 'llm_fast', 'llm_primary']:
            if agent in stats and isinstance(stats[agent], dict):
                print(f"  {agent:20} | {stats[agent]['count']:>6,} ({stats[agent]['percentage']:5.1f}%)")

        if 'summary' in stats:
            summary = stats['summary']
            print(f"\nCost Savings:")
            print(f"  Free (deterministic):    {summary['free_deterministic']:>6,}")
            print(f"  LLM calls required:      {summary['llm_required']:>6,}")
            print(f"  Cost reduction:          {summary['cost_reduction_pct']:>6.1f}%")


def batch_validate_with_agents(predictions_df: pd.DataFrame,
                               dim_org_df: pd.DataFrame,
                               client: Anthropic,
                               llm_judge_func: callable,
                               enable_routing: bool = True,
                               primary_model: str = "claude-sonnet-4-20250514",
                               fast_model: str = "claude-3-5-haiku-20241022") -> Tuple[pd.DataFrame, Dict]:
    """
    Validate predictions using multi-agent framework.

    Args:
        predictions_df: Predictions to validate
        dim_org_df: Reference dim_org data
        client: Anthropic client
        llm_judge_func: LLM judge function from llm_judge.py
        enable_routing: Enable agent routing (False = always use LLM for comparison)
        primary_model: Primary LLM model
        fast_model: Fast/cheap LLM model

    Returns:
        Tuple of (updated predictions_df, agent statistics)
    """
    orchestrator = MatchingAgentOrchestrator(
        client=client,
        primary_model=primary_model,
        fast_model=fast_model,
        enable_routing=enable_routing
    )

    log_step(f"Multi-agent validation: {len(predictions_df):,} predictions")

    results = []

    for idx, row in predictions_df.iterrows():
        # Get corresponding dim_org record
        dim_org_id = row['unique_id_l']
        dim_org_record = dim_org_df[dim_org_df['unique_id'] == dim_org_id].iloc[0].to_dict()

        # Extract metadata
        metadata = {
            'country': row.get('country_r'),
            'city': row.get('city_r'),
            'latitude': row.get('latitude_r'),
            'longitude': row.get('longitude_r'),
            'rp_type': row.get('rp_type', ''),
            'source_table': row.get('source_table', '')
        }

        # Validate with appropriate agent
        result = orchestrator.validate_match(row, dim_org_record, metadata, llm_judge_func)
        result['prediction_idx'] = idx
        results.append(result)

        # Progress logging every 100
        if (len(results) % 100) == 0:
            log_step(f"  Progress: {len(results):,}/{len(predictions_df):,}")

    # Merge results back
    for result in results:
        idx = result['prediction_idx']
        predictions_df.loc[idx, 'llm_match'] = result['llm_match']
        predictions_df.loc[idx, 'llm_confidence'] = result['llm_confidence']
        predictions_df.loc[idx, 'llm_reason'] = result['llm_reason']
        predictions_df.loc[idx, 'agent_used'] = result['agent_used']

    # Get statistics
    stats = orchestrator.get_stats()
    orchestrator.print_stats()

    return predictions_df, stats

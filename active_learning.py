"""
Active Learning Module for Entity Resolution

Implements HAML-IRL (Hybrid Active Machine Learning for Imbalanced Record Linkage)
approach combining informativeness (uncertainty) + representativeness (diversity).

Key Features:
- Uncertainty-based sampling (model confidence)
- Cluster ambiguity detection (many-to-many matches)
- Source domain diversity (representativeness)
- Budget-aware stratified sampling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils import log_step


def calculate_uncertainty_score(row: pd.Series, cluster_info: Dict[int, int],
                                source_weights: Dict[str, float]) -> float:
    """
    Calculate uncertainty score for a prediction combining multiple factors.

    Args:
        row: Prediction row with match_probability, cluster_id, source_table
        cluster_info: Dict mapping cluster_id to cluster size
        source_weights: Dict mapping source_table to priority weight

    Returns:
        Uncertainty score between 0 and 1 (higher = more uncertain, higher priority)

    Components:
    1. Model uncertainty: Distance from decision threshold
    2. Cluster ambiguity: Number of competing matches in same cluster
    3. Domain divergence: Priority weight for underrepresented sources
    """
    # Model uncertainty (closer to threshold = higher uncertainty)
    threshold = 0.85
    match_prob = row.get('match_probability', 0.5)

    # Distance from threshold (normalized to 0-1)
    if match_prob >= threshold:
        # Above threshold: uncertainty = how close to threshold
        model_uncertainty = 1 - ((match_prob - threshold) / (1 - threshold))
    else:
        # Below threshold: low priority (already filtered out in most cases)
        model_uncertainty = (match_prob / threshold) * 0.5  # Scale down

    # Cluster ambiguity (larger clusters = more ambiguous)
    cluster_id = row.get('cluster_id', -1)
    cluster_size = cluster_info.get(cluster_id, 1)
    # Normalize: clusters of 10+ get max score
    cluster_uncertainty = min(cluster_size / 10.0, 1.0)

    # Source domain divergence (prioritize less common/harder sources)
    source = row.get('source_table', '')
    domain_weight = source_weights.get(source, 0.5)

    # Combined score with weights
    # 40% model uncertainty, 30% cluster ambiguity, 30% domain importance
    combined = (0.4 * model_uncertainty +
                0.3 * cluster_uncertainty +
                0.3 * domain_weight)

    return combined


def get_source_weights(predictions_df: pd.DataFrame,
                       manual_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Calculate or use manual source priority weights.

    Args:
        predictions_df: DataFrame with source_table column
        manual_weights: Optional manual weights (0-1, higher = more important)

    Returns:
        Dict mapping source_table to weight

    Default strategy: Prioritize rare sources (inverse frequency)
    """
    if manual_weights is not None:
        return manual_weights

    # Inverse frequency weighting
    source_counts = predictions_df['source_table'].value_counts()
    total = len(predictions_df)

    # Weight = 1 - (frequency / max_frequency)
    # Rare sources get higher weight
    max_count = source_counts.max()
    weights = {}

    for source, count in source_counts.items():
        # Normalize to 0-1, invert so rare sources are higher
        freq = count / max_count
        weights[source] = 1 - (freq * 0.7)  # Scale to [0.3, 1.0] range

    return weights


def select_active_learning_batch(predictions_df: pd.DataFrame,
                                 clusters_df: pd.DataFrame,
                                 budget: int = 500,
                                 min_per_source: int = 10,
                                 source_weights: Optional[Dict[str, float]] = None,
                                 random_seed: int = 42) -> pd.DataFrame:
    """
    Select top N predictions for LLM validation using HAML-IRL approach.

    Combines:
    - Informativeness: High uncertainty samples that reduce model uncertainty
    - Representativeness: Stratified sampling across sources for diversity

    Args:
        predictions_df: All predictions with match_probability, source_table, cluster_id
        clusters_df: Cluster assignments with unique_id, cluster_id
        budget: Total number of samples to select
        min_per_source: Minimum samples per source (if available)
        source_weights: Optional manual source priority weights
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame of selected predictions for LLM validation
    """
    log_step(f"Active Learning: Selecting {budget} samples from {len(predictions_df):,} predictions")

    # Get cluster sizes
    cluster_sizes = clusters_df.groupby('cluster_id').size().to_dict()

    # Get or calculate source weights
    weights = get_source_weights(predictions_df, source_weights)

    # Calculate uncertainty scores
    predictions_df = predictions_df.copy()
    predictions_df['uncertainty_score'] = predictions_df.apply(
        lambda row: calculate_uncertainty_score(row, cluster_sizes, weights),
        axis=1
    )

    # Stratified sampling by source (representativeness)
    source_counts = predictions_df['source_table'].value_counts()
    total_sources = len(source_counts)

    # Allocate budget across sources proportionally with minimum
    samples_per_source = {}
    remaining_budget = budget

    # First pass: ensure minimum per source
    for source, count in source_counts.items():
        min_samples = min(min_per_source, count, remaining_budget)
        samples_per_source[source] = min_samples
        remaining_budget -= min_samples

    # Second pass: allocate remaining budget proportionally by frequency
    if remaining_budget > 0:
        total_count = source_counts.sum()
        for source, count in source_counts.items():
            proportion = count / total_count
            additional = int(remaining_budget * proportion)
            # Don't exceed available predictions
            max_additional = count - samples_per_source[source]
            samples_per_source[source] += min(additional, max_additional)

    # Sample from each source by uncertainty score
    selected_samples = []

    for source, n_samples in samples_per_source.items():
        if n_samples == 0:
            continue

        source_df = predictions_df[predictions_df['source_table'] == source]

        # Select top uncertain samples
        if len(source_df) <= n_samples:
            # Take all if not enough
            selected = source_df
        else:
            # Take top N by uncertainty
            selected = source_df.nlargest(n_samples, 'uncertainty_score')

        selected_samples.append(selected)

        log_step(f"  {source}: {len(selected)}/{len(source_df)} samples "
                f"(avg uncertainty: {selected['uncertainty_score'].mean():.3f})")

    # Combine all selected samples
    result = pd.concat(selected_samples, ignore_index=True)

    # Sort by uncertainty (highest first)
    result = result.sort_values('uncertainty_score', ascending=False)

    log_step(f"Selected {len(result):,} samples across {len(selected_samples)} sources")
    log_step(f"  Avg uncertainty: {result['uncertainty_score'].mean():.3f}")
    log_step(f"  Avg match_probability: {result['match_probability'].mean():.3f}")

    return result


def get_disagreement_samples(predictions_df: pd.DataFrame,
                             min_splink_score: float = 0.95,
                             max_splink_score: float = 0.60) -> pd.DataFrame:
    """
    Identify predictions where Splink and LLM strongly disagree.

    These are high-value candidates for human review to identify:
    - Systematic Splink errors (update m/u parameters)
    - LLM prompt issues (refine prompts)
    - Edge cases for model improvement

    Args:
        predictions_df: Predictions with match_probability and llm_match
        min_splink_score: Splink threshold for "high confidence"
        max_splink_score: Splink threshold for "low confidence"

    Returns:
        DataFrame of disagreement cases
    """
    # Require llm_match column
    if 'llm_match' not in predictions_df.columns:
        log_step("No LLM validation results found - cannot identify disagreements", "WARN")
        return pd.DataFrame()

    # Type 1: High Splink confidence but LLM rejects
    high_splink_rejected = predictions_df[
        (predictions_df['match_probability'] >= min_splink_score) &
        (predictions_df['llm_match'] == False)
    ].copy()
    high_splink_rejected['disagreement_type'] = 'high_splink_rejected'

    # Type 2: Low Splink confidence but LLM confirms
    low_splink_confirmed = predictions_df[
        (predictions_df['match_probability'] <= max_splink_score) &
        (predictions_df['llm_match'] == True)
    ].copy()
    low_splink_confirmed['disagreement_type'] = 'low_splink_confirmed'

    # Combine
    disagreements = pd.concat([high_splink_rejected, low_splink_confirmed], ignore_index=True)

    log_step(f"Found {len(disagreements):,} disagreements:")
    log_step(f"  High Splink rejected by LLM: {len(high_splink_rejected):,}")
    log_step(f"  Low Splink confirmed by LLM: {len(low_splink_confirmed):,}")

    return disagreements


def propagate_llm_labels(predictions_df: pd.DataFrame,
                        clusters_df: pd.DataFrame,
                        confidence_threshold: float = 0.90) -> pd.DataFrame:
    """
    Propagate LLM labels to similar predictions within clusters.

    Strategy:
    - If LLM validates one prediction in a cluster with high confidence,
      boost confidence for similar predictions in same cluster
    - Reduces need for redundant LLM calls on near-duplicates

    Args:
        predictions_df: Predictions with some LLM validation
        clusters_df: Cluster assignments
        confidence_threshold: Minimum LLM confidence to propagate

    Returns:
        Updated predictions_df with propagated labels
    """
    if 'llm_match' not in predictions_df.columns or 'llm_confidence' not in predictions_df.columns:
        log_step("No LLM validation to propagate", "WARN")
        return predictions_df

    predictions_df = predictions_df.copy()

    # Get validated predictions (high confidence)
    validated = predictions_df[
        (predictions_df['llm_match'].notna()) &
        (predictions_df['llm_confidence'] >= confidence_threshold)
    ]

    if len(validated) == 0:
        log_step("No high-confidence LLM validations to propagate", "WARN")
        return predictions_df

    propagated_count = 0

    # For each cluster with validated predictions
    for cluster_id in validated['cluster_id'].unique():
        cluster_validated = validated[validated['cluster_id'] == cluster_id]

        # Get consensus label (majority vote)
        match_votes = cluster_validated['llm_match'].sum()
        total_votes = len(cluster_validated)

        if match_votes / total_votes >= 0.7:  # 70% agree it's a match
            cluster_label = True
            avg_confidence = cluster_validated[cluster_validated['llm_match'] == True]['llm_confidence'].mean()
        elif match_votes / total_votes <= 0.3:  # 70% agree it's not a match
            cluster_label = False
            avg_confidence = cluster_validated[cluster_validated['llm_match'] == False]['llm_confidence'].mean()
        else:
            # No consensus - skip propagation
            continue

        # Apply to unvalidated predictions in same cluster
        unvalidated_mask = (
            (predictions_df['cluster_id'] == cluster_id) &
            (predictions_df['llm_match'].isna())
        )

        if unvalidated_mask.sum() > 0:
            # Propagate with reduced confidence
            predictions_df.loc[unvalidated_mask, 'llm_match'] = cluster_label
            predictions_df.loc[unvalidated_mask, 'llm_confidence'] = avg_confidence * 0.8  # 20% penalty
            predictions_df.loc[unvalidated_mask, 'llm_reason'] = f'Propagated from cluster (n={total_votes})'

            propagated_count += unvalidated_mask.sum()

    log_step(f"Propagated LLM labels to {propagated_count:,} predictions")

    return predictions_df


def analyze_uncertainty_distribution(predictions_df: pd.DataFrame) -> Dict:
    """
    Analyze uncertainty score distribution for reporting.

    Returns:
        Dict with statistics about uncertainty distribution
    """
    if 'uncertainty_score' not in predictions_df.columns:
        return {}

    stats = {
        'mean': predictions_df['uncertainty_score'].mean(),
        'median': predictions_df['uncertainty_score'].median(),
        'std': predictions_df['uncertainty_score'].std(),
        'min': predictions_df['uncertainty_score'].min(),
        'max': predictions_df['uncertainty_score'].max(),
        'quartiles': {
            'q25': predictions_df['uncertainty_score'].quantile(0.25),
            'q50': predictions_df['uncertainty_score'].quantile(0.50),
            'q75': predictions_df['uncertainty_score'].quantile(0.75)
        }
    }

    # By source
    by_source = predictions_df.groupby('source_table')['uncertainty_score'].agg(['mean', 'std', 'count'])
    stats['by_source'] = by_source.to_dict('index')

    return stats

"""
Feedback Collection and Continuous Improvement Module

Implements human-in-the-loop feedback system for:
1. Identifying high-value cases for human review
2. Recording human judgments
3. Analyzing disagreements between Splink, LLM, and humans
4. Using feedback to improve prompts and model parameters

Key Features:
- Disagreement detection (Splink vs LLM)
- Human annotation interface
- Feedback storage and retrieval
- Analysis and insights for model improvement
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from utils import log_step, save_json, load_json


class FeedbackCollector:
    """
    Collect and manage human feedback on entity resolution predictions.

    Workflow:
    1. Identify disagreements and edge cases
    2. Present to human reviewers
    3. Record judgments
    4. Analyze patterns
    5. Generate recommendations for improvement
    """

    def __init__(self, feedback_file: str = "data/feedback.jsonl",
                 config_file: str = "data/feedback_config.json"):
        """
        Initialize feedback collector.

        Args:
            feedback_file: Path to JSONL file storing feedback
            config_file: Path to JSON config file
        """
        self.feedback_file = Path(feedback_file)
        self.config_file = Path(config_file)

        # Create directory if needed
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        if self.config_file.exists():
            self.config = load_json(str(self.config_file), "Feedback config")
        else:
            self.config = {
                'created_at': datetime.now().isoformat(),
                'total_reviews': 0,
                'reviewers': []
            }
            save_json(self.config, str(self.config_file), "Feedback config")

    def identify_review_candidates(self, predictions_df: pd.DataFrame,
                                   max_candidates: int = 100,
                                   priority: str = 'disagreements') -> pd.DataFrame:
        """
        Identify predictions that should be reviewed by humans.

        Priority strategies:
        - 'disagreements': Splink vs LLM strong disagreements
        - 'uncertainty': High uncertainty from both models
        - 'diversity': Diverse sample across sources and score ranges
        - 'errors': Previously identified error patterns

        Args:
            predictions_df: Predictions with Splink and LLM scores
            max_candidates: Maximum number to return
            priority: Selection strategy

        Returns:
            DataFrame of candidates for human review
        """
        log_step(f"Identifying review candidates (priority={priority}, max={max_candidates})")

        if priority == 'disagreements':
            candidates = self._get_disagreement_candidates(predictions_df)
        elif priority == 'uncertainty':
            candidates = self._get_uncertainty_candidates(predictions_df)
        elif priority == 'diversity':
            candidates = self._get_diverse_candidates(predictions_df)
        else:
            log_step(f"Unknown priority '{priority}', using disagreements", "WARN")
            candidates = self._get_disagreement_candidates(predictions_df)

        # Limit to max_candidates
        if len(candidates) > max_candidates:
            candidates = candidates.head(max_candidates)

        log_step(f"Identified {len(candidates):,} review candidates")

        return candidates

    def _get_disagreement_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get predictions where Splink and LLM strongly disagree."""
        if 'llm_match' not in df.columns:
            log_step("No LLM validation found - cannot identify disagreements", "WARN")
            return pd.DataFrame()

        # Type 1: High Splink (>0.95) but LLM rejects
        type1 = df[
            (df['match_probability'] >= 0.95) &
            (df['llm_match'] == False)
        ].copy()
        type1['disagreement_type'] = 'high_splink_rejected'
        type1['priority_score'] = df['match_probability'] * (1 - df.get('llm_confidence', 0.5))

        # Type 2: Low Splink (<0.70) but LLM confirms
        type2 = df[
            (df['match_probability'] <= 0.70) &
            (df['llm_match'] == True)
        ].copy()
        type2['disagreement_type'] = 'low_splink_confirmed'
        type2['priority_score'] = (1 - df['match_probability']) * df.get('llm_confidence', 0.5)

        # Type 3: Medium Splink (0.70-0.95) with low LLM confidence
        type3 = df[
            (df['match_probability'] >= 0.70) &
            (df['match_probability'] < 0.95) &
            (df.get('llm_confidence', 1.0) < 0.70)
        ].copy()
        type3['disagreement_type'] = 'both_uncertain'
        type3['priority_score'] = 1 - df.get('llm_confidence', 0.5)

        # Combine and sort by priority
        candidates = pd.concat([type1, type2, type3], ignore_index=True)
        candidates = candidates.sort_values('priority_score', ascending=False)

        return candidates

    def _get_uncertainty_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get predictions with high uncertainty from both models."""
        # Calculate combined uncertainty
        df = df.copy()

        # Splink uncertainty (distance from 0.5 or thresholds)
        df['splink_uncertainty'] = 1 - abs(df['match_probability'] - 0.85)  # 0.85 = typical threshold

        # LLM uncertainty
        if 'llm_confidence' in df.columns:
            df['llm_uncertainty'] = 1 - df['llm_confidence']
        else:
            df['llm_uncertainty'] = 1.0

        # Combined uncertainty
        df['combined_uncertainty'] = (df['splink_uncertainty'] + df['llm_uncertainty']) / 2

        # Select most uncertain
        candidates = df.nlargest(1000, 'combined_uncertainty')
        candidates['disagreement_type'] = 'high_uncertainty'

        return candidates

    def _get_diverse_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get diverse sample across sources and score ranges."""
        candidates = []

        # Stratify by source
        for source in df['source_table'].unique():
            source_df = df[df['source_table'] == source]

            # Sample across score ranges
            for score_min, score_max in [(0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]:
                range_df = source_df[
                    (source_df['match_probability'] >= score_min) &
                    (source_df['match_probability'] < score_max)
                ]

                if len(range_df) > 0:
                    # Sample 2-5 from each bucket
                    n = min(5, len(range_df))
                    sample = range_df.sample(n)
                    candidates.append(sample)

        if candidates:
            result = pd.concat(candidates, ignore_index=True)
            result['disagreement_type'] = 'diversity_sample'
            return result
        else:
            return pd.DataFrame()

    def record_feedback(self, unique_id_l: str, unique_id_r: str,
                       splink_score: float, llm_match: Optional[bool],
                       llm_confidence: Optional[float],
                       human_label: bool, human_confidence: float,
                       human_reason: str, reviewer_name: str,
                       additional_metadata: Optional[Dict] = None) -> None:
        """
        Record human feedback on a prediction.

        Args:
            unique_id_l: Left entity ID (dim_org)
            unique_id_r: Right entity ID (mismatched)
            splink_score: Splink match probability
            llm_match: LLM judgment (True/False/None)
            llm_confidence: LLM confidence score
            human_label: Human judgment (True=match, False=not match)
            human_confidence: Human confidence (0-1)
            human_reason: Explanation for judgment
            reviewer_name: Name/ID of human reviewer
            additional_metadata: Optional extra info
        """
        feedback = {
            'unique_id_l': unique_id_l,
            'unique_id_r': unique_id_r,
            'splink_score': float(splink_score),
            'llm_match': llm_match,
            'llm_confidence': float(llm_confidence) if llm_confidence is not None else None,
            'human_label': bool(human_label),
            'human_confidence': float(human_confidence),
            'human_reason': human_reason,
            'reviewer_name': reviewer_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': additional_metadata or {}
        }

        # Append to JSONL file
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback) + '\n')

        # Update config
        self.config['total_reviews'] += 1
        if reviewer_name not in self.config['reviewers']:
            self.config['reviewers'].append(reviewer_name)
        save_json(self.config, str(self.config_file), "Feedback config")

        log_step(f"Recorded feedback: {unique_id_l} <-> {unique_id_r} = {human_label}")

    def load_feedback(self) -> pd.DataFrame:
        """
        Load all feedback records.

        Returns:
            DataFrame with all feedback
        """
        if not self.feedback_file.exists():
            log_step("No feedback file found", "WARN")
            return pd.DataFrame()

        records = []
        with open(self.feedback_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    log_step(f"Skipping invalid JSON line", "WARN")
                    continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        log_step(f"Loaded {len(df):,} feedback records")

        return df

    def analyze_feedback(self) -> Dict:
        """
        Analyze feedback to identify patterns and improvement opportunities.

        Returns:
            Dict with analysis results and recommendations
        """
        feedback_df = self.load_feedback()

        if feedback_df.empty:
            return {'error': 'No feedback available'}

        analysis = {
            'summary': self._analyze_summary(feedback_df),
            'disagreements': self._analyze_disagreements(feedback_df),
            'patterns': self._analyze_error_patterns(feedback_df),
            'recommendations': []
        }

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _analyze_summary(self, df: pd.DataFrame) -> Dict:
        """Basic summary statistics."""
        return {
            'total_reviews': len(df),
            'reviewers': df['reviewer_name'].nunique(),
            'human_match_rate': (df['human_label'] == True).mean(),
            'avg_human_confidence': df['human_confidence'].mean(),
            'date_range': {
                'first': df['timestamp'].min(),
                'last': df['timestamp'].max()
            }
        }

    def _analyze_disagreements(self, df: pd.DataFrame) -> Dict:
        """Analyze where models disagree with humans."""
        # Splink vs Human
        df['splink_predicted'] = df['splink_score'] >= 0.85  # threshold
        splink_vs_human = (df['splink_predicted'] != df['human_label']).sum()
        splink_accuracy = (df['splink_predicted'] == df['human_label']).mean()

        # LLM vs Human
        llm_valid = df['llm_match'].notna()
        if llm_valid.sum() > 0:
            llm_vs_human = (df.loc[llm_valid, 'llm_match'] !=
                           df.loc[llm_valid, 'human_label']).sum()
            llm_accuracy = (df.loc[llm_valid, 'llm_match'] ==
                           df.loc[llm_valid, 'human_label']).mean()
        else:
            llm_vs_human = 0
            llm_accuracy = None

        return {
            'splink_disagreements': int(splink_vs_human),
            'splink_accuracy': float(splink_accuracy),
            'llm_disagreements': int(llm_vs_human),
            'llm_accuracy': float(llm_accuracy) if llm_accuracy is not None else None
        }

    def _analyze_error_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify common error patterns."""
        patterns = {}

        # False positives (model says match, human says no)
        df['splink_predicted'] = df['splink_score'] >= 0.85

        # Splink false positives
        splink_fp = df[(df['splink_predicted'] == True) & (df['human_label'] == False)]
        if len(splink_fp) > 0:
            patterns['splink_false_positives'] = {
                'count': len(splink_fp),
                'avg_score': splink_fp['splink_score'].mean(),
                'common_reasons': splink_fp['human_reason'].value_counts().head(5).to_dict()
            }

        # Splink false negatives
        splink_fn = df[(df['splink_predicted'] == False) & (df['human_label'] == True)]
        if len(splink_fn) > 0:
            patterns['splink_false_negatives'] = {
                'count': len(splink_fn),
                'avg_score': splink_fn['splink_score'].mean(),
                'common_reasons': splink_fn['human_reason'].value_counts().head(5).to_dict()
            }

        # LLM errors
        llm_valid = df['llm_match'].notna()
        if llm_valid.sum() > 0:
            llm_fp = df[llm_valid & (df['llm_match'] == True) & (df['human_label'] == False)]
            llm_fn = df[llm_valid & (df['llm_match'] == False) & (df['human_label'] == True)]

            if len(llm_fp) > 0:
                patterns['llm_false_positives'] = {
                    'count': len(llm_fp),
                    'common_reasons': llm_fp['human_reason'].value_counts().head(5).to_dict()
                }

            if len(llm_fn) > 0:
                patterns['llm_false_negatives'] = {
                    'count': len(llm_fn),
                    'common_reasons': llm_fn['human_reason'].value_counts().head(5).to_dict()
                }

        return patterns

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        disagree = analysis.get('disagreements', {})
        patterns = analysis.get('patterns', {})

        # Splink accuracy
        splink_acc = disagree.get('splink_accuracy', 0)
        if splink_acc < 0.80:
            recommendations.append(
                f"CRITICAL: Splink accuracy is low ({splink_acc:.1%}). "
                f"Consider retraining with feedback as ground truth labels."
            )
        elif splink_acc < 0.90:
            recommendations.append(
                f"Splink accuracy is moderate ({splink_acc:.1%}). "
                f"Review m/u parameters for improvement."
            )

        # LLM accuracy
        llm_acc = disagree.get('llm_accuracy')
        if llm_acc is not None:
            if llm_acc < 0.85:
                recommendations.append(
                    f"LLM accuracy is low ({llm_acc:.1%}). "
                    f"Review and refine prompts based on false positive/negative patterns."
                )

        # False positive patterns
        if 'splink_false_positives' in patterns:
            fp = patterns['splink_false_positives']
            if fp['count'] > 5:
                recommendations.append(
                    f"Splink has {fp['count']} false positives (avg score: {fp['avg_score']:.3f}). "
                    f"Common reasons: {list(fp['common_reasons'].keys())[:3]}. "
                    f"Consider adjusting threshold or adding blocking rules."
                )

        # False negative patterns
        if 'splink_false_negatives' in patterns:
            fn = patterns['splink_false_negatives']
            if fn['count'] > 5:
                recommendations.append(
                    f"Splink has {fn['count']} false negatives (avg score: {fn['avg_score']:.3f}). "
                    f"Common reasons: {list(fn['common_reasons'].keys())[:3]}. "
                    f"Consider adding more lenient blocking rules or comparison levels."
                )

        if not recommendations:
            recommendations.append("Models performing well! Continue monitoring and collecting feedback.")

        return recommendations

    def export_training_data(self, output_file: str = "data/feedback_training.parquet") -> pd.DataFrame:
        """
        Export human-labeled feedback as training data.

        Can be used to:
        - Retrain Splink with additional ground truth
        - Fine-tune LLM prompts
        - Calibrate confidence scores

        Args:
            output_file: Path to save training data

        Returns:
            DataFrame ready for training
        """
        feedback_df = self.load_feedback()

        if feedback_df.empty:
            log_step("No feedback to export", "WARN")
            return pd.DataFrame()

        # Format for Splink training
        training_data = feedback_df[[
            'unique_id_l', 'unique_id_r', 'human_label'
        ]].rename(columns={'human_label': 'match_label'})

        # Add confidence weights (higher confidence = higher weight)
        training_data['sample_weight'] = feedback_df['human_confidence']

        # Save
        training_data.to_parquet(output_file)
        log_step(f"Exported {len(training_data):,} training examples to {output_file}")

        return training_data

    def print_analysis(self):
        """Print analysis in readable format."""
        analysis = self.analyze_feedback()

        if 'error' in analysis:
            print(f"\nERROR: {analysis['error']}")
            return

        print("\nFEEDBACK ANALYSIS")
        print("=" * 70)

        # Summary
        summary = analysis['summary']
        print(f"\nSummary:")
        print(f"  Total reviews:       {summary['total_reviews']:,}")
        print(f"  Reviewers:           {summary['reviewers']}")
        print(f"  Human match rate:    {summary['human_match_rate']:.1%}")
        print(f"  Avg confidence:      {summary['avg_human_confidence']:.2f}")

        # Disagreements
        disagree = analysis['disagreements']
        print(f"\nModel vs Human Agreement:")
        print(f"  Splink accuracy:     {disagree['splink_accuracy']:.1%} "
              f"({disagree['splink_disagreements']} disagreements)")
        if disagree['llm_accuracy'] is not None:
            print(f"  LLM accuracy:        {disagree['llm_accuracy']:.1%} "
                  f"({disagree['llm_disagreements']} disagreements)")

        # Error patterns
        patterns = analysis['patterns']
        if patterns:
            print(f"\nError Patterns:")
            for error_type, details in patterns.items():
                print(f"  {error_type}: {details['count']} cases")

        # Recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")

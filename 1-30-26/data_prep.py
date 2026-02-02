"""
Entity Resolution Data Preparation (Minimal)
"""

import re
from typing import Optional, List, Tuple, Dict, Set
from math import log, radians, sin, cos, sqrt, atan2
from collections import Counter

import pandas as pd
import numpy as np
import country_converter as coco
import jellyfish
from cleanco import basename as cleanco_basename

from config import config
from utils import DatabaseManager, parse_java_map, safe_float, log_step, timed, Timer


# =============================================================================
# GEOGRAPHIC UTILITIES
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """Calculate distance in km between two coordinates."""
    if any(x is None or pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# =============================================================================
# DATA LOADING
# =============================================================================

@timed("Loading dim_organization")
def load_dim_org(db: DatabaseManager, sample_size: Optional[int] = None) -> pd.DataFrame:
    limit_clause = f"LIMIT {sample_size}" if sample_size else ""
    query = f"""
        SELECT 
            allsci_id, name, type, country_code,
            geographic_location.city as city,
            geographic_location.latitude as latitude,
            geographic_location.longitude as longitude,
            name_aliases,
            ids.ror as ror_id,
            ids.grid as grid_id
        FROM {config.tables.DIM_ORG_TABLE}
        WHERE is_current = true
        {limit_clause}
    """
    df = db.execute_query(query, f"dim_organization (limit={sample_size})")
    df['source'] = 'dim_org'
    return df


@timed("Loading GRID data")
def load_grid(db: DatabaseManager, sample_size: Optional[int] = None) -> pd.DataFrame:
    limit_clause = f"LIMIT {sample_size}" if sample_size else ""
    query = f"""
        SELECT 
            id as grid_id, name, aliases, acronyms, types,
            address.city as city,
            address.country_code as country_code,
            address.latitude as latitude,
            address.longitude as longitude,
            COALESCE(external_ids.ror.preferred, element_at(external_ids.ror.all, 1)) as ror_id,
            organization_parent_ids,
            organization_child_ids
        FROM {config.tables.GRID_TABLE}
        WHERE status = 'active'
        {limit_clause}
    """
    df = db.execute_query(query, f"GRID (limit={sample_size})")
    df['source'] = 'grid'
    return df


@timed("Loading potential_mismatched_organizations")
def load_mismatched(db: DatabaseManager, source_table: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    where_clause = f"WHERE source_table = '{source_table}'" if source_table else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
        SELECT mismatch_id, name, source_table, source_entity_id, metadata
        FROM {config.tables.MISMATCHED_TABLE}
        {where_clause}
        {limit_clause}
    """
    df = db.execute_query(query, f"mismatched (source={source_table}, limit={limit})")
    df = parse_mismatched_metadata(df)
    df['source'] = 'mismatched'
    return df


def parse_mismatched_metadata(df: pd.DataFrame) -> pd.DataFrame:
    log_step("Parsing metadata...")
    df['metadata_parsed'] = df['metadata'].apply(parse_java_map)
    df['name_clean'] = df['metadata_parsed'].apply(lambda x: x.get('name_clean'))
    df['name_normalized'] = df['metadata_parsed'].apply(lambda x: x.get('name_normalized'))
    df['name_prefix_5'] = df['metadata_parsed'].apply(lambda x: x.get('name_prefix_5'))
    df['name_prefix_10'] = df['metadata_parsed'].apply(lambda x: x.get('name_prefix_10'))
    df['type'] = df['metadata_parsed'].apply(lambda x: x.get('class') or x.get('category'))
    df['country'] = df['metadata_parsed'].apply(lambda x: x.get('country') or x.get('countries'))
    df['city'] = df['metadata_parsed'].apply(lambda x: x.get('city'))
    df['state'] = df['metadata_parsed'].apply(lambda x: x.get('state'))
    df['latitude'] = df['metadata_parsed'].apply(lambda x: safe_float(x.get('latitude')))
    df['longitude'] = df['metadata_parsed'].apply(lambda x: safe_float(x.get('longitude')))
    df['name_aliases'] = df['metadata_parsed'].apply(lambda x: x.get('name_aliases'))
    df['rp_type'] = df['metadata_parsed'].apply(lambda x: x.get('rp_type'))
    df['raw_metadata'] = df['metadata']
    df = df.drop(columns=['metadata', 'metadata_parsed'])
    log_step(f"  Extracted {len(df):,} records with metadata fields")
    return df


# =============================================================================
# NAME PROCESSING
# =============================================================================

def clean_name(name: str) -> str:
    if pd.isna(name) or not isinstance(name, str):
        return None
    cleaned = name.lower().strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned if cleaned else None


def normalize_name(name: str) -> str:
    if pd.isna(name) or not isinstance(name, str):
        return None
    normalized = name.strip()
    normalized = re.sub(r'\s*\([^)]*\)\s*$', '', normalized)
    try:
        normalized = cleanco_basename(normalized)
    except Exception:
        pass
    normalized = normalized.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized if normalized else None


_COUNTRY_CODE_CACHE = {}

def normalize_country_code(value) -> Optional[str]:
    if pd.isna(value) or not isinstance(value, str) or len(value.strip()) == 0:
        return None
    value = value.strip()
    if ';' in value:
        value = value.split(';')[0].strip()
    if value in _COUNTRY_CODE_CACHE:
        return _COUNTRY_CODE_CACHE[value]
    if len(value) == 2:
        result = value.upper()
    else:
        result = coco.convert(value, to='ISO2', not_found=None)
        result = result if result and result != 'not found' else None
    _COUNTRY_CODE_CACHE[value] = result
    return result


def create_name_array(df: pd.DataFrame) -> pd.Series:
    def parse_bracketed_string(s):
        """Parse '[Name1, Name2, Acronym]' into ['Name1', 'Name2', 'Acronym']"""
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]  # strip brackets
            return [part.strip() for part in inner.split(',') if part.strip()]
        return [s]
    
    def combine_names(row):
        names = set()
        if pd.notna(row.get('name')):
            names.add(str(row['name']).strip())
        aliases = row.get('name_aliases') or row.get('aliases')
        if aliases:
            if isinstance(aliases, (list, tuple)):
                for a in aliases:
                    if a and isinstance(a, str) and a.startswith('['):
                        names.update(parse_bracketed_string(a))
                    elif a:
                        names.add(str(a).strip())
            elif isinstance(aliases, str):
                if aliases.startswith('['):
                    names.update(parse_bracketed_string(aliases))
                else:
                    names.add(aliases.strip())
        acronyms = row.get('acronyms')
        if acronyms and isinstance(acronyms, (list, tuple)):
            names.update(str(a).strip() for a in acronyms if a)
        return list(names) if names else None
    return df.apply(combine_names, axis=1)


def is_acronym(name: str) -> bool:
    """Check if name is a short acronym (all caps, < 6 chars)."""
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    return len(clean) < 6 and clean.isupper()


def augment_with_aliases(df: pd.DataFrame, swap_fraction: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, List[Dict]]:
    """Randomly swap name with an alias, updating all derived features."""
    import random
    df = df.copy()
    random.seed(seed)
    swap_log = []
    
    def has_multiple_names(x):
        try:
            return len(x) > 1
        except:
            return False
    
    has_aliases = df['all_names'].apply(has_multiple_names)
    candidates = df[has_aliases].index.tolist()
    n_swap = int(len(candidates) * swap_fraction)
    swap_indices = random.sample(candidates, n_swap)
    
    for idx in swap_indices:
        aliases = df.at[idx, 'all_names']
        original_name = df.at[idx, 'name']
        other_names = [a for a in aliases if a.lower() != original_name.lower()]
        other_names = [a for a in other_names if not (a.startswith('[') and ',' in a)]
        other_names = [a for a in other_names if not is_acronym(a)]
        
        if other_names:
            new_name = random.choice(other_names)
            swap_log.append({'original': original_name, 'swapped_to': new_name})
            df.at[idx, 'name'] = new_name
            new_clean = clean_name(new_name)
            new_normalized = normalize_name(new_clean)
            df.at[idx, 'name_clean'] = new_clean
            df.at[idx, 'name_normalized'] = new_normalized
            df.at[idx, 'name_prefix_5'] = new_normalized[:5] if new_normalized else None
            df.at[idx, 'name_prefix_10'] = new_normalized[:10] if new_normalized else None
            df.at[idx, 'name_first_word'] = new_normalized.split()[0] if new_normalized else None
    
    log_step(f"Swapped names for {len(swap_log):,} records ({swap_fraction*100:.0f}%)")
    return df, swap_log


# =============================================================================
# UNIFIED SCHEMA
# =============================================================================

def create_unified_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    log_step(f"Creating unified schema for {source}...")
    result = pd.DataFrame()
    
    if source == 'dim_org':
        result['unique_id'] = 'dim_' + df['allsci_id'].astype(str)
    elif source == 'grid':
        result['unique_id'] = 'grid_' + df['grid_id'].astype(str)
    elif source == 'mismatched':
        result['unique_id'] = 'mis_' + df['mismatch_id'].astype(str)
    else:
        result['unique_id'] = df.index.astype(str)

    result['name'] = df['name']
    result['name_clean'] = df.get('name_clean', df['name'].apply(clean_name))
    result['name_normalized'] = df.get('name_normalized', result['name_clean'].apply(normalize_name))
    result['name_prefix_5'] = df.get('name_prefix_5', result['name_normalized'].str[:5])
    result['name_prefix_10'] = df.get('name_prefix_10', result['name_normalized'].str[:10])
    result['all_names'] = create_name_array(df)
    result['org_type'] = df.get('type', df.get('types'))
    raw_country = df.get('country_code', df.get('country'))
    result['country_code'] = raw_country.apply(normalize_country_code)
    result['city'] = df.get('city')
    result['latitude'] = df.get('latitude').apply(safe_float)
    result['longitude'] = df.get('longitude').apply(safe_float)
    result['ror_id'] = df.get('ror_id')
    result['grid_id'] = df.get('grid_id')
    result['source'] = source
    result['source_id'] = df.get('allsci_id', df.get('grid_id', df.get('mismatch_id')))

    if source == 'mismatched':
        result['source_table'] = df.get('source_table')
        result['source_entity_id'] = df.get('source_entity_id')

    log_step(f"  Created unified schema: {len(result):,} rows, {len(result.columns)} columns")
    return result


# =============================================================================
# FILTERING
# =============================================================================

def is_generic_name(name: str) -> bool:
    if pd.isna(name) or not isinstance(name, str):
        return False
    name_lower = name.lower().strip()
    for pattern in config.filtering.GENERIC_PATTERNS:
        if pattern in name_lower:
            return True
    return False


def filter_bad_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log_step("Filtering bad records...")
    original_count = len(df)
    df['_remove'] = False
    df['_remove_reason'] = ''

    mask_null = df['name'].isna() | (df['name'].str.strip() == '')
    df.loc[mask_null, '_remove'] = True
    df.loc[mask_null, '_remove_reason'] = 'null_name'

    mask_short = df['name'].str.len() < config.filtering.MIN_NAME_LENGTH
    df.loc[mask_short & ~df['_remove'], '_remove'] = True
    df.loc[mask_short & (df['_remove_reason'] == ''), '_remove_reason'] = 'short_name'

    mask_long = df['name'].str.len() > config.filtering.MAX_NAME_LENGTH
    df.loc[mask_long & ~df['_remove'], '_remove'] = True
    df.loc[mask_long & (df['_remove_reason'] == ''), '_remove_reason'] = 'long_name'

    mask_generic = df['name'].apply(is_generic_name)
    df.loc[mask_generic & ~df['_remove'], '_remove'] = True
    df.loc[mask_generic & (df['_remove_reason'] == ''), '_remove_reason'] = 'generic_name'

    if 'rp_type' in df.columns:
        mask_person = df['rp_type'] == config.filtering.PERSON_RECORD_TYPE
        df.loc[mask_person & ~df['_remove'], '_remove'] = True
        df.loc[mask_person & (df['_remove_reason'] == ''), '_remove_reason'] = 'person_record'

    removed_df = df[df['_remove']].copy()
    filtered_df = df[~df['_remove']].drop(columns=['_remove', '_remove_reason'])

    removal_summary = removed_df['_remove_reason'].value_counts()
    log_step(f"  Removed {len(removed_df):,} of {original_count:,} records ({100*len(removed_df)/original_count:.1f}%)")
    for reason, count in removal_summary.items():
        log_step(f"    - {reason}: {count:,}")

    return filtered_df, removed_df


# =============================================================================
# PHONETIC ENCODING
# =============================================================================

def get_first_significant_word(name: str) -> Optional[str]:
    if pd.isna(name) or not isinstance(name, str):
        return None
    skip_prefixes = {
        'the', 'a', 'an', 'university', 'univ', 'college', 'institute',
        'hospital', 'center', 'centre', 'national', 'international',
        'royal', 'state', 'federal', 'central'
    }
    words = name.lower().split()
    for word in words:
        if word not in skip_prefixes and len(word) > 2:
            return word
    return words[0] if words else None


def add_phonetic_codes(df: pd.DataFrame) -> pd.DataFrame:
    log_step("Adding phonetic codes...")
    df['name_first_word'] = df['name_normalized'].apply(get_first_significant_word)
    df['name_soundex'] = df['name_first_word'].apply(
        lambda x: jellyfish.soundex(x) if pd.notna(x) and len(x) > 0 else None
    )
    df['name_metaphone'] = df['name_first_word'].apply(
        lambda x: jellyfish.metaphone(x) if pd.notna(x) and len(x) > 0 else None
    )
    soundex_coverage = df['name_soundex'].notna().sum() / len(df) * 100
    log_step(f"  Phonetic coverage: {soundex_coverage:.1f}%")
    return df


# =============================================================================
# TOKEN STATISTICS (IDF-BASED)
# =============================================================================

def compute_token_statistics(df: pd.DataFrame, name_col: str = 'name_normalized') -> Dict[str, float]:
    log_step("Computing token IDF statistics...")
    names = df[name_col].dropna().tolist()
    n_docs = len(names)
    if n_docs == 0:
        return {}

    doc_freq = Counter()
    for name in names:
        for token in set(str(name).lower().split()):
            if len(token) > 1:
                doc_freq[token] += 1

    idf_scores = {token: log(n_docs / freq) for token, freq in doc_freq.items()}

    n_tokens = len(idf_scores)
    if n_tokens > 0:
        min_idf = min(idf_scores.values())
        max_idf = max(idf_scores.values())
        log_step(f"  Computed IDF for {n_tokens:,} unique tokens")
        log_step(f"  IDF range: {min_idf:.2f} (most common) to {max_idf:.2f} (most rare)")

    return idf_scores


def get_corpus_stopwords(idf_scores: Dict[str, float], percentile: float = 0.10) -> Set[str]:
    if not idf_scores:
        return set()
    threshold = np.percentile(list(idf_scores.values()), percentile * 100)
    stopwords = {token for token, idf in idf_scores.items() if idf <= threshold}
    log_step(f"  Auto-identified {len(stopwords)} corpus stopwords (IDF <= {threshold:.2f})")
    return stopwords


# =============================================================================
# GEOGRAPHIC STOPWORDS
# =============================================================================

_GEOGRAPHIC_STOPWORDS = None

def build_geographic_stopwords() -> Set[str]:
    import pycountry
    import geonamescache

    geo_terms = set()
    for country in pycountry.countries:
        geo_terms.add(country.name.lower())
        if hasattr(country, 'common_name'):
            geo_terms.add(country.common_name.lower())
        if hasattr(country, 'official_name'):
            geo_terms.add(country.official_name.lower())

    for sub in pycountry.subdivisions:
        geo_terms.add(sub.name.lower())

    gc = geonamescache.GeonamesCache()
    for city in gc.get_cities().values():
        geo_terms.add(city['name'].lower())

    DEMONYMS = {
        'korean', 'chinese', 'japanese', 'american', 'british', 'english',
        'german', 'french', 'italian', 'spanish', 'indian', 'african',
        'brazilian', 'mexican', 'canadian', 'australian', 'russian',
        'polish', 'dutch', 'swedish', 'norwegian', 'danish', 'swiss',
        'finnish', 'irish', 'scottish', 'welsh', 'turkish', 'greek',
        'egyptian', 'nigerian', 'kenyan', 'ethiopian', 'moroccan',
        'thai', 'vietnamese', 'indonesian', 'malaysian', 'filipino',
        'taiwanese', 'singaporean', 'pakistani', 'bangladeshi', 'nepali',
        'saudi', 'iranian', 'iraqi', 'israeli', 'palestinian', 'lebanese',
        'european', 'asian', 'latin', 'nordic', 'scandinavian', 'arabic',
        'hispanic', 'caribbean', 'pacific', 'atlantic', 'mediterranean', 'philippine',
    }
    geo_terms.update(DEMONYMS)

    SHORT_COUNTRY_NAMES = {
        'korea', 'japan', 'china', 'taiwan', 'vietnam', 'thailand',
        'indonesia', 'malaysia', 'singapore', 'philippines', 'india',
        'pakistan', 'bangladesh', 'nepal', 'iran', 'iraq', 'israel',
        'palestine', 'lebanon', 'saudi', 'egypt', 'nigeria', 'kenya',
        'ethiopia', 'morocco', 'brazil', 'mexico', 'canada', 'usa',
        'britain', 'england', 'scotland', 'wales', 'ireland', 'france',
        'germany', 'italy', 'spain', 'netherlands', 'belgium', 'sweden',
        'norway', 'denmark', 'finland', 'poland', 'russia', 'turkey',
        'greece', 'australia', 'zealand',
    }
    geo_terms.update(SHORT_COUNTRY_NAMES)

    QUALIFIERS = {
        'northern', 'southern', 'eastern', 'western', 'central',
        'north', 'south', 'east', 'west', 'greater', 'metro',
        'regional', 'provincial', 'municipal', 'county', 'state',
        'national', 'federal', 'republic', 'kingdom', 'prefecture',
        'district', 'territory', 'province', 'region', 'area',
    }
    geo_terms.update(QUALIFIERS)

    ORGANIZATIONAL_STOPWORDS = {
        'association', 'society', 'institute', 'institution', 'organization',
        'foundation', 'corporation', 'company', 'group', 'consortium',
        'university', 'college', 'school', 'academy', 'center', 'centre',
        'hospital', 'clinic', 'medical', 'health', 'healthcare',
        'department', 'division', 'office', 'bureau', 'agency',
        'laboratory', 'research', 'sciences', 'studies', 'program',
        'network', 'system', 'systems', 'services', 'service',
        'international', 'global', 'world', 'worldwide',
        'general', 'special', 'advanced', 'applied', 'basic',
        'public', 'private', 'community', 'professional',
    }
    geo_terms.update(ORGANIZATIONAL_STOPWORDS)

    return geo_terms


def get_geographic_stopwords() -> Set[str]:
    global _GEOGRAPHIC_STOPWORDS
    if _GEOGRAPHIC_STOPWORDS is None:
        _GEOGRAPHIC_STOPWORDS = build_geographic_stopwords()
        log_step(f"  Loaded {len(_GEOGRAPHIC_STOPWORDS):,} geographic stopwords")
    return _GEOGRAPHIC_STOPWORDS


# =============================================================================
# DISTINCTIVE TOKENS
# =============================================================================

def get_distinctive_token_fast(name: str, idf_scores: Dict[str, float], combined_stopwords: Set[str]) -> Optional[str]:
    if pd.isna(name) or not isinstance(name, str):
        return None
    tokens = [t for t in name.lower().split() if len(t) > 3 and t not in combined_stopwords]
    if not tokens:
        tokens = [t for t in name.lower().split() if len(t) >= 3 and t not in combined_stopwords]
    if not tokens and len(name.strip()) >= 3:
        tokens = [name.lower().strip()]
    if not tokens:
        return None
    return max(tokens, key=lambda t: idf_scores.get(t, float('inf')))


def get_distinctive_tokens_fast(name: str, idf_scores: Dict[str, float], combined_stopwords: Set[str], n: int = 3) -> List[str]:
    if pd.isna(name) or not isinstance(name, str):
        return []
    tokens = [t for t in name.lower().split() if len(t) > 3 and t not in combined_stopwords]
    if not tokens:
        tokens = [t for t in name.lower().split() if len(t) >= 3 and t not in combined_stopwords]
    if not tokens and len(name.strip()) >= 3:
        tokens = [name.lower().strip()]
    if not tokens:
        return []
    sorted_tokens = sorted(tokens, key=lambda t: idf_scores.get(t, float('inf')), reverse=True)
    return sorted_tokens[:n]


def add_distinctive_tokens(df: pd.DataFrame, idf_scores: Dict[str, float], stopwords: Set[str] = None) -> pd.DataFrame:
    log_step("Adding distinctive tokens...")
    geo_stopwords = get_geographic_stopwords()
    log_step(f"  Geographic stopwords: {len(geo_stopwords):,} (locations filtered out)")

    df['distinctive_tokens'] = df['name_normalized'].apply(
        lambda x: get_distinctive_tokens_fast(x, idf_scores, geo_stopwords, n=3)
    )
    df['distinctive_token'] = df['name_normalized'].apply(
        lambda x: get_distinctive_token_fast(x, idf_scores, geo_stopwords)
    )
    df['distinctive_soundex'] = df['distinctive_token'].apply(
        lambda x: jellyfish.soundex(x) if pd.notna(x) and len(x) > 0 else None
    )

    coverage = df['distinctive_tokens'].apply(lambda x: len(x) > 0).sum() / len(df) * 100
    log_step(f"  Distinctive token coverage: {coverage:.1f}%")
    return df


# =============================================================================
# TOKEN SET FOR CONTAINMENT
# =============================================================================

def create_token_set(name: str, stopwords: Set[str] = None) -> Set[str]:
    if pd.isna(name) or not isinstance(name, str):
        return set()
    GRAMMATICAL_STOPS = {'of', 'the', 'and', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
    tokens = set(name.lower().split())
    return tokens - GRAMMATICAL_STOPS


def add_token_set_features(df: pd.DataFrame, stopwords: Set[str] = None) -> pd.DataFrame:
    log_step("Adding token set features for containment detection...")
    df['name_tokens'] = df['name_normalized'].apply(
        lambda x: sorted(list(create_token_set(x, stopwords))) if x else []
    )
    df['name_token_count'] = df['name_tokens'].apply(len)
    avg_tokens = df['name_token_count'].mean()
    log_step(f"  Average token count: {avg_tokens:.1f}")
    return df


# =============================================================================
# BLOCKING KEYS
# =============================================================================

def create_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    log_step("Creating blocking keys...")
    if 'name_normalized' not in df.columns:
        df['name_normalized'] = df['name'].apply(clean_name).apply(normalize_name)
    if 'name_prefix_5' not in df.columns:
        df['name_prefix_5'] = df['name_normalized'].str[:config.blocking.NAME_PREFIX_SHORT]
    if 'name_prefix_10' not in df.columns:
        df['name_prefix_10'] = df['name_normalized'].str[:config.blocking.NAME_PREFIX_LONG]
    if 'name_first_word' not in df.columns:
        df['name_first_word'] = df['name_normalized'].str.split().str[0]
    if 'city_name3' not in df.columns and 'city' in df.columns:
        city_clean = df['city'].fillna('').str.lower().str.strip()
        name3 = df['name_normalized'].str[:3].fillna('')
        df['city_name3'] = city_clean + '_' + name3
        df.loc[(city_clean == '') | (name3 == ''), 'city_name3'] = None
    if 'country_code' in df.columns:
        df['country_code'] = df['country_code'].str.upper().str.strip()
    if 'name_soundex' not in df.columns:
        df = add_phonetic_codes(df)
    log_step(f"  Added blocking keys to {len(df):,} records")
    return df


# =============================================================================
# IDF-BASED FEATURES (COMBINED)
# =============================================================================

def add_idf_based_features(dfs: List[pd.DataFrame], stopword_percentile: float = 0.10) -> Tuple[List[pd.DataFrame], Dict[str, float], Set[str]]:
    log_step(f"Adding IDF-based features to {len(dfs)} DataFrames...")
    all_names = []
    for df in dfs:
        if 'name_normalized' in df.columns:
            all_names.extend(df['name_normalized'].dropna().tolist())
    combined_df = pd.DataFrame({'name_normalized': all_names})
    idf_scores = compute_token_statistics(combined_df, 'name_normalized')
    stopwords = get_corpus_stopwords(idf_scores, stopword_percentile)
    if stopwords:
        sorted_stopwords = sorted(stopwords, key=lambda t: idf_scores.get(t, 0))[:20]
        log_step(f"  Top 20 corpus stopwords: {', '.join(sorted_stopwords)}")
    result_dfs = []
    for df in dfs:
        df_copy = df.copy()
        df_copy = add_distinctive_tokens(df_copy, idf_scores, stopwords)
        result_dfs.append(df_copy)
    return result_dfs, idf_scores, stopwords


# =============================================================================
# GROUND TRUTH
# =============================================================================

def create_ground_truth_pairs(dim_org_df: pd.DataFrame, grid_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log_step("Creating ground truth pairs from ROR and GRID matches...")
    
    # Positive pairs from ROR matches
    dim_with_ror = dim_org_df[dim_org_df['ror_id'].notna()].copy()
    grid_with_ror = grid_df[grid_df['ror_id'].notna()].copy()
    log_step(f"  dim_org with ROR: {len(dim_with_ror):,}")
    log_step(f"  GRID with ROR: {len(grid_with_ror):,}")
    
    ror_positives = pd.merge(
        dim_with_ror[['unique_id', 'ror_id']],
        grid_with_ror[['unique_id', 'ror_id']],
        on='ror_id', suffixes=('_l', '_r')
    )[['unique_id_l', 'unique_id_r']]
    log_step(f"  ROR positive pairs: {len(ror_positives):,}")
    
    # Positive pairs from GRID matches
    dim_with_grid = dim_org_df[dim_org_df['grid_id'].notna()].copy()
    grid_with_grid = grid_df[grid_df['grid_id'].notna()].copy()
    log_step(f"  dim_org with GRID: {len(dim_with_grid):,}")
    log_step(f"  GRID with GRID: {len(grid_with_grid):,}")
    
    grid_positives = pd.merge(
        dim_with_grid[['unique_id', 'grid_id']],
        grid_with_grid[['unique_id', 'grid_id']],
        on='grid_id', suffixes=('_l', '_r')
    )[['unique_id_l', 'unique_id_r']]
    log_step(f"  GRID positive pairs: {len(grid_positives):,}")
    
    # Combine and deduplicate
    positive_df = pd.concat([ror_positives, grid_positives]).drop_duplicates()
    positive_df['is_match'] = True
    log_step(f"  Total positive pairs (deduplicated): {len(positive_df):,}")

    # Negative sampling - use records with either identifier
    dim_with_id = dim_org_df[(dim_org_df['ror_id'].notna()) | (dim_org_df['grid_id'].notna())].copy()
    grid_with_id = grid_df[(grid_df['ror_id'].notna()) | (grid_df['grid_id'].notna())].copy()
    
    sample_size = min(len(positive_df) * 2, 100_000)
    np.random.seed(42)
    dim_sample = dim_with_id[['unique_id', 'ror_id', 'grid_id']].sample(n=sample_size, replace=True).reset_index(drop=True)
    grid_sample = grid_with_id[['unique_id', 'ror_id', 'grid_id']].sample(n=sample_size, replace=True).reset_index(drop=True)

    negative_df = pd.DataFrame({
        'unique_id_l': dim_sample['unique_id'].values,
        'unique_id_r': grid_sample['unique_id'].values,
        'ror_id_l': dim_sample['ror_id'].values,
        'ror_id_r': grid_sample['ror_id'].values,
        'grid_id_l': dim_sample['grid_id'].values,
        'grid_id_r': grid_sample['grid_id'].values,
    })
    # Exclude if ROR matches OR GRID matches
    ror_match = (negative_df['ror_id_l'].notna() & (negative_df['ror_id_l'] == negative_df['ror_id_r']))
    grid_match = (negative_df['grid_id_l'].notna() & (negative_df['grid_id_l'] == negative_df['grid_id_r']))
    negative_df = negative_df[~(ror_match | grid_match)].copy()
    negative_df['is_match'] = False
    log_step(f"  Negative pairs: {len(negative_df):,}")

    return positive_df, negative_df


# =============================================================================
# HIERARCHY ROLL-UP
# =============================================================================

def rollup_to_parent(predictions_df: pd.DataFrame, hierarchy_df: pd.DataFrame, dim_org_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    log_step("Hierarchy roll-up: checking for subsidiary matches without location context...")
    if hierarchy_df is None or len(hierarchy_df) == 0:
        log_step("  No hierarchy data available - skipping roll-up", "WARN")
        return predictions_df, {'rolled_up': 0, 'total_checked': 0}

    predictions_df = predictions_df.copy()

    def normalize_org_id(org_id):
        if pd.isna(org_id):
            return None
        org_id = str(org_id)
        if not org_id.startswith('dim_'):
            return f'dim_{org_id}'
        return org_id

    child_to_parent = {}
    for _, row in hierarchy_df.iterrows():
        child_id = normalize_org_id(row.get('child_id'))
        parent_id = normalize_org_id(row.get('parent_id'))
        if child_id and parent_id:
            child_to_parent[child_id] = parent_id
    log_step(f"  Loaded {len(child_to_parent):,} child->parent mappings")

    dim_org_lookup = {}
    for _, row in dim_org_df.iterrows():
        uid = row.get('unique_id')
        if uid:
            dim_org_lookup[uid] = {
                'name': row.get('name'),
                'name_normalized': row.get('name_normalized'),
                'country_code': row.get('country_code'),
                'city': row.get('city'),
                'all_names': row.get('all_names', row.get('name_aliases', []))
            }

    location_missing = (
        (predictions_df['country_code_r'].isna() | (predictions_df['country_code_r'] == '')) &
        (predictions_df['city_r'].isna() | (predictions_df['city_r'] == ''))
    )
    total_missing_location = location_missing.sum()
    log_step(f"  Predictions with missing source location: {total_missing_location:,}")

    rolled_up_count = 0
    rollup_details = []

    for idx in predictions_df[location_missing].index:
        matched_id = predictions_df.loc[idx, 'unique_id_l']
        if matched_id in child_to_parent:
            parent_id = child_to_parent[matched_id]
            if parent_id in dim_org_lookup:
                parent_info = dim_org_lookup[parent_id]
                original_name = predictions_df.loc[idx, 'name_l']
                predictions_df.loc[idx, 'unique_id_l'] = parent_id
                predictions_df.loc[idx, 'name_l'] = parent_info['name']
                predictions_df.loc[idx, 'name_normalized_l'] = parent_info['name_normalized']
                predictions_df.loc[idx, 'country_code_l'] = parent_info['country_code']
                predictions_df.loc[idx, 'city_l'] = parent_info['city']
                if 'all_names_l' in predictions_df.columns:
                    predictions_df.at[idx, 'all_names_l'] = parent_info['all_names']
                predictions_df.loc[idx, 'rolled_up_from'] = matched_id
                predictions_df.loc[idx, 'rolled_up_from_name'] = original_name
                rolled_up_count += 1
                rollup_details.append({
                    'original_id': matched_id, 'original_name': original_name,
                    'parent_id': parent_id, 'parent_name': parent_info['name']
                })

    stats = {'total_checked': total_missing_location, 'rolled_up': rolled_up_count, 'rollup_details': rollup_details[:10]}
    log_step(f"  Rolled up {rolled_up_count:,} predictions to parent organizations")
    if rolled_up_count > 0 and len(rollup_details) > 0:
        log_step("  Sample rollups:")
        for detail in rollup_details[:3]:
            log_step(f"    {detail['original_name'][:40]} -> {detail['parent_name'][:40]}")

    return predictions_df, stats


# =============================================================================
# FULL PIPELINE
# =============================================================================

def load_and_prepare_training_data(db: DatabaseManager, dim_org_sample: int = None, grid_sample: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dim_org_sample = dim_org_sample or config.sampling.TRAINING_DIM_ORG_SAMPLE
    grid_sample = grid_sample or config.sampling.TRAINING_GRID_SAMPLE
    with Timer("Loading and preparing training data"):
        dim_org_raw = load_dim_org(db, dim_org_sample)
        grid_raw = load_grid(db, grid_sample)
        dim_org_df = create_unified_schema(dim_org_raw, 'dim_org')
        grid_df = create_unified_schema(grid_raw, 'grid')
        dim_org_df, _ = filter_bad_records(dim_org_df)
        grid_df, _ = filter_bad_records(grid_df)
        dim_org_df = create_blocking_keys(dim_org_df)
        grid_df = create_blocking_keys(grid_df)
    return dim_org_df, grid_df


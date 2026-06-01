"""Standalone persona generator based on US demographic statistics.

Data sources:
  - Gender, Race, Age: US Census (June 2023)
  - Religion: Statista
  - Political affiliation: Pew Research

Usage:
  python generate_personas_standalone.py --n 1000 --output personas_1000.json
  python generate_personas_standalone.py --n 500 --output personas_500.json \
      --personality personality.jsonl --profile profile.jsonl
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

PATH_TO_DEMOGRAPHIC_DATA = os.path.join(os.path.dirname(__file__), 'us_demographics')

RACES = ['White', 'Black', 'American Indian/Alaska Native',
         'Asian', 'Native Hawaiian/Pacific Islander', 'Hispanic']
GENDERS = ['Man', 'Woman', 'Nonbinary']


def get_gender_race_age_cdf() -> Tuple[List[Tuple], np.ndarray]:
    """Load joint (gender, race, age) CDF from US Census data."""
    fn = os.path.join(PATH_TO_DEMOGRAPHIC_DATA, 'nc-est2023-alldata-r-file07.csv')
    if not os.path.exists(fn):
        raise FileNotFoundError(
            f"Demographic data not found: {fn}\n"
            f"Ensure us_demographics/nc-est2023-alldata-r-file07.csv exists")

    df = pd.read_csv(fn)
    df = df[(df['MONTH'] == 6) & (df['YEAR'] == 2023)]
    assert len(df) == 102

    triplet2ct = {}
    prefixes = ['NHWA', 'NHBA', 'NHIA', 'NHAA', 'NHNA', 'H']
    postfixes = ['MALE', 'FEMALE']

    for _, row in df.iterrows():
        if row['AGE'] != 999:
            age = row['AGE']
            for pre, race in zip(prefixes, RACES):
                for post, gender in zip(postfixes, GENDERS[:-1]):
                    triplet2ct[(gender, race, age)] = row[f'{pre}_{post}']

    sorted_triplets = sorted(triplet2ct.keys(),
                             key=lambda x: triplet2ct[x], reverse=True)
    counts = [triplet2ct[t] for t in sorted_triplets]
    cdf = np.cumsum(np.array(counts) / np.sum(counts))
    cdf[-1] = 1.0

    print(f"Loaded demographic data: {len(sorted_triplets)} triplets")
    return sorted_triplets, cdf


def generate_persona(seed: int, sorted_triplets: List[Tuple],
                     cdf: np.ndarray) -> Dict[str, Any]:
    """Sample one persona following US population distributions."""
    np.random.seed(seed)
    person = {}

    # Gender, race, age — US Census
    triplet_rand = np.random.random()
    for triplet, cutoff in zip(sorted_triplets, cdf):
        if triplet_rand <= cutoff:
            gender, race, age = triplet
            person['gender'] = gender
            person['race/ethnicity'] = race
            person['age'] = age
            break

    # Nonbinary adjustment — Pew Research
    nonbinary = np.random.random()
    if person['age'] < 18 and nonbinary < 0.03:
        person['gender'] = 'Nonbinary'
    elif person['age'] < 49 and nonbinary < 0.013:
        person['gender'] = 'Nonbinary'
    elif nonbinary < 0.001:
        person['gender'] = 'Nonbinary'

    # Religion — Statista
    religion = np.random.random()
    if person['race/ethnicity'] == 'White':
        if religion < 0.49:
            person['religion'] = 'Protestant'
        elif religion < 0.69:
            person['religion'] = 'Catholic'
        elif religion < 0.71:
            person['religion'] = 'Jewish'
        elif religion < 0.72:
            person['religion'] = 'Buddhist'
        else:
            person['religion'] = 'Unreligious'
    elif person['race/ethnicity'] == 'Black':
        if religion < 0.68:
            person['religion'] = 'Protestant'
        elif religion < 0.75:
            person['religion'] = 'Catholic'
        elif religion < 0.77:
            person['religion'] = 'Muslim'
        else:
            person['religion'] = 'Unreligious'
    elif person['race/ethnicity'] == 'Hispanic':
        if religion < 0.26:
            person['religion'] = 'Protestant'
        elif religion < 0.76:
            person['religion'] = 'Catholic'
        else:
            person['religion'] = 'Unreligious'
    elif person['race/ethnicity'] in ['Asian', 'Native Hawaiian/Pacific Islander']:
        if religion < 0.16:
            person['religion'] = 'Protestant'
        elif religion < 0.30:
            person['religion'] = 'Catholic'
        elif religion < 0.37:
            person['religion'] = 'Muslim'
        elif religion < 0.44:
            person['religion'] = 'Buddhist'
        elif religion < 0.59:
            person['religion'] = 'Hindu'
        else:
            person['religion'] = 'Unreligious'
    else:
        if religion < 0.47:
            person['religion'] = 'Protestant'
        elif religion < 0.58:
            person['religion'] = 'Catholic'
        elif religion < 0.60:
            person['religion'] = 'Christian'
        else:
            person['religion'] = 'Unreligious'

    # Political affiliation — Pew Research
    politics = np.random.random()
    person['political affiliation'] = 'Independent'

    if person['race/ethnicity'] == 'White':
        if person['gender'] == 'Man':
            if politics < 0.6:
                person['political affiliation'] = 'Republican'
            elif politics < 0.99:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.53:
                person['political affiliation'] = 'Republican'
            elif politics < 0.96:
                person['political affiliation'] = 'Democrat'
    elif person['race/ethnicity'] == 'Black':
        if person['gender'] == 'Man':
            if politics < 0.15:
                person['political affiliation'] = 'Republican'
            elif politics < 0.96:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.10:
                person['political affiliation'] = 'Republican'
            elif politics < 0.94:
                person['political affiliation'] = 'Democrat'
    elif person['race/ethnicity'] == 'Hispanic':
        if person['gender'] == 'Man':
            if politics < 0.39:
                person['political affiliation'] = 'Republican'
            elif politics < 1:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.32:
                person['political affiliation'] = 'Republican'
            elif politics < 0.92:
                person['political affiliation'] = 'Democrat'
    elif person['race/ethnicity'] in ['Asian', 'Native Hawaiian/Pacific Islander']:
        if person['gender'] == 'Man':
            if politics < 0.39:
                person['political affiliation'] = 'Republican'
            elif politics < 1:
                person['political affiliation'] = 'Democrat'
        else:
            if politics < 0.36:
                person['political affiliation'] = 'Republican'
            elif politics < 1:
                person['political affiliation'] = 'Democrat'
    else:
        if politics < 0.4:
            person['political affiliation'] = 'Republican'
        elif politics < 0.96:
            person['political affiliation'] = 'Democrat'

    return person


def load_jsonl_strings(filepath: str) -> List[str]:
    """Load string list from a JSONL file."""
    strings = []
    decoder = json.JSONDecoder()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pos = 0
            while pos < len(line):
                while pos < len(line) and line[pos] in ' \t':
                    pos += 1
                if pos >= len(line):
                    break
                try:
                    obj, end = decoder.raw_decode(line, pos)
                    strings.append(obj)
                    pos = end
                except json.JSONDecodeError:
                    break
    print(f"Loaded {os.path.basename(filepath)}: {len(strings)} entries")
    return strings


def generate_all_personas(n: int, seed_offset: int = 0,
                          personality_file: Optional[str] = None,
                          profile_file: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Generate n synthetic personas based on US demographic distributions."""
    sorted_triplets, cdf = get_gender_race_age_cdf()

    personalities = None
    profiles = None
    if personality_file:
        personalities = load_jsonl_strings(personality_file)
    if profile_file:
        profiles = load_jsonl_strings(profile_file)

    personas = {}
    rng = np.random.RandomState(seed_offset + 99999)

    for i in range(n):
        persona = generate_persona(i + seed_offset, sorted_triplets, cdf)

        if personalities:
            persona['personality'] = personalities[rng.randint(0, len(personalities))]
        if profiles:
            persona['profile'] = profiles[rng.randint(0, len(profiles))]

        personas[str(i)] = persona

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  Progress: {i + 1}/{n} ({(i + 1) / n * 100:.1f}%)")

    return personas


def print_statistics(personas: Dict[str, Dict[str, Any]]) -> None:
    """Print demographic distribution statistics."""
    n = len(personas)
    gender_counts, race_counts = {}, {}
    religion_counts, political_counts = {}, {}
    age_ranges = {'0-17': 0, '18-34': 0, '35-49': 0, '50-64': 0, '65+': 0}

    for person in personas.values():
        gender_counts[person['gender']] = gender_counts.get(person['gender'], 0) + 1
        race_counts[person['race/ethnicity']] = race_counts.get(person['race/ethnicity'], 0) + 1
        religion_counts[person['religion']] = religion_counts.get(person['religion'], 0) + 1
        political_counts[person['political affiliation']] = \
            political_counts.get(person['political affiliation'], 0) + 1

        age = person['age']
        if age < 18:
            age_ranges['0-17'] += 1
        elif age < 35:
            age_ranges['18-34'] += 1
        elif age < 50:
            age_ranges['35-49'] += 1
        elif age < 65:
            age_ranges['50-64'] += 1
        else:
            age_ranges['65+'] += 1

    print(f"\nStatistics ({n} personas)")
    print("=" * 50)
    for label, counts in [("Gender", gender_counts), ("Race/Ethnicity", race_counts),
                           ("Age Range", age_ranges), ("Religion", religion_counts),
                           ("Political Affiliation", political_counts)]:
        print(f"\n[{label}]")
        items = sorted(counts.items(), key=lambda x: -x[1]) if isinstance(counts, dict) else counts.items()
        for k, c in items:
            print(f"  {k}: {c} ({c / n * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic personas based on US demographic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_personas_standalone.py --n 1000 --output personas_1000.json\n"
            "  python generate_personas_standalone.py --n 500 --output personas_500.json "
            "--personality personality.jsonl --profile profile.jsonl"
        ))
    parser.add_argument('--n', type=int, required=True,
                        help='Number of personas to generate')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed offset (default: 0)')
    parser.add_argument('--personality', type=str, default=None,
                        help='Path to personality.jsonl (optional)')
    parser.add_argument('--profile', type=str, default=None,
                        help='Path to profile.jsonl (optional)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip printing statistics')

    args = parser.parse_args()

    print(f"Generating {args.n} personas...")
    print(f"Output: {args.output}")
    print(f"Seed offset: {args.seed}")
    if args.personality:
        print(f"Personality file: {args.personality}")
    if args.profile:
        print(f"Profile file: {args.profile}")
    print()

    personas = generate_all_personas(args.n, seed_offset=args.seed,
                                     personality_file=args.personality,
                                     profile_file=args.profile)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(personas)} personas to {args.output}")

    if not args.no_stats:
        print_statistics(personas)


if __name__ == '__main__':
    main()

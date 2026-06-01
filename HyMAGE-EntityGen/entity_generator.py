"""Entity generator for labeled nodes used by hypergraph construction.

Modes:
  personas  - generate from US statistics (no LLM)
  drug      - LLM-synthesized OTC drug product entities
  reactant  - LLM-synthesized biological reactant profiles
  ecommerce - LLM-synthesized heterogeneous e-commerce entities
  general   - LLM-synthesized general-purpose entities

Output: JSON dict keyed by string IDs: {"0": {...}, "1": {...}}
"""

import os
import json
import argparse
import signal
import sys
import random
from typing import Dict, Any
from openai import OpenAI

_partial_data: Dict[str, Any] = {}
_output_path: str = ""
_interrupted: bool = False


def _save_partial_on_interrupt(signum, frame):
    global _interrupted
    _interrupted = True
    if _partial_data and _output_path:
        print(f"\nInterrupt detected, saving {len(_partial_data)} generated entries...")
        try:
            with open(_output_path, 'w', encoding='utf-8') as f:
                json.dump(_partial_data, f, ensure_ascii=False, indent=2)
            print(f"Saved to {_output_path}")
        except Exception as e:
            print(f"Save failed: {e}")
    sys.exit(0)


def load_api_key(api_key_file: str) -> str:
    candidate_paths = [
        api_key_file,
        os.path.join('Hypergraph-Generator', api_key_file),
        os.path.join(os.path.dirname(__file__), '..', api_key_file),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
    raise FileNotFoundError(f'API key file not found: {api_key_file}')


def load_personas(personas_path: str) -> Dict[str, Any]:
    with open(personas_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def call_llm(client: OpenAI, model: str, system_prompt: str,
             user_prompt: str, max_tokens: int = 800,
             temperature: float = 0.7) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def ensure_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('{') and part.endswith('}'):
                    return json.loads(part)
        if '{' in text and '}' in text:
            try:
                s = text[text.find('{'):text.rfind('}') + 1]
                return json.loads(s)
            except Exception:
                pass
        raise ValueError('LLM did not return valid JSON')


TEMPLATES: Dict[str, Dict[str, Any]] = {
    "drug": {
        "system": "You are a pharmacology and OTC drug expert. Produce realistic, detailed drug product entities in strict JSON format.",
        "fields": [
            "brand_name", "active_ingredient", "strength", "dosage_form",
            "drug_category", "mechanism_of_action", "common_side_effects",
            "route_of_administration", "manufacturer",
        ],
        "batch_fmt": "Generate {n} realistic OTC drug product profiles with detailed attributes.",
    },
    "reactant": {
        "system": "You are a biochemistry expert. Produce biological reactant profiles in strict JSON.",
        "fields": ["reactant_name", "reactant_type", "cellular_location", "key_pathways"],
        "batch_fmt": (
            "Generate {n} biological reactant profiles (metabolites or proteins). Return JSON id->object with: "
            "reactant_name, reactant_type, cellular_location, key_pathways (list of 1-3)."
        ),
    },
    "ecommerce": {
        "system": "You are an e-commerce data specialist. Produce heterogeneous profiles in strict JSON.",
        "fields": ["user_profile", "item_profile", "brand_profile", "category_profile"],
        "batch_fmt": (
            "Generate {n} entities as a JSON id->object. Each object contains: "
            "user_profile: {user_id, interests: [2-3]}, item_profile: {item_name, price_range}, "
            "brand_profile: {brand_name}, category_profile: {category_name}."
        ),
    },
    "general": {
        "system": "You are a data synthesizer. Produce coherent entities in strict JSON.",
        "fields": ["name", "attributes"],
        "batch_fmt": (
            "Generate {n} general-purpose entities. Return JSON id->object with fields: "
            "name (string), attributes (object of 3-6 key: value)."
        ),
    },
}


def build_single_prompt(entity_type: str) -> str:
    if entity_type == 'drug':
        return (
            "Generate exactly ONE OTC drug product entity in strict JSON with these keys:\n"
            "{\n"
            '  "brand_name": "string - common consumer brand name",\n'
            '  "active_ingredient": "string - main active compound",\n'
            '  "strength": "string - dosage/concentration",\n'
            '  "dosage_form": "string - one of: Tablet, Capsule, Liquid, Softgel, Ointment, Cream, Spray, Drops, Lozenge, Powder",\n'
            '  "drug_category": "string - one of the 7 categories below",\n'
            '  "mechanism_of_action": "string - how it works",\n'
            '  "common_side_effects": "string - 2-3 common side effects",\n'
            '  "route_of_administration": "string - one of: Oral, Topical, Nasal, Ophthalmic, Rectal",\n'
            '  "manufacturer": "string - company name"\n'
            "}\n\n"
            "drug_category must be exactly one of:\n"
            "1) Cold & Cough  2) Pain & Fever  3) Digestive Health  4) Skin Care\n"
            "5) Allergy Relief  6) Vitamins & Minerals  7) First Aid\n\n"
            "Rules:\n"
            "- Use common, widely available consumer brand names.\n"
            "- All fields are REQUIRED and must be non-empty strings.\n"
            "- Be diverse: vary categories, manufacturers, and dosage forms.\n"
            "Return ONLY the JSON object. No code fences, no explanations."
        )
    if entity_type == 'reactant':
        return (
            "Generate exactly ONE biological reactant profile as a JSON object with keys: "
            "reactant_name, reactant_type, cellular_location, key_pathways (array of 1-3 strings).\n"
            "Return ONLY the JSON object."
        )
    if entity_type == 'ecommerce':
        return (
            "Generate exactly ONE e-commerce heterogeneous entity as a JSON object with keys: "
            "user_profile (object: user_id, interests [2-3]), "
            "item_profile (object: item_name, price_range), "
            "brand_profile (object: brand_name), "
            "category_profile (object: category_name).\n"
            "Return ONLY the JSON object."
        )
    return (
        "Generate exactly ONE general-purpose entity as a JSON object with keys: "
        "name (string), attributes (object with 3-6 key:value pairs).\n"
        "Return ONLY the JSON object."
    )


def prompt_until_json(client: OpenAI, model: str, system_prompt: str,
                      user_prompt: str, retries: int = 3,
                      temperature: float = 0.9) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            temp = min(temperature + (attempt - 1) * 0.05, 1.2)
            text = call_llm(client, model, system_prompt, user_prompt,
                            max_tokens=400, temperature=temp)
            return ensure_json(text)
        except Exception as e:
            last_err = e
    raise last_err


def generate_entities(entity_type: str, n: int, model: str,
                      api_key_file: str, base_url: str,
                      personas_path: str = None,
                      output_path: str = "") -> Dict[str, Any]:
    global _partial_data, _output_path, _interrupted

    if entity_type == 'personas':
        if not personas_path:
            raise ValueError('personas_path is required when entity_type=personas')
        return load_personas(personas_path)

    if entity_type not in TEMPLATES:
        raise ValueError(f'Unsupported entity_type: {entity_type}')

    api_key = load_api_key(api_key_file)
    client = OpenAI(api_key=api_key, base_url=base_url)

    tpl = TEMPLATES[entity_type]
    system_prompt = tpl['system']
    single_prompt = build_single_prompt(entity_type)

    _output_path = output_path

    out: Dict[str, Any] = {}
    _partial_data = out
    seen_combinations = set()
    i = 0
    consecutive_failures = 0
    max_failures = 15
    last_duplicate = ""

    base_history_size = 8
    base_temperature = 0.95
    category_counts: Dict[str, int] = {}

    while i < n:
        try:
            current_temperature = min(base_temperature + consecutive_failures * 0.08, 1.3)

            all_samples = list(out.values())
            if consecutive_failures == 0:
                history_size = min(base_history_size, len(all_samples))
                recent_samples = all_samples[-history_size:] if history_size > 0 else []
            else:
                history_size = min(base_history_size + consecutive_failures * 2,
                                   len(all_samples), 20)
                if len(all_samples) > history_size:
                    recent_count = min(3, len(all_samples))
                    random_count = history_size - recent_count
                    recent_samples = all_samples[-recent_count:]
                    older_samples = all_samples[:-recent_count]
                    if older_samples and random_count > 0:
                        random_samples = random.sample(
                            older_samples, min(random_count, len(older_samples)))
                        recent_samples = random_samples + recent_samples
                else:
                    recent_samples = all_samples

            if entity_type == 'drug' and recent_samples:
                recent_text = json.dumps(recent_samples, ensure_ascii=False)
                category_info = ", ".join(
                    f"{cat}: {cnt}" for cat, cnt in category_counts.items())
                balance_hint = (
                    f"\nCurrent category distribution: {category_info}\n"
                    "Try to generate for underrepresented categories."
                ) if category_info else ""

                avoid_hint = ""
                if last_duplicate and consecutive_failures > 0:
                    avoid_hint = (
                        f"\nIMPORTANT: '{last_duplicate}' is a DUPLICATE. "
                        "Generate something COMPLETELY DIFFERENT."
                    )

                user_prompt = (
                    f"{single_prompt}\n\n"
                    f"Previously generated (for context, generate something NEW):\n"
                    f"{recent_text}{balance_hint}{avoid_hint}\n"
                )
            else:
                user_prompt = single_prompt

            obj = prompt_until_json(client, model, system_prompt, user_prompt,
                                    retries=3, temperature=current_temperature)

            if entity_type == 'drug':
                required_fields = [
                    'brand_name', 'active_ingredient', 'strength', 'dosage_form',
                    'drug_category', 'mechanism_of_action', 'common_side_effects',
                    'route_of_administration', 'manufacturer'
                ]
                missing = [f for f in required_fields
                           if not str(obj.get(f, '')).strip()]
                if missing:
                    raise ValueError(f"Missing fields: {', '.join(missing)}")

                brand_name = str(obj.get('brand_name', '')).strip()
                active_ingredient = str(obj.get('active_ingredient', '')).strip()
                strength = str(obj.get('strength', '')).strip()
                dosage_form = str(obj.get('dosage_form', '')).strip()
                drug_category = str(obj.get('drug_category', '')).strip()
                manufacturer = str(obj.get('manufacturer', '')).strip()

                combo_key = (brand_name.lower(), active_ingredient.lower(),
                             strength.lower())
                if combo_key in seen_combinations:
                    raise ValueError(
                        f"Exact duplicate: {brand_name} + {active_ingredient} + {strength}")
                seen_combinations.add(combo_key)

                if drug_category:
                    category_counts[drug_category] = \
                        category_counts.get(drug_category, 0) + 1

            if not obj or (isinstance(obj, dict) and len(obj) == 0):
                raise ValueError("Generated empty object")

            out[str(i)] = obj
            _partial_data.update(out)
            i += 1
            consecutive_failures = 0
            last_duplicate = ""

            if entity_type == 'drug':
                print(f"[{i}/{n}] {brand_name} | {strength} {dosage_form} "
                      f"| {drug_category} | {manufacturer}")
            else:
                print(f"[{i}/{n}] Generated successfully")

            if _interrupted:
                break
        except Exception as e:
            consecutive_failures += 1
            error_msg = str(e)

            if "Exact duplicate:" in error_msg:
                last_duplicate = error_msg.replace("Exact duplicate: ", "")

            if consecutive_failures <= 3 or consecutive_failures % 3 == 0:
                print(f"[{i + 1}/{n}] Failed ({consecutive_failures}): "
                      f"{error_msg[:60]}{'...' if len(error_msg) > 60 else ''}")

            if entity_type != 'drug':
                if out:
                    out[str(i)] = json.loads(
                        json.dumps(list(out.values())[-1]))
                else:
                    out[str(i)] = {}
                i += 1
                consecutive_failures = 0
            elif consecutive_failures >= max_failures:
                print(f"[{i + 1}/{n}] Skipping after {max_failures} consecutive failures")
                i += 1
                consecutive_failures = 0
                last_duplicate = ""

    if entity_type == 'drug':
        valid_out = {
            k: v for k, v in out.items()
            if v and isinstance(v, dict)
            and v.get('brand_name') and v.get('active_ingredient')
        }
        out = valid_out
        print(f"\nValid records: {len(out)}")

    return out


def main():
    signal.signal(signal.SIGINT, _save_partial_on_interrupt)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _save_partial_on_interrupt)

    parser = argparse.ArgumentParser(
        description='Generate labeled entities for hypergraph nodes')
    parser.add_argument('--entity_type', type=str, required=True,
                        help='personas | drug | reactant | ecommerce | general')
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of entities to generate')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        help='LLM model (when entity_type != personas)')
    parser.add_argument('--api_key_file', type=str, default='api-key.txt')
    parser.add_argument('--base_url', type=str,
                        default='https://api.openai.com/v1')
    parser.add_argument('--personas_path', type=str, default=None,
                        help='Path to personas JSON (when entity_type=personas)')
    parser.add_argument('--output', type=str, default='entities.json')
    args = parser.parse_args()

    print(f"Generating {args.n} {args.entity_type} entities...")
    print(f"Output: {args.output}")
    print(f"Press Ctrl+C to interrupt (partial data will be saved)\n")

    try:
        data = generate_entities(
            args.entity_type, args.n, args.model,
            args.api_key_file, args.base_url,
            args.personas_path, args.output)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'\nDone. Saved {len(data)} entities to {args.output}')
    except KeyboardInterrupt:
        if _partial_data:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(_partial_data, f, ensure_ascii=False, indent=2)
            print(f'\nInterrupted. Saved {len(_partial_data)} entities to {args.output}')


if __name__ == '__main__':
    main()

"""Augment existing personas with academic attributes using LLM.

For each persona, generates:
  - education_level: highest degree (e.g., Bachelor, Master, PhD)
  - school: institution name
  - major: academic major
  - research_direction: specific research focus

Usage:
  python augment_personas_algebra.py --input personas.json --output personas_augmented.json
"""

import argparse
import json
from typing import Dict, Any

from openai import OpenAI
from entity_generator import load_api_key, prompt_until_json

SYSTEM_PROMPT = (
    "You are an expert in academic biographies and research careers. "
    "You will receive a short persona profile of a person (gender, race/ethnicity, age, "
    "religion, political affiliation, etc.). "
    "Based on this profile, you must design a plausible academic background and research focus. "
    "All values must be realistic and internally consistent. "
    "Use only English for all field values."
)


def build_user_prompt(persona_id: str, persona: Dict[str, Any]) -> str:
    persona_json = json.dumps(persona, ensure_ascii=False, indent=2)
    return (
        "Here is a persona profile (JSON):\n"
        f"{persona_json}\n\n"
        "Task:\n"
        "- Design one plausible academic background and research focus for this person.\n"
        "- The result should be coherent with the persona's age, religion, political affiliation, etc.\n"
        "- Prefer STEM or mathematics-related majors when reasonable, especially fields related to "
        "algebra, data science, or networks, but choose other majors if more natural.\n"
        "- The research_direction must be a specific topic naturally based on the major.\n"
        "- Use realistic university or college names.\n\n"
        "Return STRICT JSON with exactly these four keys:\n"
        "{\n"
        '  "education_level": "string",\n'
        '  "school": "string",\n'
        '  "major": "string",\n'
        '  "research_direction": "string"\n'
        "}\n\n"
        "Rules:\n"
        "- Output ONLY the JSON object.\n"
        "- No explanations, comments, or code fences.\n"
        "- All values must be in English.\n"
    )


def augment_personas_with_academia(input_path: str, output_path: str,
                                   model: str, api_key_file: str,
                                   base_url: str,
                                   max_retries: int = 3) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        personas: Dict[str, Any] = json.load(f)

    api_key = load_api_key(api_key_file)
    client = OpenAI(api_key=api_key, base_url=base_url)

    enhanced: Dict[str, Any] = {}
    ids = sorted(personas.keys(), key=lambda x: int(x))

    for idx, pid in enumerate(ids, start=1):
        persona = personas[pid]
        print(f"[{idx}/{len(ids)}] Augmenting persona id={pid}...")

        user_prompt = build_user_prompt(pid, persona)

        try:
            edu_info = prompt_until_json(client, model, SYSTEM_PROMPT,
                                         user_prompt, retries=max_retries,
                                         temperature=0.9)
            education_level = str(edu_info.get("education_level", "")).strip() or "Unknown"
            school = str(edu_info.get("school", "")).strip() or "Unknown University"
            major = str(edu_info.get("major", "")).strip() or "Undeclared"
            research_direction = (
                str(edu_info.get("research_direction", "")).strip() or "General research")

            persona_aug = dict(persona)
            persona_aug.update({
                "education_level": education_level,
                "school": school,
                "major": major,
                "research_direction": research_direction,
            })
            enhanced[pid] = persona_aug
            print(f"  -> {education_level} @ {school} | {major} | "
                  f"{research_direction[:60]}...")
        except Exception as e:
            print(f"  -> LLM failed for id={pid}: {e}")
            persona_aug = dict(persona)
            persona_aug.update({
                "education_level": "Unknown",
                "school": "Unknown University",
                "major": "Undeclared",
                "research_direction": "General research",
            })
            enhanced[pid] = persona_aug

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Augmented {len(enhanced)} personas -> {output_path}")
    return enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Augment personas with academic attributes via LLM")
    parser.add_argument("--input", type=str, required=True,
                        help="Input personas JSON path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path with augmented fields")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="LLM model for augmentation")
    parser.add_argument("--api_key_file", type=str, default="api-key.txt")
    parser.add_argument("--base_url", type=str,
                        default="https://api.openai.com/v1")
    args = parser.parse_args()

    augment_personas_with_academia(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        api_key_file=args.api_key_file,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()

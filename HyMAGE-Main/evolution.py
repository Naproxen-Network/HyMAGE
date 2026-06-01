"""Entity adapter and continuous evolution controller for SPA hypergraph generation."""

import math
from typing import List, Dict, Any


class UniversalEntityAdapter:
    """Universal entity adapter supporting arbitrary JSON entity formats."""

    def __init__(self, entities: Dict[str, Any]):
        self.entities = entities
        self.entity_ids = list(entities.keys())
        self.attribute_keys = self._detect_attributes()
        self.entity_type = self._detect_entity_type()

    def _detect_attributes(self) -> List[str]:
        if not self.entities:
            return []
        sample = next(iter(self.entities.values()))
        return list(sample.keys()) if isinstance(sample, dict) else []

    def _detect_entity_type(self) -> str:
        keys = set(self.attribute_keys)
        if "gender" in keys and "race/ethnicity" in keys:
            return "personas"
        elif "drug_name" in keys:
            return "drug"
        elif "party" in keys and "state" in keys:
            return "congress"
        elif "expertise" in keys:
            return "stackexchange"
        elif "interests" in keys:
            return "social"
        return "general"

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        return self.entities.get(str(entity_id), {})

    def format_entity_for_prompt(self, entity_id: str,
                                 include_degree: bool = False,
                                 degree: int = 0) -> str:
        entity = self.get_entity(entity_id)
        if not entity:
            return f"ID {entity_id}: (unknown entity)"

        parts = [f"ID {entity_id}"]
        if include_degree:
            vis = "high" if degree >= 3 else ("mid" if degree >= 1 else "low")
            parts.append(f"[visibility:{vis}, degree:{degree}]")

        attr_strs = []
        for key in self.attribute_keys:
            if key in entity:
                value = entity[key]
                if isinstance(value, dict):
                    nested = ", ".join(f"{k}:{v}" for k, v in list(value.items())[:3])
                    attr_strs.append(f"{key}={{{nested}}}")
                elif isinstance(value, list):
                    if len(value) <= 3:
                        attr_strs.append(f"{key}:{value}")
                    else:
                        attr_strs.append(f"{key}:[{', '.join(str(v) for v in value[:3])}...]")
                else:
                    attr_strs.append(f"{key}:{value}")
        parts.append(" - " + ", ".join(attr_strs))
        return "".join(parts)


class ContinuousEvolutionController:
    """Controls the add/delete edge probabilities using sigmoid functions.

    P_add(p) = alpha_add * sigmoid(-k_add * (p - theta_add)) + beta_add
    P_del(p) = alpha_del * sigmoid( k_del * (p - theta_del)) + beta_del
    """

    def __init__(self, target_edges: int, target_distribution: Dict[int, int]):
        self.target_edges = target_edges
        self.target_distribution = target_distribution

        # Add-edge probability parameters
        self.add_amplitude = 0.7
        self.add_baseline = 0.2
        self.add_steepness = 5.0
        self.add_midpoint = 0.6

        # Delete-edge probability parameters
        self.del_amplitude = 0.6
        self.del_baseline = 0.05
        self.del_steepness = 6.0
        self.del_midpoint = 0.4

        self.max_add_per_iter = 10
        self.max_del_per_iter = 8

    @staticmethod
    def sigmoid(x: float) -> float:
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def compute_add_probability(self, progress: float) -> float:
        sig = self.sigmoid(-self.add_steepness * (progress - self.add_midpoint))
        return self.add_amplitude * sig + self.add_baseline

    def compute_del_probability(self, progress: float) -> float:
        sig = self.sigmoid(self.del_steepness * (progress - self.del_midpoint))
        return self.del_amplitude * sig + self.del_baseline

    def get_evolution_params(self, current_edges: int,
                             current_distribution: Dict[int, int]) -> Dict[str, Any]:
        progress = current_edges / self.target_edges if self.target_edges > 0 else 0
        p_add = self.compute_add_probability(progress)
        p_del = self.compute_del_probability(progress)

        add_count = max(1, int(self.max_add_per_iter * p_add))
        del_count = max(0, int(self.max_del_per_iter * p_del))

        gap = self._calculate_distribution_gap(current_distribution)
        total_gap = sum(gap.values())

        if total_gap > self.target_edges * 0.2:
            add_count = min(add_count + 2, self.max_add_per_iter)
            del_count = max(0, del_count - 1)
        elif total_gap < -self.target_edges * 0.1:
            add_count = max(1, add_count - 1)
            del_count = min(del_count + 1, self.max_del_per_iter)

        return {
            "progress": progress,
            "p_add": p_add,
            "p_del": p_del,
            "add_count": add_count,
            "del_count": del_count,
            "priority_sizes": self._get_priority_sizes(current_distribution),
            "distribution_gap": gap,
        }

    def _calculate_distribution_gap(self, current_distribution: Dict[int, int]) -> Dict[int, int]:
        gap = {}
        all_sizes = set(self.target_distribution.keys()) | set(current_distribution.keys())
        for size in all_sizes:
            gap[size] = self.target_distribution.get(size, 0) - current_distribution.get(size, 0)
        return gap

    def _get_priority_sizes(self, current_distribution: Dict[int, int]) -> List[int]:
        gap = self._calculate_distribution_gap(current_distribution)
        return [s for s, g in sorted(gap.items(), key=lambda x: x[1], reverse=True) if g > 0]

    def get_status_description(self, progress: float) -> str:
        if progress < 0.3:
            return "Rapid Growth"
        elif progress < 0.6:
            return "Steady Growth"
        elif progress < 0.9:
            return "Dynamic Balance"
        elif progress < 1.05:
            return "Convergence"
        return "Stable"

"""LLM-based agents for semantic preferential attachment hypergraph generation."""

import re
import random
from typing import List, Dict, Any, Tuple, Optional

from evolution import UniversalEntityAdapter


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, agent_id: str, model: str = "gpt-3.5-turbo", llm_client=None):
        self.agent_id = agent_id
        self.model = model
        self.llm_client = llm_client
        self.decision_history: list = []

    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class EdgeGeneratorAgent(BaseAgent):
    """Agent that initiates new hyperedge formation via semantic PA mechanism.

    PA provides visibility (high-degree nodes are more likely to be seen),
    while the LLM performs semantic decision-making based on node attributes.
    """

    def __init__(self, agent_id: str, model: str,
                 entity_adapter: UniversalEntityAdapter,
                 no_analysis: bool = False,
                 semantic_description: str = None,
                 candidate_pool_size: int = 15,
                 llm_client=None):
        super().__init__(agent_id, model, llm_client)
        self.entity_adapter = entity_adapter
        self.api_call_count = 0
        self.api_success_count = 0
        self.no_analysis = no_analysis
        self.semantic_description = semantic_description
        self.candidate_pool_size = candidate_pool_size

    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        initiator_id = context["initiator_id"]
        target_size = context["target_edge_size"]

        node_degrees = self._calculate_node_degrees(context["existing_hyperedges"])
        initiator_degree = node_degrees.get(initiator_id, 0)
        candidates_text = self._generate_candidate_pool(
            initiator_id, node_degrees, self.candidate_pool_size
        )
        initiator_info = self.entity_adapter.format_entity_for_prompt(
            initiator_id, include_degree=True, degree=initiator_degree
        )

        output_fmt = "Selection: [ID1] [ID2] ..."
        if not self.no_analysis:
            output_fmt += "\nReason: [brief explanation]"

        semantic_ctx = ""
        if self.semantic_description:
            semantic_ctx = f"[Network Semantics]\n{self.semantic_description}\n\n"

        prompt = (
            f"You are node {initiator_id} in a network and want to form a new group.\n\n"
            f"{semantic_ctx}"
            f"[Your Info]\n{initiator_info}\n\n"
            f"[Candidate Nodes] (sorted by visibility via preferential attachment)\n"
            f"{candidates_text}\n\n"
            f"[Task]\nSelect {target_size - 1} nodes to form a group of size {target_size} with you.\n\n"
            f"[Principles]\n"
            f"1. Capital: high-degree nodes may bring more influence\n"
            f"2. Compatibility: attribute similarity/complementarity matters\n\n"
            f"[Output Format]\n{output_fmt}"
        )

        self.api_call_count += 1
        api_called = False

        try:
            max_tokens = 60 if self.no_analysis else 150
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an autonomous node forming a new group in a network."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            api_called = True
            output = response.choices[0].message.content.strip()
            if not output:
                raise ValueError("Empty API response")

            selected_ids = self._parse_selection(output, initiator_id)
            needed = target_size - 1
            if len(selected_ids) < needed:
                fallback = self._fallback_selection(initiator_id, node_degrees, needed - len(selected_ids))
                selected_ids.extend(fallback)
                seen = set()
                selected_ids = [s for s in selected_ids if s != initiator_id and not (s in seen or seen.add(s))]
                selected_ids = selected_ids[:needed]

            self.api_success_count += 1
            decision = {
                "action": "add_edge",
                "agent_id": self.agent_id,
                "initiator_id": initiator_id,
                "selected_members": [initiator_id] + selected_ids,
                "target_size": target_size,
                "reasoning": output,
                "api_called": True,
            }
            self.decision_history.append(decision)
            return decision

        except Exception as e:
            if not api_called:
                raise RuntimeError(f"Edge-add LLM API call failed: {e}")
            fallback_ids = self._fallback_selection(initiator_id, node_degrees, target_size - 1)
            return {
                "action": "add_edge",
                "agent_id": self.agent_id,
                "initiator_id": initiator_id,
                "selected_members": [initiator_id] + fallback_ids,
                "target_size": target_size,
                "reasoning": f"API error ({e}), PA fallback used",
                "api_called": api_called,
                "api_error": str(e),
            }

    # ------------------------------------------------------------------
    def _calculate_node_degrees(self, hyperedges: List[List[str]]) -> Dict[str, int]:
        deg: Dict[str, int] = {}
        for he in hyperedges:
            for n in he:
                deg[n] = deg.get(n, 0) + 1
        return deg

    def _generate_candidate_pool(self, initiator_id: str,
                                 node_degrees: Dict[str, int],
                                 pool_size: int) -> str:
        candidates = [{"id": eid, "degree": node_degrees.get(eid, 0)}
                      for eid in self.entity_adapter.entity_ids if eid != initiator_id]

        if len(candidates) <= pool_size:
            selected = candidates
        else:
            weights = [c["degree"] + 1 for c in candidates]
            sampled = random.choices(candidates, weights=weights, k=pool_size)
            seen = set()
            selected = [c for c in sampled if not (c["id"] in seen or seen.add(c["id"]))]
            selected = selected[:pool_size]

        selected.sort(key=lambda x: x["degree"], reverse=True)
        lines = []
        for i, c in enumerate(selected, 1):
            fmt = self.entity_adapter.format_entity_for_prompt(
                c["id"], include_degree=True, degree=c["degree"])
            lines.append(f"{i}. {fmt}")
        return "\n".join(lines)

    def _parse_selection(self, output: str, initiator_id: str) -> List[str]:
        selected: List[str] = []
        valid_ids = set(self.entity_adapter.entity_ids)
        for line in output.split("\n"):
            if "select" in line.lower() or "ID" in line.upper():
                for id_str in re.findall(r"\b(\d+)\b", line):
                    if id_str in valid_ids and id_str != initiator_id:
                        selected.append(id_str)
        if not selected:
            for id_str in re.findall(r"\b(\d+)\b", output):
                if id_str in valid_ids and id_str != initiator_id and id_str not in selected:
                    selected.append(id_str)
        return selected

    def _fallback_selection(self, initiator_id: str,
                            node_degrees: Dict[str, int],
                            count: int) -> List[str]:
        candidates = [(eid, node_degrees.get(eid, 0) + 1)
                      for eid in self.entity_adapter.entity_ids if eid != initiator_id]
        if not candidates:
            return []
        weights = [w for _, w in candidates]
        ids = [eid for eid, _ in candidates]
        return list(set(random.choices(ids, weights=weights, k=count)))[:count]


class EdgeDissolutionAgent(BaseAgent):
    """Agent that evaluates and potentially dissolves hyperedges.

    A node assesses the groups it belongs to and may initiate dissolution
    if it finds a group unsatisfactory.
    """

    def __init__(self, agent_id: str, model: str,
                 entity_adapter: UniversalEntityAdapter,
                 no_analysis: bool = False,
                 semantic_description: str = None,
                 llm_client=None):
        super().__init__(agent_id, model, llm_client)
        self.entity_adapter = entity_adapter
        self.api_call_count = 0
        self.api_success_count = 0
        self.no_analysis = no_analysis
        self.semantic_description = semantic_description

    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        initiator_id = context["initiator_id"]
        candidate_edges = context["candidate_edges"]

        if not candidate_edges:
            return {
                "action": "dissolve_edge",
                "agent_id": self.agent_id,
                "initiator_id": initiator_id,
                "decision": "KEEP_ALL",
                "edge_to_dissolve": None,
                "reasoning": "No candidate edges",
                "api_called": False,
                "skip_reason": "no_candidate_edges",
            }

        initiator_info = self.entity_adapter.format_entity_for_prompt(initiator_id)
        edges_info = self._format_candidate_edges(candidate_edges)

        output_fmt = "Decision: DISSOLVE [group#] or KEEP_ALL"
        if not self.no_analysis:
            output_fmt += "\nReason: [brief explanation]"

        semantic_ctx = ""
        if self.semantic_description:
            semantic_ctx = f"[Network Semantics]\n{self.semantic_description}\n\n"

        prompt = (
            f"You are node {initiator_id} evaluating your group memberships.\n\n"
            f"{semantic_ctx}"
            f"[Your Info]\n{initiator_info}\n\n"
            f"[Your Groups]\n{edges_info}\n\n"
            f"[Task]\nDecide whether to dissolve any one group.\n\n"
            f"[Criteria]\n"
            f"1. Belonging: do you share commonalities with other members?\n"
            f"2. Value: is this group valuable to you?\n"
            f"3. Conflict: are there severe value or attribute conflicts?\n\n"
            f"[Output Format]\n{output_fmt}"
        )

        self.api_call_count += 1
        api_called = False

        try:
            max_tokens = 30 if self.no_analysis else 100
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an autonomous node evaluating group membership."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
            api_called = True
            output = response.choices[0].message.content.strip()
            if not output:
                raise ValueError("Empty API response")

            edge_to_dissolve = None
            decision = "KEEP_ALL"
            if "DISSOLVE" in output.upper():
                decision = "DISSOLVE"
                for num in re.findall(r"\d+", output):
                    idx = int(num)
                    if 1 <= idx <= len(candidate_edges):
                        edge_to_dissolve = candidate_edges[idx - 1][0]
                        break

            self.api_success_count += 1
            result = {
                "action": "dissolve_edge",
                "agent_id": self.agent_id,
                "initiator_id": initiator_id,
                "decision": decision,
                "edge_to_dissolve": edge_to_dissolve,
                "reasoning": output,
                "api_called": True,
            }
            self.decision_history.append(result)
            return result

        except Exception as e:
            if not api_called:
                raise RuntimeError(f"Edge-dissolve LLM API call failed: {e}")
            return {
                "action": "dissolve_edge",
                "agent_id": self.agent_id,
                "initiator_id": initiator_id,
                "decision": "KEEP_ALL",
                "edge_to_dissolve": None,
                "reasoning": f"API error ({e}), conservative keep-all",
                "api_called": api_called,
                "api_error": str(e),
            }

    def _format_candidate_edges(self, candidate_edges: List[Tuple[int, List[str]]]) -> str:
        lines = []
        for i, (edge_idx, members) in enumerate(candidate_edges, 1):
            member_info = [f"    - {self.entity_adapter.format_entity_for_prompt(m)}"
                           for m in members[:5]]
            if len(members) > 5:
                member_info.append(f"    - ...and {len(members) - 5} more")
            lines.append(f"Group {i} ({len(members)} members):")
            lines.extend(member_info)
        return "\n".join(lines)

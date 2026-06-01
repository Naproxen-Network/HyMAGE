"""Semantic Preferential Attachment (SPA) Hypergraph Generator.

Generates hypergraphs where nodes autonomously add and dissolve hyperedges
via LLM-based semantic decision-making combined with preferential attachment.
"""

import json
import argparse
import os
import random
import time
import collections
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils import get_llm_client, resolve_model_name
from evolution import UniversalEntityAdapter, ContinuousEvolutionController
from agents import EdgeGeneratorAgent, EdgeDissolutionAgent

# Optional: hyperedge label prediction module
try:
    from LLM_HyperedgeLabeler import HyperedgeLabeler, LABELED_DATASETS  # type: ignore[reportMissingImports]
except ImportError:
    HyperedgeLabeler = None
    LABELED_DATASETS = {}


# Semantic presets for common dataset types
SEMANTIC_PRESETS = {
    "email": "An email communication network. Nodes are email addresses, and hyperedges contain the sender and all recipients of an email.",
    "coauth-cs": "A co-authorship network in computer science. Nodes represent authors, and hyperedges represent joint publications.",
    "bars": "A Yelp-based social network where nodes represent Yelp users, and hyperedges denote groups of users who have viewed specific categories of bars over the course of a month.",
    "geometry": "Derived from MathOverflow. Nodes represent users, and hyperedges consist of users responding to question categories related to various geometry labels.",
    "music": "Sourced from Amazon reviews. Nodes represent reviewers, and hyperedges consist of reviewers who evaluated a product category within a one-month period.",
    "restaurant": "A Yelp-based social network where nodes represent Yelp users, and hyperedges represent groups of users who have viewed certain categories of restaurants over the course of a month.",
    "algebra": "Derived from MathOverflow. Nodes are users, and hyperedges are sets of users who answered questions in the algebra category.",
    "coauth-geology": "A large-scale co-authorship network in geology. Nodes represent authors, and hyperedges represent joint publications.",
    "dawn": "A drug abuse warning network where nodes are drugs and hyperedges are sets of drugs used by a patient.",
    "ndc-classes": "A drug classification network where nodes are drug class labels and hyperedges are sets of labels associated with a single drug.",
    "tags": "Online forum data where nodes are tags and hyperedges are sets of tags applied to a single question.",
    "threads": "Online forum data where nodes are users and hyperedges are sets of users participating in a discussion thread.",
}


class SemanticPAHypergraphGenerator:
    """Orchestrates hypergraph generation using semantic PA mechanism."""

    def __init__(self, entities_file: str, config_hypergraph_file: str,
                 output_path: str, max_iterations: int = 100,
                 model: str = "gpt-3.5-turbo", pa_probability: float = 0.7,
                 no_analysis: bool = False, semantic_description: str = None,
                 labeled_dataset: str = None, base_url: str = "https://api.openai.com/v1",
                 candidate_pool_size: int = 15,
                 local_endpoints: Optional[dict] = None):
        self.entities_file = entities_file
        self.config_hypergraph_file = config_hypergraph_file
        self.labeled_dataset = labeled_dataset
        self.max_iterations = max_iterations
        self.model = model
        self.pa_probability = pa_probability
        self.no_analysis = no_analysis
        self.semantic_description = semantic_description
        self.candidate_pool_size = candidate_pool_size

        # Build run directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = os.path.splitext(os.path.basename(config_hypergraph_file))[0]
        pa_str = f"PA{int(pa_probability * 100)}"
        self.run_dir = f"SPA_Run_{config_name}_{pa_str}_{self.run_timestamp}"

        if output_path.endswith(".txt"):
            self.output_file = output_path
            base_dir = os.path.dirname(output_path) or "."
            self.protected_run_dir = os.path.join(base_dir, self.run_dir)
        else:
            self.protected_run_dir = os.path.join(output_path, self.run_dir)
            self.output_file = os.path.join(self.protected_run_dir, "final_hypergraph.txt")

        os.makedirs(self.protected_run_dir, exist_ok=True)
        self.snapshots_dir = os.path.join(self.protected_run_dir, "snapshots")
        self.checkpoints_dir = os.path.join(self.protected_run_dir, "checkpoints")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Load entities
        self.entities = self._load_entities()
        self.entity_adapter = UniversalEntityAdapter(self.entities)

        # Analyse target distribution
        self.edge_size_distribution = self._analyze_edge_size_distribution()
        self.total_target_edges = sum(self.edge_size_distribution.values())
        self.edge_size_sequence = self._generate_edge_size_sequence()

        # Evolution controller
        self.evolution_controller = ContinuousEvolutionController(
            self.total_target_edges, self.edge_size_distribution
        )

        # Hypergraph state
        self.hyperedges: List[List[str]] = []
        self.current_edge_index = 0

        # LLM client
        self.llm_client = get_llm_client(model, base_url=base_url,
                                         local_endpoints=local_endpoints)
        self.actual_model = resolve_model_name(model)

        # Agents
        self.agents = {
            "generator": EdgeGeneratorAgent(
                "edge_generator", self.actual_model, self.entity_adapter,
                self.no_analysis, self.semantic_description,
                self.candidate_pool_size, self.llm_client,
            ),
            "dissolver": EdgeDissolutionAgent(
                "edge_dissolver", self.actual_model, self.entity_adapter,
                self.no_analysis, self.semantic_description, self.llm_client,
            ),
        }

        # Optional label predictor
        self.hyperedge_labeler = None
        if self.labeled_dataset and HyperedgeLabeler is not None:
            config_dir = os.path.dirname(os.path.abspath(config_hypergraph_file))
            labels_file = os.path.join(config_dir, "hyperedge-labels.txt")
            config_labels = labels_file if os.path.exists(labels_file) else None
            self.hyperedge_labeler = HyperedgeLabeler(
                dataset_type=self.labeled_dataset,
                entity_adapter=self.entity_adapter,
                llm_client=self.llm_client,
                model=self.actual_model,
                no_analysis=self.no_analysis,
                config_labels_file=config_labels,
            )

        # Logs
        self.evolution_history: list = []
        self.decision_log: list = []

        self._save_run_configuration()
        self._print_banner()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _print_banner(self):
        print(f"\n{'=' * 60}")
        print("Semantic PA Hypergraph Generator")
        print(f"{'=' * 60}")
        print(f"  Run directory : {self.protected_run_dir}")
        print(f"  Entities      : {len(self.entities)}")
        print(f"  Target edges  : {self.total_target_edges}")
        print(f"  PA probability: {self.pa_probability:.0%}")
        print(f"  Candidate pool: {self.candidate_pool_size}")
        print(f"  Model         : {self.model}")
        if self.semantic_description:
            print(f"  Semantics     : {self.semantic_description[:80]}...")
        if self.labeled_dataset:
            print(f"  Labels        : {self.labeled_dataset}")
        print(f"{'=' * 60}\n")

    def _load_entities(self) -> Dict[str, Any]:
        paths = [
            self.entities_file,
            os.path.join(os.path.dirname(__file__), self.entities_file),
        ]
        for path in paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {str(k): v for k, v in data.items()}
        raise FileNotFoundError(f"Entity file not found: {self.entities_file}")

    def _analyze_edge_size_distribution(self) -> Dict[int, int]:
        paths = [
            self.config_hypergraph_file,
            os.path.join(os.path.dirname(__file__), self.config_hypergraph_file),
        ]
        for path in paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    lines = f.readlines()
                sizes = [len(line.strip().split()) for line in lines if line.strip()]
                return dict(collections.Counter(sizes))
        raise FileNotFoundError(f"Config file not found: {self.config_hypergraph_file}")

    def _generate_edge_size_sequence(self) -> List[int]:
        seq = []
        for size, count in self.edge_size_distribution.items():
            seq.extend([size] * count)
        random.shuffle(seq)
        return seq

    def _save_run_configuration(self):
        config = {
            "method": "Semantic Preferential Attachment (SPA)",
            "entities_file": self.entities_file,
            "config_hypergraph_file": self.config_hypergraph_file,
            "entity_type": self.entity_adapter.entity_type,
            "max_iterations": self.max_iterations,
            "model": self.model,
            "pa_probability": self.pa_probability,
            "candidate_pool_size": self.candidate_pool_size,
            "total_target_edges": self.total_target_edges,
            "edge_size_distribution": {str(k): v for k, v in self.edge_size_distribution.items()},
            "start_time": datetime.now().isoformat(),
        }
        path = os.path.join(self.protected_run_dir, "run_configuration.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _save_checkpoint(self, iteration: int):
        ckpt = {
            "iteration": iteration,
            "current_edge_index": self.current_edge_index,
            "hyperedges": self.hyperedges,
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.checkpoints_dir, f"checkpoint_{iteration:03d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    def _get_current_distribution(self) -> Dict[int, int]:
        return dict(collections.Counter(len(e) for e in self.hyperedges))

    def _calculate_node_degrees(self) -> Dict[str, int]:
        deg: Dict[str, int] = {}
        for edge in self.hyperedges:
            for n in edge:
                deg[n] = deg.get(n, 0) + 1
        return deg

    # ------------------------------------------------------------------
    # Initiator selection
    # ------------------------------------------------------------------
    def _select_initiator_for_add(self, node_degrees: Dict[str, int]) -> str:
        ids = self.entity_adapter.entity_ids
        if not node_degrees:
            return random.choice(ids)
        if random.random() < self.pa_probability:
            weights = [node_degrees.get(e, 0) + 1 for e in ids]
            return random.choices(ids, weights=weights, k=1)[0]
        return random.choice(ids)

    def _select_initiator_for_del(self, node_degrees: Dict[str, int]) -> Optional[str]:
        if not node_degrees:
            return None
        return random.choice(list(node_degrees.keys()))

    def _select_edge_size(self, evolution_params: Dict[str, Any]) -> int:
        priority = evolution_params.get("priority_sizes", [])
        if priority and evolution_params["progress"] > 0.7:
            return random.choice(priority[:3]) if len(priority) >= 3 else priority[0]
        if self.current_edge_index < len(self.edge_size_sequence):
            return self.edge_size_sequence[self.current_edge_index]
        sizes = list(self.edge_size_distribution.keys())
        weights = list(self.edge_size_distribution.values())
        return random.choices(sizes, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------
    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        current_dist = self._get_current_distribution()
        evo = self.evolution_controller.get_evolution_params(len(self.hyperedges), current_dist)

        progress = evo["progress"]
        add_count = evo["add_count"]
        del_count = evo["del_count"]
        status = self.evolution_controller.get_status_description(progress)

        print(f"\n{'─' * 55}")
        print(f"  Iter {iteration + 1} | {status} | {progress * 100:.1f}%")
        print(f"  P_add={evo['p_add']:.3f} -> {add_count}, P_del={evo['p_del']:.3f} -> {del_count}")
        print(f"{'─' * 55}")

        results = {
            "iteration": iteration,
            "progress": progress,
            "p_add": evo["p_add"],
            "p_del": evo["p_del"],
            "edges_before": len(self.hyperedges),
            "added": 0,
            "dissolved": 0,
            "api_calls": 0,
            "api_errors": [],
        }

        node_degrees = self._calculate_node_degrees()
        api_calls = 0

        # --- Edge addition ---
        added = 0
        for _ in range(add_count):
            target_size = self._select_edge_size(evo)
            initiator = self._select_initiator_for_add(node_degrees)

            if target_size == 1:
                self.hyperedges.append([initiator])
                added += 1
                if self.hyperedge_labeler:
                    self.hyperedge_labeler.predict_label([initiator])
                if self.current_edge_index < len(self.edge_size_sequence):
                    self.current_edge_index += 1
                node_degrees[initiator] = node_degrees.get(initiator, 0) + 1
                continue

            ctx = {
                "initiator_id": initiator,
                "initiator_data": self.entities.get(initiator, {}),
                "existing_hyperedges": self.hyperedges,
                "target_edge_size": target_size,
            }
            try:
                decision = self.agents["generator"].make_decision(ctx)
                api_calls += 1
                new_edge = decision["selected_members"]
                if new_edge:
                    self.hyperedges.append(new_edge)
                    added += 1
                    if self.hyperedge_labeler:
                        self.hyperedge_labeler.predict_label(new_edge)
                    self.decision_log.append({
                        "iteration": iteration, "action": "add_edge",
                        "edge_id": len(self.hyperedges),
                        "initiator": initiator,
                        "target_size": target_size,
                        "actual_size": len(new_edge),
                        "members": new_edge,
                        "reasoning": decision.get("reasoning", ""),
                        "timestamp": datetime.now().isoformat(),
                    })
                    if self.current_edge_index < len(self.edge_size_sequence):
                        self.current_edge_index += 1
                    for n in new_edge:
                        node_degrees[n] = node_degrees.get(n, 0) + 1
            except Exception as e:
                results["api_errors"].append(str(e))

        results["added"] = added

        # --- Edge dissolution ---
        dissolved = 0
        for _ in range(del_count):
            if not self.hyperedges:
                break
            initiator = self._select_initiator_for_del(node_degrees)
            if initiator is None:
                break

            cands = [(i, e) for i, e in enumerate(self.hyperedges) if initiator in e]
            if not cands:
                continue

            ctx = {
                "initiator_id": initiator,
                "initiator_data": self.entities.get(initiator, {}),
                "candidate_edges": cands,
                "all_hyperedges": self.hyperedges,
            }
            try:
                decision = self.agents["dissolver"].make_decision(ctx)
                if decision.get("api_called"):
                    api_calls += 1
                if decision["decision"] == "DISSOLVE" and decision["edge_to_dissolve"] is not None:
                    idx = decision["edge_to_dissolve"]
                    if 0 <= idx < len(self.hyperedges):
                        removed = self.hyperedges.pop(idx)
                        dissolved += 1
                        if self.hyperedge_labeler and idx < len(self.hyperedge_labeler.label_history):
                            self.hyperedge_labeler.label_history.pop(idx)
                        self.decision_log.append({
                            "iteration": iteration, "action": "dissolve_edge",
                            "edge_id": idx, "initiator": initiator,
                            "dissolved_members": removed,
                            "edge_size": len(removed),
                            "reasoning": decision.get("reasoning", ""),
                            "timestamp": datetime.now().isoformat(),
                        })
                        for n in removed:
                            if n in node_degrees:
                                node_degrees[n] -= 1
                                if node_degrees[n] <= 0:
                                    del node_degrees[n]
            except Exception as e:
                results["api_errors"].append(str(e))

        results["dissolved"] = dissolved
        results["edges_after"] = len(self.hyperedges)
        results["api_calls"] = api_calls

        net = added - dissolved
        pct = 100 * len(self.hyperedges) / self.total_target_edges if self.total_target_edges else 0
        print(f"  +{added} -{dissolved} = {'+' if net >= 0 else ''}{net}  |  "
              f"{len(self.hyperedges)}/{self.total_target_edges} ({pct:.1f}%)  |  "
              f"API calls: {api_calls}")

        self.evolution_history.append(results)
        self._save_checkpoint(iteration)
        return results

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------
    def _check_convergence(self) -> bool:
        cur = len(self.hyperedges)
        tgt = self.total_target_edges
        if tgt == 0:
            return True
        if abs(cur - tgt) / tgt <= 0.05:
            dist = self._get_current_distribution()
            gap = self.evolution_controller._calculate_distribution_gap(dist)
            max_gap = max(abs(g) for g in gap.values()) if gap else 0
            if max_gap <= max(3, tgt * 0.02):
                return True
        return False

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def _generate_analysis_report(self) -> Dict[str, Any]:
        add_dec = [d for d in self.decision_log if d["action"] == "add_edge"]
        dis_dec = [d for d in self.decision_log if d["action"] == "dissolve_edge"]

        participation: Dict[str, int] = {}
        for d in add_dec:
            for m in d.get("members", []):
                participation[m] = participation.get(m, 0) + 1

        initiators: Dict[str, int] = {}
        for d in self.decision_log:
            init = d.get("initiator", "")
            initiators[init] = initiators.get(init, 0) + 1

        sizes = [d.get("actual_size", 0) for d in add_dec]
        final_dist = self._get_current_distribution()

        return {
            "summary": {
                "total_add_decisions": len(add_dec),
                "total_dissolve_decisions": len(dis_dec),
                "unique_nodes_participated": len(participation),
                "unique_initiators": len(initiators),
                "avg_edge_size": sum(sizes) / len(sizes) if sizes else 0,
            },
            "edge_size_distribution": {str(k): v for k, v in sorted(final_dist.items())},
            "target_distribution": {str(k): v for k, v in sorted(self.edge_size_distribution.items())},
        }

    def save_final_results(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            for edge in self.hyperedges:
                f.write(" ".join(map(str, edge)) + "\n")

        hist_path = os.path.join(self.protected_run_dir, "evolution_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.evolution_history, f, indent=2)

        log_path = os.path.join(self.protected_run_dir, "decision_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.decision_log, f, indent=2)

        analysis = self._generate_analysis_report()
        rpt_path = os.path.join(self.protected_run_dir, "analysis_report.json")
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        if self.hyperedge_labeler:
            lbl_cnt = len(self.hyperedge_labeler.label_history)
            edge_cnt = len(self.hyperedges)
            if lbl_cnt > edge_cnt:
                self.hyperedge_labeler.label_history = self.hyperedge_labeler.label_history[:edge_cnt]
            elif lbl_cnt < edge_cnt:
                labels = list(self.hyperedge_labeler.dataset_config["labels"].keys())
                while len(self.hyperedge_labeler.label_history) < edge_cnt:
                    self.hyperedge_labeler.label_history.append({
                        "predicted_label": random.choice(labels),
                        "label_name": "auto-filled",
                        "reasoning": "Automatically filled to match edge count",
                    })
            self.hyperedge_labeler.save_labels(self.protected_run_dir)

        summary = {
            "method": "Semantic Preferential Attachment (SPA)",
            "completion_time": datetime.now().isoformat(),
            "total_iterations": len(self.evolution_history),
            "final_edges": len(self.hyperedges),
            "target_edges": self.total_target_edges,
            "total_added": sum(r["added"] for r in self.evolution_history),
            "total_dissolved": sum(r["dissolved"] for r in self.evolution_history),
        }
        sum_path = os.path.join(self.protected_run_dir, "run_summary.json")
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {self.protected_run_dir}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self):
        print(f"\n{'=' * 60}")
        print("Starting Semantic PA Hypergraph Generation")
        print(f"{'=' * 60}\n")

        start = time.time()
        converged = False

        try:
            for it in range(self.max_iterations):
                self.run_iteration(it)
                if self._check_convergence():
                    print("\nConverged to target distribution.")
                    converged = True
                    break

            if not converged:
                print("\nReached maximum iterations.")

            self.save_final_results()
            elapsed = time.time() - start

            print(f"\n{'=' * 60}")
            print(f"Generation complete  |  {elapsed:.1f}s")
            print(f"Final edges: {len(self.hyperedges)} / {self.total_target_edges}")
            print(f"Output: {self.output_file}")
            print(f"{'=' * 60}\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
            self.save_final_results()
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            self.save_final_results()


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Semantic Preferential Attachment (SPA) Hypergraph Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --entities personas.json --config email-Eu.txt
  python run.py --entities personas.json --config email-Eu.txt --pa_prob 0.85
  python run.py --entities personas.json --config email-Eu.txt --candidate_pool_size 30
  python run.py --entities personas.json --config email-Eu.txt --model gpt-4o
        """,
    )
    parser.add_argument("--entities", type=str, required=True,
                        help="Path to entity JSON file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config hypergraph file (edge size distribution)")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--max_iterations", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        choices=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o",
                                 "qwen2.5-3b", "qwen2.5-7b"],
                        help="LLM model to use")
    parser.add_argument("--pa_prob", type=float, default=0.7,
                        help="Preferential attachment probability 0-1 (default: 0.7)")
    parser.add_argument("--candidate_pool_size", type=int, default=15,
                        help="Number of candidate nodes shown to the LLM per edge-add decision (default: 15)")
    parser.add_argument("--no_analysis", action="store_true",
                        help="Suppress detailed analysis output to save cost")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                        help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--labeled_dataset", type=str, default=None,
                        choices=["music", "restaurant", "bars", "geometry", "algebra"],
                        help="Labeled dataset type for hyperedge label prediction")
    parser.add_argument("--semantic", type=str, default=None,
                        help=f"Semantic preset ({', '.join(SEMANTIC_PRESETS)}) or custom description")

    args = parser.parse_args()

    if args.candidate_pool_size < 1:
        parser.error("--candidate_pool_size must be a positive integer")

    semantic_desc = None
    if args.semantic:
        semantic_desc = SEMANTIC_PRESETS.get(args.semantic.lower(), args.semantic)

    generator = SemanticPAHypergraphGenerator(
        entities_file=args.entities,
        config_hypergraph_file=args.config,
        output_path=args.output,
        max_iterations=args.max_iterations,
        model=args.model,
        pa_probability=args.pa_prob,
        no_analysis=args.no_analysis,
        semantic_description=semantic_desc,
        labeled_dataset=args.labeled_dataset,
        base_url=args.base_url,
        candidate_pool_size=args.candidate_pool_size,
    )
    generator.run()


if __name__ == "__main__":
    main()

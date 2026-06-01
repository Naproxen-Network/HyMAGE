"""
Structural Metrics Module — Python port of MATLAB Hypergraph-Evaluation
========================================================================
8 structural patterns of real-world hypergraphs:
  P1. Heavy-tailed degree distribution           (Distribution → JS divergence)
  P2. Heavy-tailed hyperedge size distribution    (Distribution → JS divergence)
  P3. Heavy-tailed intersection size distribution (Distribution → JS divergence)
  P4. Skewed singular value distribution          (Trend → Pearson correlation)
  P5. Intersecting pairs                          (Trend → Pearson correlation)
  P6. Heavy-tailed group degree distribution      (Distribution → JS divergence)
  P7. Heavy-tailed hypercoreness distribution     (Distribution → JS divergence)
  P8. Power-law persistence                       (Distribution → JS divergence)
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
from collections import defaultdict, Counter
from itertools import combinations
import io, base64, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────── Hypergraph Loader ───────────────────────────

class Hypergraph:
    """Hypergraph data structure with pre-computed adjacency."""
    def __init__(self):
        self.hyperedges = []
        self.nodes = set()
        self.node_to_edges = defaultdict(list)

    @classmethod
    def from_file(cls, filepath):
        hg = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    nodes = list(map(int, line.split()))
                except ValueError:
                    continue
                if len(nodes) >= 2:
                    eidx = len(hg.hyperedges)
                    hg.hyperedges.append(nodes)
                    for n in nodes:
                        hg.nodes.add(n)
                        hg.node_to_edges[n].append(eidx)
        return hg

    @classmethod
    def from_lines(cls, lines):
        hg = cls()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                nodes = list(map(int, line.split()))
            except ValueError:
                continue
            if len(nodes) >= 2:
                eidx = len(hg.hyperedges)
                hg.hyperedges.append(nodes)
                for n in nodes:
                    hg.nodes.add(n)
                    hg.node_to_edges[n].append(eidx)
        return hg

    def build_incidence_matrix(self):
        node_list = sorted(self.nodes)
        node_idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)
        n_edges = len(self.hyperedges)
        H = lil_matrix((n_nodes, n_edges), dtype=np.float32)
        for j, he in enumerate(self.hyperedges):
            for nid in he:
                if nid in node_idx:
                    H[node_idx[nid], j] = 1.0
        return H.tocsr(), node_list, node_idx

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.hyperedges)

    def summary(self):
        sizes = [len(he) for he in self.hyperedges]
        if not sizes:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "avg_edge_size": round(np.mean(sizes), 2),
            "min_edge_size": min(sizes),
            "max_edge_size": max(sizes),
            "avg_degree": round(sum(sizes) / max(self.num_nodes, 1), 2),
        }

# ─────────────────────── Analysis Functions ───────────────────────

METRIC_INFO = [
    {"id": 1, "name": "Degree Distribution",           "type": "distribution", "xlabel": "Degree",            "ylabel": "Count"},
    {"id": 2, "name": "Hyperedge Size Distribution",    "type": "distribution", "xlabel": "Edge Size",          "ylabel": "Count"},
    {"id": 3, "name": "Intersection Size Distribution", "type": "distribution", "xlabel": "Intersection Size",  "ylabel": "Count"},
    {"id": 4, "name": "Singular Value Distribution",    "type": "trend",        "xlabel": "Rank",               "ylabel": "Singular Value"},
    {"id": 5, "name": "Intersecting Pairs",             "type": "trend",        "xlabel": "# All Pairs",        "ylabel": "# Intersecting Pairs"},
    {"id": 6, "name": "Group Degree Distribution",      "type": "distribution", "xlabel": "Group Degree",       "ylabel": "Count"},
    {"id": 7, "name": "Hypercoreness Distribution",     "type": "distribution", "xlabel": "Hypercoreness",      "ylabel": "Count"},
    {"id": 8, "name": "Power-law Persistence",          "type": "distribution", "xlabel": "Persistence",        "ylabel": "Count"},
]


def analyze_degrees(hg):
    """P1: Node degree distribution."""
    degrees = [len(hg.node_to_edges[n]) for n in hg.nodes]
    counter = Counter(degrees)
    vals = sorted(counter.keys())
    freqs = [counter[v] for v in vals]
    return {"values": vals, "frequencies": freqs}


def analyze_hyperedge_sizes(hg):
    """P2: Hyperedge size distribution."""
    sizes = [len(he) for he in hg.hyperedges]
    counter = Counter(sizes)
    vals = sorted(counter.keys())
    freqs = [counter[v] for v in vals]
    return {"values": vals, "frequencies": freqs}


def analyze_intersection_sizes(hg):
    """P3: Pairwise hyperedge intersection size distribution."""
    size_cnt = Counter()
    for i in range(len(hg.hyperedges)):
        counter = defaultdict(int)
        for node in hg.hyperedges[i]:
            for j in hg.node_to_edges[node]:
                if j < i:
                    counter[j] += 1
        for j, isz in counter.items():
            size_cnt[isz] += 1
    vals = sorted(size_cnt.keys())
    freqs = [size_cnt[v] for v in vals]
    return {"values": vals, "frequencies": freqs}


def analyze_singular_values(hg):
    """P4: Singular value distribution of incidence matrix."""
    H, _, _ = hg.build_incidence_matrix()
    k = min(100, min(H.shape) - 1)
    if k < 1:
        return {"ranks": [], "values": []}
    try:
        s = svds(H, k=k, return_singular_vectors=False)
        s = np.sort(s)[::-1]
    except Exception:
        s = np.linalg.svd(H.toarray(), compute_uv=False)
        s = s[:k]
    ranks = list(range(1, len(s) + 1))
    return {"ranks": ranks, "values": [round(float(v), 6) for v in s]}


def analyze_intersecting_pairs(hg):
    """P5: Cumulative intersecting pairs as edges are added."""
    num_edges = len(hg.hyperedges)
    node_to_past = defaultdict(set)
    all_pairs_list = []
    inter_pairs_list = []
    cumulative = 0
    num_points = min(20, num_edges)
    step = max(1, num_edges // num_points)

    for i in range(num_edges):
        intersecting = set()
        for node in hg.hyperedges[i]:
            for j in node_to_past[node]:
                if j < i:
                    intersecting.add(j)
            node_to_past[node].add(i)
        cumulative += len(intersecting)
        if (i + 1) % step == 0 or i == num_edges - 1:
            total_possible = (i + 1) * i // 2
            if total_possible > 0 and cumulative > 0:
                all_pairs_list.append(total_possible)
                inter_pairs_list.append(cumulative)

    return {"all_pairs": all_pairs_list, "intersecting_pairs": inter_pairs_list}


def analyze_group_degrees(hg):
    """P6: Group (pair/triple) degree distribution."""
    group_counts = Counter()
    for he in hg.hyperedges:
        nodes = sorted(set(he))
        # Pairs
        for pair in combinations(nodes, 2):
            group_counts[pair] += 1
        # Triples (sample if too large)
        sample = nodes[:20] if len(nodes) > 20 else nodes
        for triple in combinations(sample, 3):
            group_counts[triple] += 1

    degrees = list(group_counts.values())
    if not degrees:
        return {"values": [], "frequencies": []}
    counter = Counter(degrees)
    vals = sorted(counter.keys())
    freqs = [counter[v] for v in vals]
    return {"values": vals, "frequencies": freqs}


def analyze_hypercoreness(hg):
    """P7: Hypercoreness (k-core decomposition) distribution."""
    node_list = sorted(hg.nodes)
    n = len(node_list)
    if n == 0:
        return {"values": [], "frequencies": []}

    node_idx = {nd: i for i, nd in enumerate(node_list)}
    degrees = np.array([len(hg.node_to_edges[nd]) for nd in node_list], dtype=int)
    coreness = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)

    # Build edge membership for fast updates
    edge_members = [set(hg.hyperedges[j]) for j in range(len(hg.hyperedges))]

    k = 1
    max_k = int(degrees.max()) if n > 0 else 0

    while np.any(remaining):
        to_remove = np.where(remaining & (degrees < k))[0]
        if len(to_remove) == 0:
            k += 1
            if k > max_k:
                coreness[remaining] = k - 1
                break
        else:
            coreness[to_remove] = k - 1
            remaining[to_remove] = False
            removed_nodes = {node_list[i] for i in to_remove}
            # Update degrees
            for i in np.where(remaining)[0]:
                nd = node_list[i]
                eff_deg = 0
                for eidx in hg.node_to_edges[nd]:
                    has_other = False
                    for m in edge_members[eidx]:
                        if m != nd and m in node_idx and remaining[node_idx[m]]:
                            has_other = True
                            break
                    if has_other:
                        eff_deg += 1
                degrees[i] = eff_deg

    core_values = coreness.tolist()
    counter = Counter(core_values)
    vals = sorted(counter.keys())
    freqs = [counter[v] for v in vals]
    return {"values": vals, "frequencies": freqs}


def analyze_power_law_persistence(hg):
    """P8: Power-law persistence of node groups."""
    persistence = Counter()
    max_edges = min(len(hg.hyperedges), 2000)
    for i in range(max_edges):
        nodes = sorted(set(hg.hyperedges[i]))
        for pair in combinations(nodes, 2):
            persistence[pair] += 1
        sample = nodes[:10] if len(nodes) > 10 else nodes
        for triple in combinations(sample, 3):
            persistence[triple] += 1

    persist_vals = [v for v in persistence.values() if v > 1]
    if not persist_vals:
        return {"values": [], "frequencies": []}
    counter = Counter(persist_vals)
    vals = sorted(counter.keys())
    freqs = [counter[v] for v in vals]
    return {"values": vals, "frequencies": freqs}


ALL_ANALYSES = [
    analyze_degrees,
    analyze_hyperedge_sizes,
    analyze_intersection_sizes,
    analyze_singular_values,
    analyze_intersecting_pairs,
    analyze_group_degrees,
    analyze_hypercoreness,
    analyze_power_law_persistence,
]


def run_all_analyses(hg):
    """Run all 8 analyses and return results dict."""
    results = {}
    for i, func in enumerate(ALL_ANALYSES):
        info = METRIC_INFO[i]
        try:
            data = func(hg)
            results[info["id"]] = {"info": info, "data": data, "error": None}
        except Exception as e:
            results[info["id"]] = {"info": info, "data": None, "error": str(e)}
    return results


# ─────────────────────── Comparison Metrics ───────────────────────

def compute_js_divergence(dist1, dist2):
    """
    Compute Jensen-Shannon divergence between two frequency distributions.
    dist1, dist2: dicts with 'values' and 'frequencies' keys.
    Returns float in [0, 1].
    """
    v1, f1 = dist1["values"], dist1["frequencies"]
    v2, f2 = dist2["values"], dist2["frequencies"]
    if not v1 or not f1 or not v2 or not f2:
        return float('nan')

    # Reconstruct raw data from value-frequency pairs
    seq1 = []
    for v, freq in zip(v1, f1):
        seq1.extend([v] * freq)
    seq2 = []
    for v, freq in zip(v2, f2):
        seq2.extend([v] * freq)

    seq1 = np.array(seq1, dtype=float)
    seq2 = np.array(seq2, dtype=float)

    min_val = min(seq1.min(), seq2.min())
    max_val = max(seq1.max(), seq2.max())

    if max_val - min_val > 1000:
        num_bins = min(1000, max(50, int(np.ceil(np.sqrt(len(seq1) + len(seq2))))))
        edges = np.linspace(min_val, max_val + 1e-10, num_bins + 1)
    else:
        edges = np.arange(min_val, max_val + 2, 1)

    p, _ = np.histogram(seq1, bins=edges, density=True)
    q, _ = np.histogram(seq2, bins=edges, density=True)

    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(max(0, min(1, jsd)))


def compute_pearson_for_trend(data1, data2, metric_id):
    """
    Compute Pearson correlation for trend-type metrics.
    metric_id 4: singular values (rank → value)
    metric_id 5: intersecting pairs (all_pairs → intersecting_pairs)
    """
    if metric_id == 4:
        v1 = np.array(data1.get("values", []), dtype=float)
        v2 = np.array(data2.get("values", []), dtype=float)
    elif metric_id == 5:
        v1 = np.array(data1.get("intersecting_pairs", []), dtype=float)
        v2 = np.array(data2.get("intersecting_pairs", []), dtype=float)
    else:
        return float('nan')

    if len(v1) == 0 or len(v2) == 0:
        return float('nan')

    min_len = min(len(v1), len(v2))
    v1 = v1[:min_len]
    v2 = v2[:min_len]

    if np.std(v1) == 0 or np.std(v2) == 0:
        return float('nan')

    r, _ = pearsonr(v1, v2)
    return float(r)


def compare_two_hypergraphs(results1, results2):
    """
    Compare two sets of analysis results.
    Distribution metrics: JS divergence (lower is better)
    Trend metrics: Pearson correlation (higher is better)
    """
    comparison = {}
    for mid in range(1, 9):
        info = METRIC_INFO[mid - 1]
        r1 = results1.get(mid, {})
        r2 = results2.get(mid, {})
        d1 = r1.get("data")
        d2 = r2.get("data")

        if d1 is None or d2 is None:
            comparison[mid] = {
                "metric": info["name"],
                "type": info["type"],
                "score": None,
                "score_label": "JS Divergence" if info["type"] == "distribution" else "Pearson r",
                "error": "Data unavailable"
            }
            continue

        try:
            if info["type"] == "distribution":
                score = compute_js_divergence(d1, d2)
                label = "JS Divergence"
            else:
                score = compute_pearson_for_trend(d1, d2, mid)
                label = "Pearson r"
            comparison[mid] = {
                "metric": info["name"],
                "type": info["type"],
                "score": round(score, 6) if not math.isnan(score) else None,
                "score_label": label,
                "error": None
            }
        except Exception as e:
            comparison[mid] = {
                "metric": info["name"],
                "type": info["type"],
                "score": None,
                "score_label": "JS Divergence" if info["type"] == "distribution" else "Pearson r",
                "error": str(e)
            }
    return comparison


# ─────────────────────── Plot Generation ───────────────────────

def generate_plot_base64(data, info, title_suffix=""):
    """Generate a log-log or linear plot and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    if info["type"] == "distribution":
        vals = data.get("values", [])
        freqs = data.get("frequencies", [])
        if not vals or not freqs:
            plt.close(fig)
            return None
        positive = [(v, f) for v, f in zip(vals, freqs) if v > 0 and f > 0]
        if len(positive) < 2:
            plt.close(fig)
            return None
        x, y = zip(*positive)
        ax.loglog(x, y, 'o', color='#64b5f6', markersize=5, alpha=0.85)
        ax.set_xlabel(info["xlabel"], fontsize=11, color='white')
        ax.set_ylabel(info["ylabel"], fontsize=11, color='white')
    elif info["id"] == 4:
        ranks = data.get("ranks", [])
        values = data.get("values", [])
        if not ranks or not values:
            plt.close(fig)
            return None
        positive = [(r, v) for r, v in zip(ranks, values) if r > 0 and v > 0]
        if len(positive) < 2:
            plt.close(fig)
            return None
        x, y = zip(*positive)
        ax.loglog(x, y, 'o', color='#81c784', markersize=5, alpha=0.85)
        ax.set_xlabel(info["xlabel"], fontsize=11, color='white')
        ax.set_ylabel(info["ylabel"], fontsize=11, color='white')
    elif info["id"] == 5:
        ap = data.get("all_pairs", [])
        ip = data.get("intersecting_pairs", [])
        if not ap or not ip:
            plt.close(fig)
            return None
        positive = [(a, i) for a, i in zip(ap, ip) if a > 0 and i > 0]
        if len(positive) < 2:
            plt.close(fig)
            return None
        x, y = zip(*positive)
        ax.loglog(x, y, 'o-', color='#ffb74d', markersize=5, alpha=0.85)
        ax.set_xlabel(info["xlabel"], fontsize=11, color='white')
        ax.set_ylabel(info["ylabel"], fontsize=11, color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#555')
    if title_suffix:
        ax.set_title(title_suffix, fontsize=10, color='#aaa')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

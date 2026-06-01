"""
HyMAGE Benchmark Tools — Flask Web Platform
===========================================
Modules:
  1. Structural Metrics (P1-P8)
  2. Hypergraph Annotation
"""

import os
import json
import uuid
import threading
import datetime
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   session, send_from_directory)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
ANNOTATE_DIR = os.path.join(BASE_DIR, 'annotations')
os.makedirs(ANNOTATE_DIR, exist_ok=True)

# ── Global task store (in-memory, for simplicity) ──
tasks = {}

# ── Network type templates for annotation ──
NETWORK_TYPES = {
    'collaboration': {
        'label': 'Collaboration Network',
        'icon': 'fa-users',
        'profile_fields': [
            {'key': 'name', 'label': 'Name', 'type': 'text', 'required': True},
            {'key': 'workplace', 'label': 'Workplace / Affiliation', 'type': 'text', 'required': True},
            {'key': 'age', 'label': 'Age', 'type': 'number', 'required': False},
            {'key': 'title', 'label': 'Title / Position', 'type': 'text', 'required': False},
            {'key': 'gender', 'label': 'Gender', 'type': 'select',
             'options': ['Male', 'Female', 'Other'], 'required': False},
            {'key': 'research_interests', 'label': 'Research Interests', 'type': 'text', 'required': False},
            {'key': 'academic_service', 'label': 'Academic Service', 'type': 'text', 'required': False},
        ],
        'edge_fields': [
            {'key': 'paper_title', 'label': 'Paper Title', 'type': 'text'},
            {'key': 'category', 'label': 'Category (Label)', 'type': 'label_select'},
            {'key': 'abstract', 'label': 'Abstract', 'type': 'textarea'},
        ]
    },
    'social': {
        'label': 'Social Network',
        'icon': 'fa-share-alt',
        'profile_fields': [
            {'key': 'name', 'label': 'Name', 'type': 'text', 'required': True},
            {'key': 'platform', 'label': 'Platform', 'type': 'text', 'required': False},
            {'key': 'location', 'label': 'Location', 'type': 'text', 'required': False},
            {'key': 'age', 'label': 'Age', 'type': 'number', 'required': False},
            {'key': 'gender', 'label': 'Gender', 'type': 'select',
             'options': ['Male', 'Female', 'Other'], 'required': False},
            {'key': 'interests', 'label': 'Interests', 'type': 'text', 'required': False},
            {'key': 'bio', 'label': 'Bio', 'type': 'textarea', 'required': False},
        ],
        'edge_fields': [
            {'key': 'group_name', 'label': 'Group / Event Name', 'type': 'text'},
            {'key': 'category', 'label': 'Category (Label)', 'type': 'label_select'},
            {'key': 'description', 'label': 'Description', 'type': 'textarea'},
        ]
    },
    'drug': {
        'label': 'Drug Network',
        'icon': 'fa-pills',
        'profile_fields': [
            {'key': 'name', 'label': 'Drug Name', 'type': 'text', 'required': True},
            {'key': 'drug_type', 'label': 'Drug Type', 'type': 'text', 'required': False},
            {'key': 'mechanism', 'label': 'Mechanism', 'type': 'text', 'required': False},
            {'key': 'indication', 'label': 'Indication', 'type': 'text', 'required': False},
            {'key': 'side_effects', 'label': 'Side Effects', 'type': 'text', 'required': False},
            {'key': 'molecular_weight', 'label': 'Molecular Weight', 'type': 'text', 'required': False},
        ],
        'edge_fields': [
            {'key': 'interaction_type', 'label': 'Interaction Type', 'type': 'text'},
            {'key': 'category', 'label': 'Category (Label)', 'type': 'label_select'},
            {'key': 'description', 'label': 'Description', 'type': 'textarea'},
        ]
    },
    'protein': {
        'label': 'Protein Network',
        'icon': 'fa-dna',
        'profile_fields': [
            {'key': 'name', 'label': 'Protein Name', 'type': 'text', 'required': True},
            {'key': 'organism', 'label': 'Organism', 'type': 'text', 'required': False},
            {'key': 'gene', 'label': 'Gene', 'type': 'text', 'required': False},
            {'key': 'function', 'label': 'Function', 'type': 'text', 'required': False},
            {'key': 'structure', 'label': 'Structure Type', 'type': 'text', 'required': False},
            {'key': 'pathway', 'label': 'Pathway', 'type': 'text', 'required': False},
        ],
        'edge_fields': [
            {'key': 'complex_name', 'label': 'Complex Name', 'type': 'text'},
            {'key': 'category', 'label': 'Category (Label)', 'type': 'label_select'},
            {'key': 'description', 'label': 'Description', 'type': 'textarea'},
        ]
    },
    'other': {
        'label': 'Other Network',
        'icon': 'fa-project-diagram',
        'profile_fields': [
            {'key': 'name', 'label': 'Name / ID', 'type': 'text', 'required': True},
            {'key': 'node_type', 'label': 'Type', 'type': 'text', 'required': False},
            {'key': 'description', 'label': 'Description', 'type': 'text', 'required': False},
            {'key': 'attribute_1', 'label': 'Attribute 1', 'type': 'text', 'required': False},
            {'key': 'attribute_2', 'label': 'Attribute 2', 'type': 'text', 'required': False},
        ],
        'edge_fields': [
            {'key': 'edge_name', 'label': 'Edge Name', 'type': 'text'},
            {'key': 'category', 'label': 'Category (Label)', 'type': 'label_select'},
            {'key': 'description', 'label': 'Description', 'type': 'textarea'},
        ]
    }
}

# ══════════════════════════ Page Routes ══════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/structural')
def structural():
    return render_template('structural.html')

@app.route('/annotate')
def annotate():
    return render_template('annotate.html')

# ══════════════════════════ Dataset API ══════════════════════════

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all uploaded datasets."""
    datasets = []
    if os.path.isdir(UPLOAD_DIR):
        for name in sorted(os.listdir(UPLOAD_DIR)):
            dpath = os.path.join(UPLOAD_DIR, name)
            if os.path.isdir(dpath):
                files = [f for f in os.listdir(dpath)
                         if os.path.isfile(os.path.join(dpath, f))]
                txt_files = [f for f in files if f.endswith('.txt')]
                json_files = [f for f in files if f.endswith('.json')]
                datasets.append({
                    'name': name,
                    'files': files,
                    'txt_count': len(txt_files),
                    'json_count': len(json_files),
                })
    return jsonify(datasets)


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload files to a dataset."""
    dataset_name = request.form.get('dataset', 'default')
    dataset_name = dataset_name.strip() or 'default'
    # Sanitize
    dataset_name = "".join(c for c in dataset_name if c.isalnum() or c in '-_ ')
    dpath = os.path.join(UPLOAD_DIR, dataset_name)
    os.makedirs(dpath, exist_ok=True)

    saved = []
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    for f in request.files.getlist('files'):
        if f.filename:
            fname = f.filename
            fpath = os.path.join(dpath, fname)
            f.save(fpath)
            saved.append(fname)

    return jsonify({'dataset': dataset_name, 'files': saved})


@app.route('/api/dataset/<name>/files', methods=['GET'])
def dataset_files(name):
    """List files in a dataset."""
    dpath = os.path.join(UPLOAD_DIR, name)
    if not os.path.isdir(dpath):
        return jsonify({'error': 'Dataset not found'}), 404
    files = []
    for f in sorted(os.listdir(dpath)):
        fp = os.path.join(dpath, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp)
            files.append({'name': f, 'size': size,
                          'type': 'json' if f.endswith('.json') else 'txt'})
    return jsonify({'dataset': name, 'files': files})


# ══════════════════════════ Structural Metrics API ══════════════════════════

@app.route('/api/structural/analyze', methods=['POST'])
def structural_analyze():
    """Run structural analysis on uploaded files."""
    from modules.structural_metrics import (
        Hypergraph, run_all_analyses, compare_two_hypergraphs,
        generate_plot_base64, METRIC_INFO
    )

    data = request.get_json()
    dataset = data.get('dataset', '')
    file_names = data.get('files', [])

    if len(file_names) < 1:
        return jsonify({'error': 'Please select at least 1 file'}), 400

    dpath = os.path.join(UPLOAD_DIR, dataset)
    if not os.path.isdir(dpath):
        return jsonify({'error': 'Dataset not found'}), 404

    # Load hypergraphs
    hypergraphs = {}
    for fname in file_names:
        fpath = os.path.join(dpath, fname)
        if not os.path.isfile(fpath):
            continue
        hg = Hypergraph.from_file(fpath)
        if hg.num_nodes > 0:
            hypergraphs[fname] = hg

    if not hypergraphs:
        return jsonify({'error': 'No valid hypergraph files'}), 400

    # Run analyses
    all_results = {}
    all_plots = {}
    for fname, hg in hypergraphs.items():
        results = run_all_analyses(hg)
        all_results[fname] = results

        # Generate plots
        plots = {}
        for mid, rdata in results.items():
            if rdata['data'] is not None:
                img = generate_plot_base64(rdata['data'], rdata['info'],
                                            title_suffix=fname)
                if img:
                    plots[mid] = img
        all_plots[fname] = plots

    # Comparisons: always compare against the real (reference) file
    real_file = data.get('real_file', '')
    comparisons = {}
    if real_file and real_file in hypergraphs:
        for fname in hypergraphs:
            if fname == real_file:
                continue
            comp = compare_two_hypergraphs(
                all_results[real_file], all_results[fname]
            )
            comparisons[fname] = comp

    # Summaries
    summaries = {fn: hg.summary() for fn, hg in hypergraphs.items()}

    # Serialize results
    serialized = {}
    for fname, results in all_results.items():
        serialized[fname] = {}
        for mid, rdata in results.items():
            info = rdata['info']
            d = rdata['data']
            ser_data = None
            if d is not None:
                ser_data = _convert_numpy(d)
            serialized[fname][mid] = {
                'name': info['name'],
                'type': info['type'],
                'data': ser_data,
                'error': rdata['error']
            }

    return jsonify({
        'summaries': _convert_numpy(summaries),
        'results': serialized,
        'plots': all_plots,
        'comparisons': _convert_numpy(comparisons),
        'real_file': real_file,
    })


# ══════════════════════════ Annotation API ══════════════════════════

def _load_anno(name):
    """Load an annotation project's data from disk."""
    pdir = os.path.join(ANNOTATE_DIR, name)
    config_path = os.path.join(pdir, 'config.json')
    profiles_path = os.path.join(pdir, 'profiles.json')
    edges_path = os.path.join(pdir, 'edges.json')

    if not os.path.isfile(config_path):
        return None, None, None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    profiles = {}
    if os.path.isfile(profiles_path):
        with open(profiles_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
    edges = []
    if os.path.isfile(edges_path):
        with open(edges_path, 'r', encoding='utf-8') as f:
            edges = json.load(f)

    return config, profiles, edges


def _save_anno(name, config, profiles, edges):
    """Save annotation project data to disk (JSON + TXT)."""
    pdir = os.path.join(ANNOTATE_DIR, name)
    os.makedirs(pdir, exist_ok=True)

    with open(os.path.join(pdir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pdir, 'profiles.json'), 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    with open(os.path.join(pdir, 'edges.json'), 'w', encoding='utf-8') as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    # Also write clean hyperedges.txt
    with open(os.path.join(pdir, 'hyperedges.txt'), 'w', encoding='utf-8') as f:
        for edge in edges:
            f.write(' '.join(str(nid) for nid in edge['node_ids']) + '\n')


@app.route('/api/annotate/network-types', methods=['GET'])
def annotate_network_types():
    """Return available network type templates."""
    return jsonify(NETWORK_TYPES)


@app.route('/api/annotate/projects', methods=['GET'])
def annotate_list_projects():
    """List all annotation projects."""
    projects = []
    if os.path.isdir(ANNOTATE_DIR):
        for name in sorted(os.listdir(ANNOTATE_DIR)):
            pdir = os.path.join(ANNOTATE_DIR, name)
            conf_path = os.path.join(pdir, 'config.json')
            if os.path.isdir(pdir) and os.path.isfile(conf_path):
                with open(conf_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Count profiles and edges
                p_count, e_count = 0, 0
                pp = os.path.join(pdir, 'profiles.json')
                ep = os.path.join(pdir, 'edges.json')
                if os.path.isfile(pp):
                    with open(pp, 'r', encoding='utf-8') as f:
                        p_count = len(json.load(f))
                if os.path.isfile(ep):
                    with open(ep, 'r', encoding='utf-8') as f:
                        e_count = len(json.load(f))
                projects.append({
                    'name': name,
                    'network_type': config.get('network_type', 'other'),
                    'labels': config.get('labels', []),
                    'profile_count': p_count,
                    'edge_count': e_count,
                    'created_at': config.get('created_at', ''),
                })
    return jsonify(projects)


@app.route('/api/annotate/projects', methods=['POST'])
def annotate_create_project():
    """Create a new annotation project."""
    data = request.get_json()
    name = data.get('name', '').strip()
    network_type = data.get('network_type', 'other')
    labels = data.get('labels', [])

    if not name:
        return jsonify({'error': 'Project name is required'}), 400
    # Sanitize
    safe_name = "".join(c for c in name if c.isalnum() or c in '-_ ')
    if not safe_name:
        return jsonify({'error': 'Invalid project name'}), 400

    pdir = os.path.join(ANNOTATE_DIR, safe_name)
    if os.path.isdir(pdir):
        return jsonify({'error': f'Project "{safe_name}" already exists'}), 409

    config = {
        'name': safe_name,
        'network_type': network_type,
        'labels': labels,
        'created_at': datetime.datetime.now().isoformat(),
    }
    _save_anno(safe_name, config, {}, [])
    return jsonify({'success': True, 'name': safe_name})


@app.route('/api/annotate/project/<name>', methods=['GET'])
def annotate_get_project(name):
    """Get full project data."""
    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404

    net_type = config.get('network_type', 'other')
    type_info = NETWORK_TYPES.get(net_type, NETWORK_TYPES['other'])

    return jsonify({
        'config': config,
        'profiles': profiles,
        'edges': edges,
        'type_info': type_info,
    })


@app.route('/api/annotate/project/<name>/labels', methods=['PUT'])
def annotate_update_labels(name):
    """Update the predefined labels for a project."""
    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json()
    config['labels'] = data.get('labels', [])
    _save_anno(name, config, profiles, edges)
    return jsonify({'success': True, 'labels': config['labels']})


@app.route('/api/annotate/project/<name>/search', methods=['GET'])
def annotate_search_profiles(name):
    """Search profiles in a project by keyword."""
    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404

    q = request.args.get('q', '').strip().lower()
    if not q:
        # Return all profiles (limited)
        results = [{'id': pid, **pdata} for pid, pdata in profiles.items()]
        return jsonify(results[:50])

    results = []
    for pid, pdata in profiles.items():
        # Search across all string fields
        match = False
        for key, val in pdata.items():
            if isinstance(val, str) and q in val.lower():
                match = True
                break
            if isinstance(val, dict):
                for cv in val.values():
                    if isinstance(cv, str) and q in cv.lower():
                        match = True
                        break
        if match:
            results.append({'id': pid, **pdata})

    return jsonify(results[:30])


# API Configuration (same as Hypergraph-Generator)
BASE_URL_OPENAI = "https://api.aigc369.com/v1"

def _load_api_key():
    """Load OpenAI API key from environment variable or api-key.txt file (like Hypergraph-Generator)."""
    # First try environment variable
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # Then try api-key.txt file (same as Hypergraph-Generator)
    possible_paths = [
        os.path.join(BASE_DIR, 'api-key.txt'),
        os.path.join(BASE_DIR, 'Hypergraph-Generator', 'api-key.txt'),
        os.path.join(os.path.dirname(BASE_DIR), 'Hypergraph-Generator', 'api-key.txt'),
        'api-key.txt',
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            return line
            except Exception:
                continue
    
    return None


@app.route('/api/annotate/project/<name>/ai-parse/test', methods=['GET'])
def annotate_ai_parse_test(name):
    """Test endpoint to check if AI parse API is accessible."""
    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404
    
    api_key = _load_api_key()
    has_openai = False
    try:
        import openai
        has_openai = True
    except ImportError:
        pass
    
    return jsonify({
        'project': name,
        'has_openai_lib': has_openai,
        'has_api_key': bool(api_key),
        'api_key_length': len(api_key) if api_key else 0,
        'api_key_source': 'env' if os.environ.get('OPENAI_API_KEY') else ('file' if api_key else 'none'),
        'base_url': BASE_URL_OPENAI,
        'status': 'ready' if (has_openai and api_key) else 'not_ready'
    })


@app.route('/api/annotate/project/<name>/ai-parse', methods=['POST'])
def annotate_ai_parse(name):
    """Use GPT-3.5 Turbo to parse text and extract hyperedge metadata and authors."""
    try:
        from openai import OpenAI
    except ImportError:
        return jsonify({'error': 'OpenAI library not installed. Run: pip install openai'}), 500

    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404

    # Parse request JSON
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON in request body'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to parse request JSON: {str(e)}'}), 400

    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Text is required'}), 400

    # Get OpenAI API key from environment or api-key.txt file
    api_key = _load_api_key()
    if not api_key:
        return jsonify({
            'error': 'OpenAI API key not found. Please set OPENAI_API_KEY environment variable or create api-key.txt file (same as Hypergraph-Generator)'
        }), 500

    net_type = config.get('network_type', 'collaboration')
    type_info = NETWORK_TYPES.get(net_type, NETWORK_TYPES['collaboration'])
    labels = config.get('labels', [])

    # Build prompt for GPT-3.5 Turbo
    profile_fields = [f['key'] for f in type_info.get('profile_fields', [])]
    edge_fields = [f['key'] for f in type_info.get('edge_fields', [])]

    labels_str = ', '.join(labels) if labels else 'None defined yet'
    prompt = f"""You are helping to extract information from academic paper text for hypergraph annotation.

Available labels (if any): {labels_str}

IMPORTANT INSTRUCTIONS:
1. For "category" field:
   - If available labels exist, try to match the paper's research area to one of them
   - If no labels match, suggest a new category based on the paper's research field (e.g., "Materials Science", "Machine Learning", "Systems", etc.)
   - If no labels are defined yet, infer a category from the paper content (e.g., from journal name, keywords, or research area)
   - ALWAYS provide a category value, never leave it empty

2. For authors:
   - Extract ALL authors from the text
   - Parse author names carefully (may be separated by commas, "&", or "and")
   - Extract affiliations/institutions if mentioned
   - If multiple authors share the same affiliation, assign it to all of them
   - If affiliation is not mentioned, leave workplace empty (don't guess)

3. For paper_title:
   - Extract the exact title from the text
   - If title is not found, use the first line or a descriptive title

4. For abstract:
   - Extract the full abstract if available
   - If abstract is not in the text, summarize the main content

Extract the following information and return as JSON:
1. Edge metadata:
   - paper_title: The title of the paper
   - category: A category label (match from available labels if possible, otherwise suggest based on research field)
   - abstract: The abstract or summary

2. Authors: List of ALL authors with their information:
   - name: Full name (first name + last name)
   - workplace: Affiliation/institution (if mentioned)
   - title: Position/title (if mentioned, e.g., "Professor", "PhD Student")
   - research_interests: Research area based on paper content (if inferable)
   - Other fields from: {', '.join(profile_fields)}

Return JSON in this format:
{{
  "edge_metadata": {{
    "paper_title": "...",
    "category": "...",
    "abstract": "..."
  }},
  "authors": [
    {{
      "name": "...",
      "workplace": "...",
      "title": "...",
      "research_interests": "..."
    }}
  ]
}}

Text to parse:
{text}

Return only valid JSON, no markdown formatting. Ensure "category" is always provided."""

    try:
        # Use same base_url as Hypergraph-Generator
        client = OpenAI(api_key=api_key, base_url=BASE_URL_OPENAI)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured information from academic papers. You must always return valid JSON only. When extracting categories, infer from paper content (journal, keywords, research area) if no labels match. Always extract ALL authors mentioned in the text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )

        result_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()

        parsed = json.loads(result_text)

        # Ensure edge_metadata has category
        edge_metadata = parsed.get('edge_metadata', {})
        if not edge_metadata.get('category'):
            # Try to infer category from paper title or journal
            title = edge_metadata.get('paper_title', '').lower()
            abstract = edge_metadata.get('abstract', '').lower()
            text_lower = text.lower()
            
            # Simple keyword-based inference
            category = None
            if any(kw in text_lower for kw in ['nature', 'science', 'materials', 'alloy', 'metal', 'nanomaterial']):
                category = 'Materials Science'
            elif any(kw in text_lower for kw in ['machine learning', 'neural network', 'deep learning', 'ai', 'artificial intelligence']):
                category = 'Machine Learning'
            elif any(kw in text_lower for kw in ['system', 'distributed', 'network', 'protocol']):
                category = 'Systems'
            elif any(kw in text_lower for kw in ['algorithm', 'complexity', 'theory', 'mathematical']):
                category = 'Theory'
            elif any(kw in text_lower for kw in ['drug', 'protein', 'biological', 'medical', 'health']):
                category = 'Biology' if net_type == 'protein' else 'Drug'
            else:
                category = 'Other' if labels else 'General'
            
            edge_metadata['category'] = category
            parsed['edge_metadata'] = edge_metadata

        # Match authors against existing profiles
        matched_authors = []
        new_authors = []

        for author in parsed.get('authors', []):
            author_name = author.get('name', '').strip()
            if not author_name:
                continue

            # Search for matching profile by name
            matched = None
            for pid, pdata in profiles.items():
                pname = pdata.get('name', '').strip()
                if pname.lower() == author_name.lower() or \
                   (pname and author_name.lower() in pname.lower()) or \
                   (pname and pname.lower() in author_name.lower()):
                    matched = {'id': pid, **pdata}
                    break

            if matched:
                # Merge new info into existing profile
                for k, v in author.items():
                    if v and k not in ['name', 'id']:  # Don't overwrite name/id
                        if not matched.get(k) or matched.get(k) == '':
                            matched[k] = v
                matched_authors.append(matched)
            else:
                # New author - add label if available
                if labels:
                    author['label'] = labels[0]  # Default to first label, user can change
                new_authors.append(author)

        return jsonify({
            'success': True,
            'edge_metadata': parsed.get('edge_metadata', {}),
            'matched_authors': matched_authors,
            'new_authors': new_authors,
            'edge_size': len(matched_authors) + len(new_authors)
        })

    except json.JSONDecodeError as e:
        import traceback
        print(f"JSON decode error: {str(e)}")
        print(f"Response text: {result_text[:500] if 'result_text' in locals() else 'N/A'}")
        return jsonify({'error': f'Failed to parse AI response as JSON: {str(e)}. Response may be malformed.'}), 500
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"AI parse error: {str(e)}")
        print(error_trace)
        return jsonify({'error': f'AI parsing failed: {str(e)}'}), 500


@app.route('/api/annotate/project/<name>/edge', methods=['POST'])
def annotate_add_edge(name):
    """Add a hyperedge with profiles.

    Body: {
        label: str,
        profiles: [ {id: str|null, ...profile_data}, ... ],
        metadata: { paper_title, abstract, ... }
    }
    """
    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json()
    edge_label = data.get('label', '')
    prof_list = data.get('profiles', [])
    metadata = data.get('metadata', {})

    if not prof_list:
        return jsonify({'error': 'At least 1 node profile is required'}), 400

    # Determine next profile ID
    max_id = 0
    for pid in profiles:
        try:
            max_id = max(max_id, int(pid))
        except ValueError:
            pass

    node_ids = []
    for p in prof_list:
        existing_id = p.get('id')
        if existing_id and str(existing_id) in profiles:
            # Reuse existing profile – optionally update fields
            pid = str(existing_id)
            # Update fields if provided
            for k, v in p.items():
                if k != 'id' and v not in (None, ''):
                    profiles[pid][k] = v
            node_ids.append(int(pid))
        else:
            # Create new profile
            max_id += 1
            pid = str(max_id)
            new_profile = {k: v for k, v in p.items()
                          if k != 'id' and v is not None}
            profiles[pid] = new_profile
            node_ids.append(max_id)

    # Create edge
    edge_id = len(edges) + 1
    edge = {
        'id': edge_id,
        'node_ids': node_ids,
        'label': edge_label,
        'metadata': metadata,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    edges.append(edge)

    _save_anno(name, config, profiles, edges)

    return jsonify({
        'success': True,
        'edge': edge,
        'profile_count': len(profiles),
        'edge_count': len(edges),
    })


@app.route('/api/annotate/project/<name>/edge/<int:edge_id>', methods=['DELETE'])
def annotate_delete_edge(name, edge_id):
    """Delete a hyperedge by ID."""
    config, profiles, edges = _load_anno(name)
    if config is None:
        return jsonify({'error': 'Project not found'}), 404

    edges = [e for e in edges if e['id'] != edge_id]
    _save_anno(name, config, profiles, edges)
    return jsonify({'success': True, 'edge_count': len(edges)})


@app.route('/api/annotate/project/<name>/export', methods=['GET'])
def annotate_export(name):
    """Return downloadable paths for the annotation files."""
    pdir = os.path.join(ANNOTATE_DIR, name)
    if not os.path.isdir(pdir):
        return jsonify({'error': 'Project not found'}), 404

    files = []
    for fn in ['profiles.json', 'hyperedges.txt', 'edges.json', 'config.json']:
        fp = os.path.join(pdir, fn)
        if os.path.isfile(fp):
            files.append(fn)

    return jsonify({'project': name, 'files': files})


@app.route('/api/annotate/project/<name>/download/<filename>')
def annotate_download(name, filename):
    """Download an annotation file."""
    pdir = os.path.join(ANNOTATE_DIR, name)
    safe = os.path.basename(filename)
    return send_from_directory(pdir, safe, as_attachment=True)


# ══════════════════════════ Utility ══════════════════════════

def _convert_numpy(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Flask 2.3+ uses json_provider_class; older uses json_encoder
try:
    from flask.json.provider import DefaultJSONProvider
    class NumpyJSONProvider(DefaultJSONProvider):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    app.json_provider_class = NumpyJSONProvider
    app.json = NumpyJSONProvider(app)
except (ImportError, AttributeError):
    try:
        app.json_encoder = NumpyEncoder
    except Exception:
        pass


# ══════════════════════════ Main ══════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Hypergraph Benchmarks Platform")
    print("  http://127.0.0.1:5005")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5005, use_reloader=False)

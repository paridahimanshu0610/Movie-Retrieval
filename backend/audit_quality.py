import json
import os
import glob
import re
from collections import defaultdict, Counter

OUTPUT_DIR = "c:/Users/saswa/projects/csce670_screenplay/output"
MANIFEST_PATH = "c:/Users/saswa/projects/csce670_screenplay/screenplays/manifest.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None, str(e)

def safe_load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

# ── load manifest ─────────────────────────────────────────────────────────────
manifest, err = safe_load(MANIFEST_PATH)
if err:
    print(f"CRITICAL: Cannot load manifest: {err}")
    raise SystemExit(1)

issues = {"CRITICAL": [], "WARNING": [], "INFO": []}

def crit(msg):  issues["CRITICAL"].append(msg)
def warn(msg):  issues["WARNING"].append(msg)
def info(msg):  issues["INFO"].append(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — TMDB ID correctness
# ═══════════════════════════════════════════════════════════════════════════════
print("Running Check 1: TMDB ID correctness...")

for filename, entry in manifest.items():
    slug = entry["slug"]
    manifest_title = entry["title"]
    tmdb_id = entry["tmdb_id"]

    meta_path = os.path.join(OUTPUT_DIR, f"{slug}_metadata.json")
    meta, err = safe_load(meta_path)
    if err or meta is None:
        crit(f"[TMDB] Missing metadata file for slug '{slug}' (manifest title: '{manifest_title}')")
        continue

    meta_title = meta.get("title", "")
    meta_year_raw = meta.get("release_date", "")
    meta_year = None
    if meta_year_raw:
        try:
            meta_year = int(meta_year_raw[:4])
        except:
            pass

    # Title mismatch check (normalize for comparison)
    def normalize(t):
        return re.sub(r"[^a-z0-9 ]", "", t.lower().strip())

    norm_manifest = normalize(manifest_title)
    norm_meta = normalize(meta_title)

    # Check if manifest title words appear in meta title
    manifest_words = set(norm_manifest.split())
    meta_words = set(norm_meta.split())
    overlap = manifest_words & meta_words
    # Remove stop words for overlap check
    stopwords = {"the", "of", "and", "a", "an", "in", "to", "at", "on"}
    meaningful_manifest = manifest_words - stopwords
    meaningful_overlap = overlap - stopwords

    if meaningful_manifest and len(meaningful_overlap) / len(meaningful_manifest) < 0.4:
        crit(f"[TMDB] Title mismatch — manifest: '{manifest_title}' vs metadata title: '{meta_title}' (tmdb_id={tmdb_id}, slug={slug})")
    elif norm_manifest != norm_meta:
        info(f"[TMDB] Minor title difference — manifest: '{manifest_title}' vs metadata: '{meta_title}' (slug={slug})")

    # Year check from filename
    year_match = re.search(r"[^a-zA-Z](\d{4})[^a-zA-Z]", filename)
    if not year_match:
        # try end of filename before extension
        year_match = re.search(r"(\d{4})\.", filename)
    if year_match and meta_year:
        file_year = int(year_match.group(1))
        if abs(file_year - meta_year) > 1:
            warn(f"[TMDB] Year mismatch — filename year {file_year} vs metadata release year {meta_year} (slug={slug}, title='{manifest_title}')")

    # Flag generic titles
    generic_titles = ["2012", "nosferatu", "wicked", "f1", "sinners"]
    if normalize(manifest_title) in generic_titles:
        info(f"[TMDB] Generic/ambiguous title '{manifest_title}' — metadata title: '{meta_title}', release: {meta_year_raw} (slug={slug}, tmdb_id={tmdb_id})")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Scene parsing quality
# ═══════════════════════════════════════════════════════════════════════════════
print("Running Check 2: Scene parsing quality...")

scene_files = glob.glob(os.path.join(OUTPUT_DIR, "*_scenes.json"))

for spath in sorted(scene_files):
    slug = os.path.basename(spath).replace("_scenes.json", "")
    scenes, err = safe_load(spath)
    if err or scenes is None:
        crit(f"[SCENES] Cannot load {os.path.basename(spath)}: {err}")
        continue

    scene_count = len(scenes)
    if scene_count == 0:
        crit(f"[SCENES] '{slug}': 0 scenes parsed")
        continue

    # Dialogue count across all scenes
    total_dialogue = sum(len(s.get("dialogue", [])) for s in scenes)
    if total_dialogue == 0:
        warn(f"[SCENES] '{slug}': {scene_count} scenes but 0 dialogue entries total")

    # Average word count
    word_counts = [s.get("word_count", 0) for s in scenes]
    avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0
    if avg_wc < 10:
        warn(f"[SCENES] '{slug}': very low avg word_count per scene ({avg_wc:.1f})")

    # Empty scenes (no action AND no dialogue)
    empty_scenes = [
        s for s in scenes
        if len(s.get("action_lines", [])) == 0 and len(s.get("dialogue", [])) == 0
    ]
    empty_pct = len(empty_scenes) / scene_count * 100
    if empty_pct > 20:
        warn(f"[SCENES] '{slug}': {len(empty_scenes)}/{scene_count} scenes ({empty_pct:.0f}%) are completely empty (no action, no dialogue)")
    elif len(empty_scenes) > 0:
        info(f"[SCENES] '{slug}': {len(empty_scenes)}/{scene_count} scenes ({empty_pct:.0f}%) are empty")

    # Sample mid-film scene
    mid_idx = scene_count // 2
    mid_scene = scenes[mid_idx]
    action_sample = mid_scene.get("action_lines", [])[:2]
    dialogue_sample = mid_scene.get("dialogue", [])[:2]

    # Heuristic: check if action lines look like noise (all caps, very short, etc.)
    for al in action_sample:
        text = al if isinstance(al, str) else str(al)
        if len(text.strip()) < 3:
            info(f"[SCENES] '{slug}': mid-scene action line suspiciously short: {repr(text)}")

    # Check for scenes with implausibly high dialogue
    scene_with_lots_of_dialogue = max(scenes, key=lambda s: len(s.get("dialogue", [])))
    max_dial = len(scene_with_lots_of_dialogue.get("dialogue", []))
    if max_dial > 300:
        warn(f"[SCENES] '{slug}': one scene has {max_dial} dialogue entries — may indicate scene boundary parsing failure")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Metadata completeness
# ═══════════════════════════════════════════════════════════════════════════════
print("Running Check 3: Metadata completeness...")

meta_files = glob.glob(os.path.join(OUTPUT_DIR, "*_metadata.json"))

for mpath in sorted(meta_files):
    slug = os.path.basename(mpath).replace("_metadata.json", "")
    meta, err = safe_load(mpath)
    if err or meta is None:
        crit(f"[META] Cannot load {os.path.basename(mpath)}: {err}")
        continue

    problems = []
    if not meta.get("overview", "").strip():
        problems.append("missing overview")
    directors = meta.get("directors", meta.get("director", []))
    if isinstance(directors, str):
        directors = [directors] if directors else []
    if not directors:
        problems.append("no directors")
    cast = meta.get("cast", [])
    if not cast:
        problems.append("0 cast members")
    genres = meta.get("genres", [])
    if not genres:
        problems.append("no genres")
    poster = meta.get("poster_url", meta.get("poster_path", ""))
    if not poster:
        problems.append("no poster_url")

    if problems:
        severity = "CRITICAL" if "0 cast members" in problems or "missing overview" in problems else "WARNING"
        msg = f"[META] '{slug}': {', '.join(problems)}"
        if severity == "CRITICAL":
            crit(msg)
        else:
            warn(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Character reconciliation match rate
# ═══════════════════════════════════════════════════════════════════════════════
print("Running Check 4: Character reconciliation...")

char_files = glob.glob(os.path.join(OUTPUT_DIR, "*_characters.json"))

for cpath in sorted(char_files):
    slug = os.path.basename(cpath).replace("_characters.json", "")
    chars, err = safe_load(cpath)
    if err or chars is None:
        crit(f"[CHARS] Cannot load {os.path.basename(cpath)}: {err}")
        continue

    # Determine structure
    # Could be a list or a dict with different shapes
    if isinstance(chars, dict):
        # Check common keys
        if "characters" in chars:
            char_list = chars["characters"]
        elif "matched" in chars or "unmatched" in chars:
            matched_list = chars.get("matched", [])
            unmatched_list = chars.get("unmatched", [])
            total = len(matched_list) + len(unmatched_list)
            match_rate = len(matched_list) / total * 100 if total > 0 else 0
            if match_rate < 40 and total > 5:
                # Find top unmatched by frequency
                unmatched_names = []
                for u in unmatched_list:
                    if isinstance(u, dict):
                        name = u.get("name", u.get("screenplay_name", str(u)))
                        freq = u.get("count", u.get("frequency", u.get("scene_count", 1)))
                    else:
                        name = str(u)
                        freq = 1
                    unmatched_names.append((name, freq))
                top5 = sorted(unmatched_names, key=lambda x: -x[1])[:5]
                warn(f"[CHARS] '{slug}': match rate {match_rate:.0f}% ({len(matched_list)}/{total}) — top unmatched: {top5}")
            elif match_rate < 40:
                info(f"[CHARS] '{slug}': match rate {match_rate:.0f}% ({len(matched_list)}/{total}) but only {total} total characters")
            continue
        else:
            char_list = list(chars.values()) if chars else []
    elif isinstance(chars, list):
        char_list = chars
    else:
        warn(f"[CHARS] '{slug}': unexpected structure type {type(chars)}")
        continue

    if not char_list:
        warn(f"[CHARS] '{slug}': character list is empty")
        continue

    # Try to compute match rate from list
    total = len(char_list)
    matched = 0
    unmatched_items = []

    for c in char_list:
        if isinstance(c, dict):
            # Check various possible matched indicators
            is_matched = (
                c.get("tmdb_match") is not None or
                c.get("matched", False) or
                c.get("tmdb_id") is not None or
                c.get("tmdb_character") is not None or
                (c.get("cast_match") is not None and c.get("cast_match") != "")
            )
            if is_matched:
                matched += 1
            else:
                name = c.get("name", c.get("screenplay_name", c.get("character", str(c))))
                freq = c.get("count", c.get("frequency", c.get("scene_count", c.get("dialogue_count", 1))))
                unmatched_items.append((name, freq))
        else:
            # Can't determine match status from non-dict
            matched += 1  # assume matched

    if total == 0:
        continue

    match_rate = matched / total * 100
    if match_rate < 40 and total > 5:
        top5 = sorted(unmatched_items, key=lambda x: -x[1])[:5]
        warn(f"[CHARS] '{slug}': match rate {match_rate:.0f}% ({matched}/{total}) — top unmatched: {top5}")
    elif total <= 5:
        info(f"[CHARS] '{slug}': only {total} characters total (possible sparse parse)")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Duplicate detection
# ═══════════════════════════════════════════════════════════════════════════════
print("Running Check 5: Duplicate detection...")

tmdb_id_map = defaultdict(list)
slug_map = defaultdict(list)

for filename, entry in manifest.items():
    tmdb_id_map[entry["tmdb_id"]].append((filename, entry["title"]))
    slug_map[entry["slug"]].append((filename, entry["title"]))

for tmdb_id, entries in tmdb_id_map.items():
    if len(entries) > 1:
        crit(f"[DUPES] Duplicate tmdb_id={tmdb_id}: {entries}")

for slug, entries in slug_map.items():
    if len(entries) > 1:
        crit(f"[DUPES] Duplicate slug='{slug}': {entries}")

# Also check if any manifest entry has no corresponding output files at all
print("Running Check: Missing output files...")
for filename, entry in manifest.items():
    slug = entry["slug"]
    missing = []
    for suffix in ["_scenes.json", "_metadata.json", "_characters.json"]:
        fpath = os.path.join(OUTPUT_DIR, f"{slug}{suffix}")
        if not os.path.exists(fpath):
            missing.append(suffix)
    if missing:
        crit(f"[FILES] '{slug}' ('{entry['title']}'): missing output files: {missing}")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT CONSOLIDATED REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 80)
print("QUALITY AUDIT REPORT — CONSOLIDATED ISSUES ONLY")
print("=" * 80)

total_issues = sum(len(v) for v in issues.values())
if total_issues == 0:
    print("\nNo issues found. All checks passed.")
else:
    for severity in ["CRITICAL", "WARNING", "INFO"]:
        items = issues[severity]
        if items:
            print(f"\n{'-'*80}")
            print(f"  {severity} ({len(items)} issue{'s' if len(items)>1 else ''})")
            print(f"{'-'*80}")
            for item in items:
                print(f"  * {item}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(issues['CRITICAL'])} CRITICAL  |  {len(issues['WARNING'])} WARNING  |  {len(issues['INFO'])} INFO")
    print(f"{'='*80}")

# ─── Extra: print sample scene data for flagged movies ────────────────────────
print("\n\n=== SAMPLE SCENE DATA (mid-film scene from each movie, for eyeball check) ===\n")
for spath in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_scenes.json")))[:5]:
    slug = os.path.basename(spath).replace("_scenes.json", "")
    scenes, _ = safe_load(spath)
    if not scenes:
        continue
    mid = scenes[len(scenes) // 2]
    print(f"--- {slug} (scene {len(scenes)//2}/{len(scenes)}) ---")
    print(f"  heading: {mid.get('heading', mid.get('scene_heading', ''))}")
    actions = mid.get("action_lines", [])
    print(f"  action[0]: {str(actions[0])[:120] if actions else '(none)'}")
    print(f"  action[1]: {str(actions[1])[:120] if len(actions)>1 else '(none)'}")
    dialogues = mid.get("dialogue", [])
    d0 = dialogues[0] if dialogues else None
    d1 = dialogues[1] if len(dialogues) > 1 else None
    if d0:
        spkr = d0.get("character", d0.get("speaker", "?"))
        line = d0.get("line", d0.get("text", d0.get("dialogue", "")))
        print(f"  dialogue[0]: {spkr}: {str(line)[:100]}")
    else:
        print(f"  dialogue[0]: (none)")
    if d1:
        spkr = d1.get("character", d1.get("speaker", "?"))
        line = d1.get("line", d1.get("text", d1.get("dialogue", "")))
        print(f"  dialogue[1]: {spkr}: {str(line)[:100]}")
    else:
        print(f"  dialogue[1]: (none)")
    print()

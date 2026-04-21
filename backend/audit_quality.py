import json
import os
import glob
import re
from collections import defaultdict, Counter
from pathlib import Path

ROOT         = Path(__file__).parent.resolve()
OUTPUT_DIR   = str(ROOT / "output")
MANIFEST_PATH = str(ROOT / "screenplays" / "manifest.json")


def safe_load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


manifest, err = safe_load(MANIFEST_PATH)
if err:
    print(f"CRITICAL: Cannot load manifest: {err}")
    raise SystemExit(1)

issues = {"CRITICAL": [], "WARNING": [], "INFO": []}

def crit(msg): issues["CRITICAL"].append(msg)
def warn(msg): issues["WARNING"].append(msg)
def info(msg): issues["INFO"].append(msg)


# CHECK 1 — TMDB ID correctness
print("Running Check 1: TMDB ID correctness...")

for filename, entry in manifest.items():
    slug           = entry["slug"]
    manifest_title = entry["title"]
    tmdb_id        = entry["tmdb_id"]

    meta, err = safe_load(os.path.join(OUTPUT_DIR, f"{slug}_metadata.json"))
    if err or meta is None:
        crit(f"[TMDB] Missing metadata for slug '{slug}' (manifest title: '{manifest_title}')")
        continue

    meta_title   = meta.get("title", "")
    meta_year_raw = meta.get("release_date", "")
    meta_year    = None
    if meta_year_raw:
        try:
            meta_year = int(meta_year_raw[:4])
        except Exception:
            pass

    def normalize(t):
        return re.sub(r"[^a-z0-9 ]", "", t.lower().strip())

    norm_manifest = normalize(manifest_title)
    norm_meta     = normalize(meta_title)

    stopwords         = {"the", "of", "and", "a", "an", "in", "to", "at", "on"}
    manifest_words    = set(norm_manifest.split())
    meta_words        = set(norm_meta.split())
    overlap           = manifest_words & meta_words
    meaningful_manifest = manifest_words - stopwords
    meaningful_overlap  = overlap - stopwords

    if meaningful_manifest and len(meaningful_overlap) / len(meaningful_manifest) < 0.4:
        crit(f"[TMDB] Title mismatch — manifest: '{manifest_title}' vs metadata: '{meta_title}' (tmdb_id={tmdb_id}, slug={slug})")
    elif norm_manifest != norm_meta:
        info(f"[TMDB] Minor title difference — manifest: '{manifest_title}' vs metadata: '{meta_title}' (slug={slug})")

    year_match = re.search(r"[^a-zA-Z](\d{4})[^a-zA-Z]", filename) or re.search(r"(\d{4})\.", filename)
    if year_match and meta_year:
        file_year = int(year_match.group(1))
        if abs(file_year - meta_year) > 1:
            warn(f"[TMDB] Year mismatch — filename {file_year} vs metadata {meta_year} (slug={slug})")

    generic_titles = ["2012", "nosferatu", "wicked", "f1", "sinners"]
    if normalize(manifest_title) in generic_titles:
        info(f"[TMDB] Ambiguous title '{manifest_title}' — metadata: '{meta_title}', release: {meta_year_raw} (slug={slug})")


# CHECK 2 — Scene parsing quality
print("Running Check 2: Scene parsing quality...")

for spath in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_scenes.json"))):
    slug   = os.path.basename(spath).replace("_scenes.json", "")
    scenes, err = safe_load(spath)
    if err or scenes is None:
        crit(f"[SCENES] Cannot load {os.path.basename(spath)}: {err}")
        continue

    scene_count = len(scenes)
    if scene_count == 0:
        crit(f"[SCENES] '{slug}': 0 scenes parsed")
        continue

    total_dialogue = sum(len(s.get("dialogue", [])) for s in scenes)
    if total_dialogue == 0:
        warn(f"[SCENES] '{slug}': {scene_count} scenes but 0 dialogue entries")

    word_counts = [s.get("word_count", 0) for s in scenes]
    avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0
    if avg_wc < 10:
        warn(f"[SCENES] '{slug}': very low avg word count per scene ({avg_wc:.1f})")

    empty_scenes = [s for s in scenes if not s.get("action_lines") and not s.get("dialogue")]
    empty_pct    = len(empty_scenes) / scene_count * 100
    if empty_pct > 20:
        warn(f"[SCENES] '{slug}': {len(empty_scenes)}/{scene_count} scenes ({empty_pct:.0f}%) are empty")
    elif empty_scenes:
        info(f"[SCENES] '{slug}': {len(empty_scenes)}/{scene_count} scenes ({empty_pct:.0f}%) are empty")

    max_dial = max((len(s.get("dialogue", [])) for s in scenes), default=0)
    if max_dial > 300:
        warn(f"[SCENES] '{slug}': one scene has {max_dial} dialogue entries — possible scene boundary failure")


# CHECK 3 — Metadata completeness
print("Running Check 3: Metadata completeness...")

for mpath in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_metadata.json"))):
    slug       = os.path.basename(mpath).replace("_metadata.json", "")
    meta, err  = safe_load(mpath)
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
    if not meta.get("cast"):
        problems.append("0 cast members")
    if not meta.get("genres"):
        problems.append("no genres")
    if not meta.get("poster_url", meta.get("poster_path", "")):
        problems.append("no poster_url")

    if problems:
        severity = "CRITICAL" if ("0 cast members" in problems or "missing overview" in problems) else "WARNING"
        msg = f"[META] '{slug}': {', '.join(problems)}"
        (crit if severity == "CRITICAL" else warn)(msg)


# CHECK 4 — Character reconciliation match rate
print("Running Check 4: Character reconciliation...")

for cpath in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_characters.json"))):
    slug       = os.path.basename(cpath).replace("_characters.json", "")
    chars, err = safe_load(cpath)
    if err or chars is None:
        crit(f"[CHARS] Cannot load {os.path.basename(cpath)}: {err}")
        continue

    if isinstance(chars, dict):
        char_list = chars.get("characters", list(chars.values()) if chars else [])
    elif isinstance(chars, list):
        char_list = chars
    else:
        warn(f"[CHARS] '{slug}': unexpected structure type {type(chars)}")
        continue

    if not char_list:
        warn(f"[CHARS] '{slug}': character list is empty")
        continue

    total      = len(char_list)
    matched    = 0
    unmatched_items = []

    for c in char_list:
        if not isinstance(c, dict):
            matched += 1
            continue
        is_matched = (
            c.get("match_type", "unmatched") != "unmatched" or
            c.get("tmdb_character") is not None
        )
        if is_matched:
            matched += 1
        else:
            name = c.get("screenplay_name", c.get("name", str(c)))
            freq = c.get("scene_count", c.get("count", 1))
            unmatched_items.append((name, freq))

    match_rate = matched / total * 100
    if match_rate < 40 and total > 5:
        top5 = sorted(unmatched_items, key=lambda x: -x[1])[:5]
        warn(f"[CHARS] '{slug}': match rate {match_rate:.0f}% ({matched}/{total}) — top unmatched: {top5}")
    elif total <= 5:
        info(f"[CHARS] '{slug}': only {total} characters (possible sparse parse)")


# CHECK 5 — Duplicate detection
print("Running Check 5: Duplicate detection...")

tmdb_id_map = defaultdict(list)
slug_map    = defaultdict(list)

for filename, entry in manifest.items():
    tmdb_id_map[entry["tmdb_id"]].append((filename, entry["title"]))
    slug_map[entry["slug"]].append((filename, entry["title"]))

for tmdb_id, entries in tmdb_id_map.items():
    if len(entries) > 1:
        crit(f"[DUPES] Duplicate tmdb_id={tmdb_id}: {entries}")

for slug, entries in slug_map.items():
    if len(entries) > 1:
        crit(f"[DUPES] Duplicate slug='{slug}': {entries}")

print("Running Check: Missing output files...")
for filename, entry in manifest.items():
    slug    = entry["slug"]
    missing = [s for s in ("_scenes.json", "_metadata.json", "_characters.json")
               if not os.path.exists(os.path.join(OUTPUT_DIR, f"{slug}{s}"))]
    if missing:
        crit(f"[FILES] '{slug}' ('{entry['title']}'): missing output files: {missing}")


# Report
print("\n")
print("=" * 80)
print("QUALITY AUDIT REPORT")
print("=" * 80)

total_issues = sum(len(v) for v in issues.values())
if total_issues == 0:
    print("\nNo issues found.")
else:
    for severity in ("CRITICAL", "WARNING", "INFO"):
        items = issues[severity]
        if items:
            print(f"\n{'-'*80}")
            print(f"  {severity} ({len(items)} issue{'s' if len(items) > 1 else ''})")
            print(f"{'-'*80}")
            for item in items:
                print(f"  * {item}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(issues['CRITICAL'])} CRITICAL  |  {len(issues['WARNING'])} WARNING  |  {len(issues['INFO'])} INFO")
    print(f"{'='*80}")


# Sample scenes for a quick eyeball check
print("\n\n=== SAMPLE SCENE DATA (mid-film scene, first 5 movies) ===\n")
for spath in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_scenes.json")))[:5]:
    slug   = os.path.basename(spath).replace("_scenes.json", "")
    scenes, _ = safe_load(spath)
    if not scenes:
        continue
    mid = scenes[len(scenes) // 2]
    print(f"--- {slug} (scene {len(scenes)//2}/{len(scenes)}) ---")
    print(f"  heading: {mid.get('heading', '')}")
    actions = mid.get("action_lines", [])
    print(f"  action[0]: {str(actions[0])[:120] if actions else '(none)'}")
    print(f"  action[1]: {str(actions[1])[:120] if len(actions) > 1 else '(none)'}")
    dialogues = mid.get("dialogue", [])
    for i, d in enumerate(dialogues[:2]):
        spkr = d.get("character", "?")
        line = d.get("line", "")
        print(f"  dialogue[{i}]: {spkr}: {str(line)[:100]}")
    if not dialogues:
        print("  dialogue[0]: (none)")
    print()

#!/usr/bin/env bash
# Verify, move and unpack VTUAD v1.3.2 (audio + metadata) after the Chrome downloads finish.
# Usage (macOS Terminal): copy into the folder that should hold the data, then
#   bash <that folder>/setup_vtuad.sh      (SRC=<download dir> overrides ~/Downloads)
# Source: VTUAD v1.3.2, Yuan et al., doi:10.5281/zenodo.23037895 (CC BY 4.0), from ONC Oceans 3.0.
set -u
SRC="${SRC:-$HOME/Downloads}"
DST="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DST/raw" "$DST/wav" "$DST/meta"

md5_of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }

FILES="
CITATION.cff c949821a32d2755d063b45e60c39d9bf
clip_acoustic_stats.csv 2aa7f471aeadd2b543b3980eb968a5a4
cut_rule_simulation_525clips.csv ded9fe0b66d52a2d55dffe0f2268adf6
dataset_card.md 761d13ef032d66dbc2e54ad0bcf5dc53
environment.yml 1f2fa0de81f4ddee3ef1a43df9ad4b25
grouping_defect_3day_audit.csv c59342205eab5e7e178892640d9966c4
interval_audit_readme.md 0b01439aa272bee6a5968c02c3be976d
interval_uid_closure_v1deployment.csv cf01683a796aa135eb2ad469bafc6295
license_evidence_v131.md 9bbc4f36cedc06e5aac938135c2a25c1
metadata.csv 3d597065eab6a9400a8e84eea1ae8cd2
mmsi_stability_20160908.csv bb2454c063de228dbf4f51837aa62aef
README_vtuad_strict_wav_vessel_parts.txt 43bab1eba183cdfac74130cdfa135620
registry_anonymized.csv f58655efd07502583b4b2cf5ca2bc5de
registry_verification_sample30.csv 9c0a8e28866058561434fc8189e1dd73
registry_verification_sample30_README.md aa60894a148b94a0fb9001c4436dfe77
SHA256SUMS.txt c265c4a8b3c172fed87efde79cc5fa29
static_audit_README.md c7fa02fd4cadeb683bca69bb337ee5c3
static_field_conflicts.csv 8b1a0cc09481243d08bf72d4f66ad3b4
static_field_fill_audit.csv e1fc933f13a6ef3dc83b2c1de1c90a65
strict_adjacency_intervals_3day_sample.csv d1b41b6b07013412e8a8af1704f988f8
strict_interval_wav_SHA256SUMS.txt 3c19f42c80a9c5979c15c4ad1abc9c67
v131_supplementary_SHA256SUMS.txt 9d1502a89b56cd555f12eae14b8b779e
vtuad_wraparound_splice_clips_v1.csv ca6c17427fb0bd1a3f47ebce667ad9f3
vtuad_wraparound_splice_clips_v1_README.md b9ba92ced35d22b78851788d71c4bbd8
wrap_feature_recompute_check.csv 38892842b9c8850958a6df76670d558c
zenodo_vtuad_v1.2.0.zip 1a20665cb43946386aed6e0162e2306d
vtuad_strict_wav_background.zip d28ad0fb6ccc98341ba4f7e953d9c68f
vtuad_strict_wav_vessel.zip.part01 6e3fe8245218faa4f94868531e272344
vtuad_strict_wav_vessel.zip.part02 af6408581f6981f45ffdbce811e9f558
vtuad_strict_wav_vessel.zip.part03 dfa59398483a651765a6265df76febad
vtuad_wav_background.zip 8d78d71525a4a2db047894b6cd19787a
vtuad_wav_cargo.zip 1532ccf2d7db6b69165a70a6b62d0e19
vtuad_wav_dredger.zip 4d943970f39eff2097633f1660121284
vtuad_wav_fishing.zip 9a7943b1dc86f8af4d5d0828b9073e22
vtuad_wav_other.zip 4b5d767bb7f79f07914aa9952a836fab
vtuad_wav_passengership.zip 5adbf5d145f083e1105173d3b6a34cdf
vtuad_wav_pilotvessel.zip 0c625f6782bd77d754ea98127d60237d
vtuad_wav_pleasurecraft.zip 27f4074ed73dc71f48cc32f19c66a401
vtuad_wav_rescue.zip 7e267c5efa0e9321191943fac78cd343
vtuad_wav_sailing.zip 464cd6ecfac6ab62ebf522fe9284ba91
vtuad_wav_tanker.zip cac0820693b3b5db3ba46cae5b460d52
vtuad_wav_tug.zip f60f56537ee9ae801f7fdc00220a6f0d
"

ok=0; bad=0; missing=0
while read -r name sum; do
  [ -z "$name" ] && continue
  target="$DST/raw/$name"
  if [ ! -f "$target" ]; then
    # Chrome may still be writing (.crdownload) or may have renamed duplicates "name (1).ext".
    cand=""
    for f in "$SRC/$name" "$SRC/${name%.*} (1).${name##*.}"; do [ -f "$f" ] && cand="$f" && break; done
    if [ -z "$cand" ]; then
      if ls "$SRC/$name"*.crdownload >/dev/null 2>&1; then echo "IN PROGRESS  $name"; else echo "MISSING      $name"; fi
      missing=$((missing+1)); continue
    fi
    mv "$cand" "$target"
  fi
  if [ "$(md5_of "$target")" = "$sum" ]; then ok=$((ok+1)); else echo "BAD CHECKSUM $name (delete it and re-download)"; bad=$((bad+1)); fi
done <<< "$FILES"
echo "verified: $ok   bad: $bad   missing/in progress: $missing   (of 42)"

if [ "$bad" -ne 0 ] || [ "$missing" -ne 0 ]; then
  echo "Not unpacking yet - rerun this script once every file is verified."; exit 1
fi

# Rebuild the byte-split strict vessel archive.
if [ ! -f "$DST/raw/vtuad_strict_wav_vessel.zip" ]; then
  cat "$DST"/raw/vtuad_strict_wav_vessel.zip.part0{1,2,3} > "$DST/raw/vtuad_strict_wav_vessel.zip"
fi
want=$(grep -h "vtuad_strict_wav_vessel.zip$" "$DST"/raw/*SHA256SUMS*.txt "$DST/raw/README_vtuad_strict_wav_vessel_parts.txt" 2>/dev/null | grep -oE '[0-9a-f]{64}' | head -1)
if [ -n "$want" ]; then
  got=$(shasum -a 256 "$DST/raw/vtuad_strict_wav_vessel.zip" | cut -d' ' -f1)
  [ "$got" = "$want" ] && echo "strict vessel archive: SHA-256 OK" || { echo "strict vessel archive: SHA-256 MISMATCH"; exit 1; }
else
  echo "strict vessel archive: no whole-file SHA-256 found in the docs; part MD5s were verified"
fi

# Unpack audio; copy small metadata files next to it.
for z in "$DST"/raw/vtuad_wav_*.zip; do
  d="$DST/wav/containers/$(basename "$z" .zip | sed 's/^vtuad_wav_//')"
  [ -d "$d" ] || { mkdir -p "$d" && unzip -q "$z" -d "$d" && echo "unzipped $(basename "$z")"; }
done
for z in vtuad_strict_wav_vessel vtuad_strict_wav_background; do
  d="$DST/wav/strict/${z#vtuad_strict_wav_}"
  [ -d "$d" ] || { mkdir -p "$d" && unzip -q "$DST/raw/$z.zip" -d "$d" && echo "unzipped $z.zip"; }
done
cp -n "$DST"/raw/*.csv "$DST"/raw/*.md "$DST"/raw/*.txt "$DST"/raw/*.cff "$DST"/raw/*.yml "$DST/meta/" 2>/dev/null
echo "WAV files: $(find "$DST/wav" -name '*.wav' | wc -l | tr -d ' ')  (expect 525 strict + 525 containers)"
echo "Done. Zips are kept in $DST/raw; delete them once the WAVs check out to free ~30 GB."

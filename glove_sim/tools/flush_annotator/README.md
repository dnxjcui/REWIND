# Flush Annotator

Lightweight local three.js app to manually annotate flush regions on hand and glove GLBs.

## Run

From repository root:

```bash
python -m http.server 8080
```

Open:

- `http://localhost:8080/glove_sim/tools/flush_annotator/`

## Workflow

1. Load hand frame GLB and glove frame GLB.
2. Select `Target: Hand` and click 3+ points on dorsum patch.
3. Select `Target: Glove` and click corresponding 3+ points on glove-base underside.
4. Set `Reference frame idx` to the frame used for those GLBs.
5. Download `annotations.json` and copy to:
   - `glove_sim/outputs/knot_alignment/annotations.json`

## Alignment usage

```bash
python glove_sim/knot_alignment.py --annotations glove_sim/outputs/knot_alignment/annotations.json --max-frames 6
```

The script validates schema and enforces connectivity before running alignment.

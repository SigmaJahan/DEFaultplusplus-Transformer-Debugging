from __future__ import annotations
import argparse
from pathlib import Path

from .ndg import build_ndg, build_schema, load_json
from .plotting import build_publication_plots

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_detection", type=Path, required=True)
    ap.add_argument("--enc_categorization", type=Path, required=True)
    ap.add_argument("--xai_enc", type=Path, required=True)
    ap.add_argument("--dec_detection", type=Path, required=True)
    ap.add_argument("--dec_categorization", type=Path, required=True)
    ap.add_argument("--xai_dec", type=Path, required=True)
    ap.add_argument("--feature_core_map", type=Path, required=True)
    ap.add_argument("--enc_diagnosis", type=Path, default=None)
    ap.add_argument("--dec_diagnosis", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--top_confusions", type=int, default=2)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # schema
    schema = build_schema([
        args.enc_detection.name, args.enc_categorization.name, args.xai_enc.name,
        args.dec_detection.name, args.dec_categorization.name, args.xai_dec.name,
        args.feature_core_map.name,
        *( [args.enc_diagnosis.name] if args.enc_diagnosis else [] ),
        *( [args.dec_diagnosis.name] if args.dec_diagnosis else [] ),
    ])
    (args.out_dir/"ndg_schema.json").write_text(__import__("json").dumps(schema, indent=2))

    enc_out = args.out_dir/"ndg_encoder.json"
    dec_out = args.out_dir/"ndg_decoder.json"

    build_ndg(
        detection_path=args.enc_detection,
        categorization_path=args.enc_categorization,
        xai_path=args.xai_enc,
        out_path=enc_out,
        feature_core_map_path=args.feature_core_map,
        diagnosis_path=args.enc_diagnosis,
        top_confusions=args.top_confusions,
    )
    build_ndg(
        detection_path=args.dec_detection,
        categorization_path=args.dec_categorization,
        xai_path=args.xai_dec,
        out_path=dec_out,
        feature_core_map_path=args.feature_core_map,
        diagnosis_path=args.dec_diagnosis,
        top_confusions=args.top_confusions,
    )

    if args.plots:
        plot_dir = args.out_dir/"plots"
        build_publication_plots(enc_out, plot_dir, prefix="encoder")
        build_publication_plots(dec_out, plot_dir, prefix="decoder")

    # provenance
    prov = []
    prov.append("# NDG Stage-3 Provenance\n\n")
    prov.append("## Inputs\n")
    for p in [args.enc_detection,args.enc_categorization,args.xai_enc,args.dec_detection,args.dec_categorization,args.xai_dec,args.feature_core_map]:
        prov.append(f"- `{p.name}`\n")
    if args.enc_diagnosis: prov.append(f"- `{args.enc_diagnosis.name}`\n")
    if args.dec_diagnosis: prov.append(f"- `{args.dec_diagnosis.name}`\n")
    prov.append("\nOutputs:\n")
    prov.append(f"- `{enc_out.name}`\n- `{dec_out.name}`\n- `ndg_schema.json`\n")
    if args.plots:
        prov.append("- `plots/` (PDF summaries)\n")
    (args.out_dir/"ndg_provenance.md").write_text("".join(prov))

    print("Wrote NDG artifacts to", args.out_dir)

if __name__ == "__main__":
    main()

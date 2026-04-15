from __future__ import annotations

import argparse
import json

from cases import CASES
from contract_checks import evaluate_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distilled attention fault benchmarks.")
    parser.add_argument("--case", help="Run a single case by slug.", default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = CASES
    if args.case is not None:
        cases = [case for case in CASES if case.metadata.slug == args.case]
        if not cases:
            raise SystemExit(f"unknown case: {args.case}")

    results = []
    for case in cases:
        result = case.run()
        contract_eval = evaluate_contract(case.metadata.slug, result.details)
        passed = result.reproduced and contract_eval.passed
        payload = {
            "row_id": case.metadata.row_id,
            "slug": case.metadata.slug,
            "title": case.metadata.title,
            "issue_url": case.metadata.issue_url,
            "fix_url": case.metadata.fix_url,
            "fault_family": case.metadata.fault_family,
            "contract": contract_eval.to_dict(),
            "passed": passed,
            **result.to_dict(),
        }
        results.append(payload)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    width = max(len(r["slug"]) for r in results)
    ok = 0
    for item in results:
        status = "OK" if item["passed"] else "FAIL"
        ok += int(item["passed"])
        contract = item["contract"]
        flags = f"M={int(contract['mechanism']['passed'])} S={int(contract['symptom']['passed'])} C={int(contract['buggy_vs_fixed']['passed'])}"
        print(f"{status:<4} {item['slug']:<{width}} {item['summary']} [{flags}]")
    print(f"\nValidated {ok}/{len(results)} cases.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

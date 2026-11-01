"""Read-only source inventory; report hashes, review flags and missing coverage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from ingestion.validate_structured_data import build_report

ROOT=Path(__file__).resolve().parents[1]
def inventory():
    with (ROOT/"data/document_manifest.csv").open(encoding="utf-8-sig",newline="") as f:
        manifest=list(csv.DictReader(f))
    cleaned=ROOT/"data/processed/cleaned_run_4/cleaned_pages.jsonl"
    pages=[json.loads(x) for x in cleaned.read_text(encoding="utf-8").splitlines()] if cleaned.exists() else []
    docs=[]
    for row in manifest:
        path=ROOT/"data/raw_documents"/row["filename"]
        digest=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        extracted={p["source_sha256"] for p in pages if p["document_id"]==row["document_id"]}
        docs.append({"document_id":row["document_id"],"filename":str(path.relative_to(ROOT)),
            "exists":path.exists(),"sha256":digest,"extraction_hash_matches":bool(extracted) and extracted=={digest},
            "policy_status":row.get("status"),"access":row.get("access"),
            "redistribution_review_required": "intranet" in row.get("notes","").lower() or row.get("access") != "public",
            "applicability":"Not independently verified as latest applicable policy."})
    truth=json.loads((ROOT/"evaluation/ground_truth.json").read_text(encoding="utf-8"))
    relevant={(e["document_id"],p) for g in truth["questions"].values() for e in g["evidence"] for p in e["pages"]}
    flags=[{"document_id":p["document_id"],"page":p["page_number"],
            "review_fields":{k:v for k,v in p.items() if "review" in k},
            "benchmark_relevant":(p["document_id"],p["page_number"]) in relevant}
           for p in pages if p.get("needs_review") or p.get("review_reasons")]
    return {"documents":docs,"document_count":len(docs),"extracted_pages":len(pages),
            "all_paths_and_extraction_hashes_valid":all(d["exists"] and d["extraction_hash_matches"] for d in docs),
            "review_flagged_pages":flags,"structured":build_report("2026-27-S1"),
            "source_snapshot":"Existing full-v1 corpus retained; no source import/promotion performed.",
            "official_resource_check":"No applicable complete timetable established from existing authorized resources or public IITJ search. Intranet restrictions were not bypassed.",
            "historical_note":"Existing Policy sections are historical within revised circulars; supersession does not prove latest applicability."}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"reports/data_readiness_full_v2.json")
    a=p.parse_args();r=inventory();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"documents":r["document_count"],"hashes_valid":r["all_paths_and_extraction_hashes_valid"],"structured_ready":r["structured"]["ready"]}))
if __name__=="__main__":main()

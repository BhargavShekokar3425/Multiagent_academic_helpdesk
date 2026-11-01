"""Submission gates; do not confuse runnable code with external verification."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def verify(run_tests=False):
    checks=[]
    def check(name,passed,detail=""):checks.append({"check":name,"passed":bool(passed),"detail":detail})
    for directory in ("src","ingestion","evaluation","scripts","tests"):
        for p in (ROOT/directory).rglob("*.py"):
            ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))
    check("python_syntax",True)
    for name in ("src.agent","src.single_agent","src.orchestration","evaluation.evaluate",
                 "evaluation.compare_models","evaluation.export_results"):
        importlib.import_module(name)
    check("required_imports",True)
    from evaluation.scoring import load_benchmark
    load_benchmark();check("benchmark_sources",True)
    old=json.loads((ROOT/"reports/full_v1_preservation.json").read_text(encoding="utf-8"))
    check("full_v1_unchanged",all((ROOT/p).exists() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old.items()))
    from ingestion.validate_structured_data import build_report
    data=build_report("2026-27-S1");check("structured_integrity",data["structurally_valid"])
    for filename in ("README.md","reports/final_report.md","reports/final_report.pdf",
                     "reports/data_readiness_full_v2.json","reports/hardware_verification.json",
                     "evaluation/results/raw/full-v2/comparison.json"):
        check("artifact:"+filename,(ROOT/filename).is_file())
    selection_path=ROOT/"evaluation/results/raw/full-v2/final_selection.json"
    if selection_path.exists():
        selection=json.loads(selection_path.read_text(encoding="utf-8"))
        final=ROOT/selection["run_dir"]
        for mode in ("single","multi"):
            records=json.loads((final/("qwen3_8b-"+mode)/"records.json").read_text(encoding="utf-8"))
            check("final_records:"+mode,len(records)==90 and len({(r["question_id"],r["repetition"]) for r in records})==90)
        check("final_selection",True,selection["run_dir"])
    else:
        check("final_selection",False)
    pdf=ROOT/"reports/final_report.pdf"
    if pdf.exists():
        import pymupdf
        with pymupdf.open(pdf) as doc:
            check("report_pdf_content",len(doc)>0 and sum(len(p.get_text()) for p in doc)>3000)
    from src.single_agent import build_agent
    from src.orchestration import build_orchestrator
    check("both_architectures_construct",bool(build_agent()) and bool(build_orchestrator()))
    staged=subprocess.run(["git","diff","--cached","--name-only"],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",check=True).stdout.splitlines()
    risky=[p for p in staged if (p.startswith(("data/raw_documents/","data/vector_store/","evaluation/results/raw/"))
           or Path(p).name.startswith(".env") and Path(p).name!=".env.example"
           or Path(p).suffix in (".sqlite",".sqlite3",".db",".gguf",".safetensors"))]
    check("staged_sensitive_path_check",not risky,"Paths only: "+", ".join(risky))
    import re
    staged_diff=subprocess.run(["git","diff","--cached","--no-ext-diff","--unified=0"],cwd=ROOT,
        capture_output=True,text=True,encoding="utf-8",check=True).stdout
    suspect=bool(re.search(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{24,}",staged_diff))
    check("staged_credential_pattern_check",not suspect,"No credential values are printed.")
    if run_tests:
        r=subprocess.run([sys.executable,"-m","pytest","tests","-q","-p","no:cacheprovider"],cwd=ROOT,
            capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","LITELLM_LOCAL_MODEL_COST_MAP":"True"})
        check("full_pytest",r.returncode==0,r.stdout[-2000:]+r.stderr[-1000:])
    return {"code_gates_passed":all(c["passed"] for c in checks),"checks":checks,
            "submission_status":"CODE COMPLETE — EXTERNAL VERIFICATION PENDING" if all(c["passed"] for c in checks) else "PARTIALLY COMPLETE",
            "external_gates":["Independent human review","Windows 16 GB hardware verification","Llama comparison if required",
                              "Authorized complete timetable/offerings","Latest policy applicability verification"],
            "note":"Sensitive-path checks do not establish redistribution rights or scan unstaged private file contents. Manual source/license review remains required."}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--tests",action="store_true")
    p.add_argument("--output",type=Path,default=ROOT/"reports/submission_verification.json")
    a=p.parse_args();r=verify(a.tests);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8");print(json.dumps(r,indent=2))
    return 0 if r["code_gates_passed"] else 1
if __name__=="__main__":raise SystemExit(main())

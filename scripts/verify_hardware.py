"""Measure the current machine honestly; optional live isolated smoke and RSS sampling."""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]

def inspect_hardware(live=False):
    report={"platform":platform.platform(),"python":sys.version,"ram_gib":psutil.virtual_memory().total/1024**3,
            "is_windows":platform.system()=="Windows","target_hardware_verified":False,
            "live_smokes":[],"memory_note":"RSS sums may double-count shared pages and exclude GPU memory; GPU residency is reported separately."}
    if live:
        from evaluation.evaluate import ollama_request
        with tempfile.TemporaryDirectory(prefix="helpdesk-hardware-") as folder:
            for mode in ("single","multi"):
                target=Path(folder)/(mode+".json")
                code=("import asyncio,json;from pathlib import Path;from src.single_agent import answer_policy;"
                      "r=asyncio.run(answer_policy('What are minor and major exam durations and minimum Major examination weightage?',"
                      "mode="+repr(mode)+",extractive=True));Path("+repr(str(target))+").write_text(json.dumps(r),encoding='utf-8')")
                env={**os.environ,"DATABASE_PATH":str(Path(folder)/"review.sqlite3"),"PYTHONDONTWRITEBYTECODE":"1"}
                with (Path(folder)/(mode+".log")).open("w",encoding="utf-8") as log:
                    start=time.perf_counter()
                    process=subprocess.Popen([sys.executable,"-c",code],cwd=ROOT,env=env,stdout=log,stderr=log)
                    peak=0
                    while process.poll() is None:
                        if time.perf_counter()-start>240:
                            process.kill();process.wait()
                            raise TimeoutError("Hardware smoke timed out.")
                        try:
                            child=psutil.Process(process.pid)
                            processes=[child]+child.children(recursive=True)
                            processes += [p for p in psutil.process_iter(["name"]) if "ollama" in (p.info["name"] or "").lower()]
                            peak=max(peak,sum(p.memory_info().rss for p in {p.pid:p for p in processes}.values()))
                        except (psutil.NoSuchProcess,psutil.AccessDenied):
                            pass
                        time.sleep(.2)
                result=json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
                report["live_smokes"].append({"mode":mode,"exit_code":process.returncode,"status":result.get("status"),
                    "elapsed_seconds":time.perf_counter()-start,"peak_process_rss_gib_estimate":peak/1024**3,
                    "ollama_residency":ollama_request("/api/ps")})
        report["automated_target_smoke_passed"]=bool(report["is_windows"] and 15<=report["ram_gib"]<=17.5
            and all(r["status"]=="answer" and r["exit_code"]==0 for r in report["live_smokes"]))
        report["manual_target_review_required"]=True
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--live",action="store_true")
    p.add_argument("--output",type=Path,default=ROOT/"reports/hardware_verification.json")
    a=p.parse_args();r=inspect_hardware(a.live);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8");print(json.dumps(r,indent=2))
if __name__=="__main__":main()

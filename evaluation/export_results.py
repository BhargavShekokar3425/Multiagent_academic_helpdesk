"""Export real saved comparison metrics; never runs inference or changes grades."""
import argparse
import csv
import json
from pathlib import Path

def export_results(comparison, output):
    comparison, output = Path(comparison), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for label, folder in [("full-v1", next((parent / "full-v1" for parent in comparison.parents if (parent / "full-v1").is_dir()), comparison.parent / "full-v1")), ("full-v2", comparison)]:
        path = folder / "comparison_reviewed.json"
        if not path.exists():
            path = folder / "comparison.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            for c in report["configurations"]:
                if c["status"] != "completed":
                    continue
                m = c["metrics"]
                reports.append({"experiment":label, "model":c["model"], "mode":c["mode"],
                    "accuracy":m["answer_accuracy"]["value"], "graded":m["answer_accuracy"]["denominator"],
                    "eligible":m["answer_accuracy"]["eligible"], "pending":m["answer_accuracy"]["pending"],
                    "refusal":m["out_of_scope_refusal"]["value"],
                    "abstention":m["unsupported_abstention"]["value"],
                    "citation":m["citation_correctness"]["value"],
                    "groundedness":m["claim_groundedness"]["value"],
                    **{"latency_"+k:m["latency_seconds"][k] for k in ("mean","median","p95")}})
    with (output/"comparison.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(reports[0]) if reports else ["experiment"])
        writer.writeheader();writer.writerows(reports)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=[r["experiment"]+"\n"+r["model"]+"\n"+r["mode"] for r in reports]
    for field,title in [("accuracy","Graded answer accuracy (inspect pending counts)"),
                        ("latency_mean","Observed mean latency; v1 cold/warm conditions differ")]:
        fig,ax=plt.subplots(figsize=(10,5))
        values=[r[field] if r[field] is not None else 0 for r in reports]
        bars=ax.bar(labels,values)
        if field=="accuracy":
            ax.set_ylim(0,1.15)
            for b,r in zip(bars,reports):
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,
                    str(r["graded"])+"/"+str(r["eligible"])+" graded",ha="center",fontsize=8)
        ax.set_title(title);ax.set_ylabel("Fraction" if field=="accuracy" else "Seconds")
        fig.tight_layout();fig.savefig(output/(field+".png"),dpi=150);plt.close(fig)
    errors=[];stability=[]
    for folder in sorted(comparison.glob("*/")):
        path=folder/"records_graded.json"
        if not path.exists():path=folder/"records.json"
        if not path.exists():continue
        records=json.loads(path.read_text(encoding="utf-8"))
        for r in records:
            g=r["grading"]
            if g["accuracy"]==0 or (not g["accuracy_eligible"] and not g["expected_behavior_correct"]):
                errors.append({"question_id":r["question_id"],"mode":r["mode"],"repetition":r["repetition"],
                    "status":r["result"]["status"],"reason":r["result"].get("reason"),
                    "triage_label":"workflow_or_validation" if r["result"]["status"]!="answer" else "content_requires_review",
                    "note":"Triage only; not an independently verified root-cause label."})
        mp=folder/("metrics_graded.json" if (folder/"metrics_graded.json").exists() else "metrics.json")
        stability.extend(json.loads(mp.read_text(encoding="utf-8"))["stability"])
    for name,value in [("error_classification.json",errors),("stability_summary.json",stability)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return reports

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(export_results(a.comparison,a.output),indent=2))

if __name__=="__main__":main()

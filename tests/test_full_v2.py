"""Full-v2 regressions: source context, canonical results and truthful request facts."""
import asyncio
import json
from types import SimpleNamespace
import pytest
from google.genai import types
from google.adk.models.llm_response import LlmResponse

from src.single_agent import reset_turn, validate_final, RESULT_KEY
from src.tools.agent_tools import remember, select_structured_result
from src.validation.excerpts import excerpt_catalog
from src.validation.extractive import verify_extractive_claims
from src.validation.passages import table_passages
from src.request_context import request_details, unavailable_reason, user_facts
from src.orchestration import route_request

def context(question):
    return SimpleNamespace(state={}, user_content=types.Content(role="user", parts=[types.Part(text=question)]))

@pytest.mark.parametrize("question", ["Who is the instructor for TEST401?", "Draft a sick-leave email for TEST401."])
def test_successful_structured_schema_recovery(question):
    c=context(question);c.state["temp:structured_results"]={}
    if question.startswith("Who"):
        name="instructor";data={"instructors":[{"instructor_name":"Synthetic","email":"faculty@example.invalid"}]}
    else:
        name="email_draft";data={"to":["faculty@example.invalid"],"subject":"Synthetic","body":"[Student name]","sent":False}
    remember(c,name,{"status":"success","data":data,"evidence":[],"message":""})
    asyncio.run(validate_final(c,LlmResponse(content=types.Content(role="model",parts=[
        types.Part(text='{"status":"answer","claims":[{"text":"untrusted prose"}]}')]))))
    assert c.state[RESULT_KEY]["status"]=="answer"
    assert c.state[RESULT_KEY]["data"]==data
    assert "untrusted prose" not in c.state[RESULT_KEY]["answer"]

def test_unknown_explicit_result_is_never_recovered():
    assert select_structured_result('{"status":"tool_result","result_id":"fake"}',
        {"result_1":{"tool":"instructor","status":"success"}},"Who is the instructor?") is None

def test_leave_abbreviations_keep_programme_conditions():
    text=("A PhD student cannot avail leave for more than 15 days at a stretch\n"
          "An M.Tech./M.Sc. student cannot avail leave for more than 10 days at a stretch\n"
          "The leave beyond 30 days may be granted in exceptional cases without assistantship.")
    source={"chunk_id":"x","document_id":"reg","page_start":1,"text":text}
    items=excerpt_catalog([source])
    assert any("15 days" in i["text"] and "M.Tech./M.Sc. student" in i["text"] and "10 days" in i["text"] for i in items)
    assert not any(i["text"].startswith("student cannot") for i in items)

def test_table_headers_and_full_rows_remain_exact():
    text=("Table 11.1 Maximum and Minimum credits\n"
          "Category                         Maximum Minimum\n"
          "Full-time (Regular/Sponsored)    17      10\n"
          "External                         20      10\n\n"
          "11.1.1 Registration Kept Alive\nA student must apply.")
    passage=table_passages(text)[0]
    assert "17      10" in passage and "Maximum Minimum" in passage
    source={"chunk_id":"x","document_id":"reg","page_start":1,"text":text}
    claim={"text":passage,"supports":[{"chunk_id":"x","quote":passage}]}
    assert verify_extractive_claims([claim],[source])["supported"]
    clipped="17      10"
    assert not verify_extractive_claims([{"text":clipped,"supports":[{"chunk_id":"x","quote":clipped}]}],[source])["supported"]

def test_generic_email_negative_details_do_not_route_schedule():
    assert route_request("Draft a sick-leave email; I have not supplied the course code or semester.")==["email"]

@pytest.mark.parametrize("question", ["What is my personal academic CGPA and class rank?",
    "What academic scholarship amount was deposited into my bank account?"])
def test_private_academic_records_are_unsupported_not_out_of_scope(question):
    c=context(question); reset_turn(c)
    assert c.state[RESULT_KEY]["status"]=="insufficient_evidence"
    assert c.state[RESULT_KEY]["validation"]=="academic_applicability_abstention"

def test_future_year_does_not_use_old_calendar():
    c=context('Student context: {"semester":"2030-31-S1"}\nQuestion: What is the exact Major examination date?')
    reset_turn(c)
    assert c.state[RESULT_KEY]["status"]=="insufficient_evidence"

def test_only_explicit_date_facts_are_extracted():
    assert "start_date" not in user_facts("I missed the exam due to illness.")
    assert user_facts("I missed the exam on 2026-10-01.")["start_date"]=="2026-10-01"

def test_incomplete_condition_is_still_rejected():
    source={"chunk_id":"x","text":"If absence exceeds two weeks,\n\nThe instructor may recommend de-registration."}
    quote="The instructor may recommend de-registration."
    assert not verify_extractive_claims([{"text":quote,"supports":[{"chunk_id":"x","quote":quote}]}],[source])["supported"]

def test_source_fragment_at_chunk_start_is_not_catalogued():
    assert excerpt_catalog([{"chunk_id":"x","document_id":"reg","page_start":1,
                             "text":"student can register for 24 credits."}])==[]


def test_known_corpus_source_is_pending_not_fabricated():
    from evaluation.scoring import check_citation
    assert check_citation({"document_id":"calendar_2026_27","page":2,"quote":"a passage"}, {})["score"] is None
    assert check_citation({"document_id":"nonexistent_fabricated","page":2}, {})["score"]==0

def test_stability_includes_minutes_and_internal_tool_outputs():
    from evaluation.scoring import extract_values
    assert "120 minutes" in extract_values({"answer":"The major lasted 120 minutes."})["numeric_units"]

def test_generic_email_is_canonical_in_both_modes(monkeypatch):
    from src.single_agent import answer_policy, baseline_agent
    async def forbidden(*args, **kwargs):
        raise AssertionError("Generic draft must not use inference")
        yield
    monkeypatch.setattr(type(baseline_agent.model),"generate_content_async",forbidden)
    for mode in ("single","multi"):
        r=asyncio.run(answer_policy("Draft a sick-leave email; I have not supplied the course code.", mode=mode))
        assert r["status"]=="answer"
        assert r.get("data",{}).get("to")==["[Instructor email]"]
        assert r["data"]["sent"] is False

def test_invented_personal_facts_are_guarded_in_baseline():
    from src.single_agent import guard_baseline_tool
    c=context("Draft a sick-leave email for TEST401.")
    c.state["temp:multi_request"]=request_details(c.user_content,c.state)
    args={"course_code":"TEST401","semester":"invented","request_type":"sick_leave",
          "start_date":"2026-04-05","student_name":"Invented Name"}
    guard_baseline_tool(SimpleNamespace(name="email_draft"),args,c)
    assert args["start_date"]==args["student_name"]==""


def test_exact_table_fallback_filters_programme(monkeypatch, tmp_path):
    import hashlib
    import src.tools.rag as rag
    import chromadb
    import ollama
    model_info={"test":"embedding"}
    report={"model":"nomic-embed-text","model_identity":hashlib.sha256(json.dumps(model_info,sort_keys=True).encode()).hexdigest(),
            "snapshot":"synthetic","collection":"synthetic","query_prefix":"search_query: "}
    (tmp_path/"index_report.json").write_text(json.dumps(report),encoding="utf-8")
    monkeypatch.setenv("CHROMA_PATH",str(tmp_path))
    monkeypatch.setenv("EMBED_MODEL","nomic-embed-text")
    class Client:
        def __init__(self,**kwargs):pass
        def show(self,model):return SimpleNamespace(model_dump=lambda **kwargs:model_info)
    monkeypatch.setattr(ollama,"Client",Client)
    monkeypatch.setattr("ingestion.build_vector_store.embeddings",lambda *args:[[1.,2.]])
    metadata=[{"document_id":name,"page_start":1,"page_end":1,"program":program,"chunk_index":0,"type":"regulation"}
              for name,program in [("pg","PG"),("ug","B.Tech.")]]
    bodies=["Table 12.8 Continuation\nProgramme   CGPA\nM.Tech.    6.0\nPh.D.      6.5",
            "Table 12.8 Wrong programme\nB.Tech. 9.9"]
    class Collection:
        metadata={"snapshot":"synthetic"}
        def get(self,include,where=None):
            if include==["metadatas"]:return {"metadatas":metadata}
            # Filter the fake collection just as Chroma must filter production.
            return {"ids":["pg"],"documents":[bodies[0]],"metadatas":[metadata[0]]}
        def query(self,**kwargs):
            return {"ids":[[]],"documents":[[]],"metadatas":[[]],"distances":[[]]}
    monkeypatch.setattr(chromadb,"PersistentClient",lambda **kwargs:SimpleNamespace(get_collection=lambda *a,**k:Collection()))
    result=rag.search_policy("According to Table 12.8, what is the assistantship requirement?","PG")
    assert result["status"]=="success"
    assert result["evidence"][0]["retrieval_role"]=="exact_reference"
    assert {e["document_id"] for e in result["evidence"]}=={"pg"}
    assert rag.search_policy("According to Table 99.9, what is required?","PG")["status"]=="insufficient_evidence"


def test_explicit_source_scope_excludes_neighbour_policy_noise():
    from src.validation.context_budget import focused_excerpts
    sources=[{"chunk_id":"x","document_id":"reg","page_start":1,"text":"Special grades are I, S, X and U.","retrieval_role":"exact_reference"},
             {"chunk_id":"y","document_id":"old","page_start":2,"text":"Students require C grade when absent.","retrieval_role":"neighbor_context"}]
    result=focused_excerpts(sources,"According to Section 5.1 what special grades apply?")
    assert {r["document_id"] for r in result}=={"reg"}

def test_query_qualifiers_apply_before_model_selection():
    from src.validation.context_budget import focused_excerpts
    sources=[{"chunk_id":"x","document_id":"reg","page_start":1,"text":
        "Automatic semester withdrawal applies to all academic activities. The instructor may recommend de-registration."}]
    result=focused_excerpts(sources,"When can an instructor recommend de-registration?")
    assert not any("withdrawal" in r["text"] for r in result)


def test_policy_projection_changes_model_task_not_user_facts():
    from src.agents.common import project_specialist_task
    q="Explain the leave policy and draft an email for TEST401."
    content=types.Content(role="user",parts=[types.Part(text='Student context: {}\nQuestion: '+q)])
    ctx=SimpleNamespace(state={"temp:multi_request":{"question":q,"agents":["policy","email"]},
                               "temp:active_role":"policy"})
    request=SimpleNamespace(contents=[content])
    project_specialist_task(ctx,request)
    assert "draft an email" not in request.contents[0].parts[0].text
    assert ctx.state["temp:multi_request"]["question"]==q
    assert "draft an email" in content.parts[0].text


def test_content_and_outer_status_are_distinct():
    from evaluation.scoring import load_benchmark, grade_response
    ds,truth=load_benchmark()
    q=next(q for q in ds["questions"] if q["id"]=="E03")
    gt=truth["questions"]["E03"]
    result={"status":"partial_answer","citations":[],"sections":[
        {"status":"insufficient_evidence","answer":"Unneeded schedule failed."},
        {"status":"answer","tool":"email_draft","data":{"to":["[Instructor email]"],
         "subject":"Request for sick leave","body":gt["acceptable_email_bodies"][0],
         "sent":False,"requires_student_review":True}}]}
    g=grade_response(q,gt,result,truth["sources"],[],"multi")
    assert g["content_accuracy"]==1
    assert g["accuracy"]==0
    assert g["outer_status_correct"]==0

def test_internal_tool_output_changes_are_material():
    from evaluation.scoring import stability
    import copy
    r={"model":"fake","mode":"single","question_id":"x","result":{"status":"answer","answer":"same"},
       "tool_calls":[],"grading":{"accuracy_eligible":True,"accuracy":1},
       "observations":{"tool_results":[{"tool":"email_draft","data":{"body":"date 2026-04-05"}}]}}
    other=copy.deepcopy(r)
    other["observations"]["tool_results"][0]["data"]["body"]="[Start/exam date]"
    s=stability([r,other])[0]
    assert s["fields"]["tool_outputs"]["all_equal"] is False
    assert s["material_change"] is True

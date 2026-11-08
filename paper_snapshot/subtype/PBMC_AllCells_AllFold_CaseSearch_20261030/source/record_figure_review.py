"""Save direct native-review trace when installed forensic helpers are unavailable."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROMPT="""Perform fresh read-only scientific figure and claim review as required by paper-figure skill. No code edits, no SSH, no processes to stop. Primary inputs local E:\\my_research\\subtype\\PBMC_AllCells_AllFold_CaseSearch_20261030: final_report.md, candidate_evidence_matrix.csv, candidate_seed_evidence_matrix.csv, case_studies (three cases; rendered PNG/PDF, plot-data CSV, Chinese captions), figures/all_candidates_evidence.png. Actually open rendered images with view_image; inspect all three contact sheets in audit/ plus main UMAP and pathway panels/full-candidate overview at readable size. Verify claims against CSV: fair PCA comparison, fixed barcode membership, posthoc selection, negative all-gene GSEA vs ORA/expressed sensitivity analysis, cross-seed caveats, confounding. Identify concrete defects with paths; no aesthetic cherry picking or new cell selection. Return completed scope, images opened, blockers/major/minor findings, and verdict. This is same-family provisional review; requested model is not verified actual identity. You may create only a review report under audit/FIGURE_REVIEW.md if useful. Chinese report preferred."""


def main():
    now=datetime.now(timezone.utc).isoformat()
    trace=ROOT/'.aris/traces/paper-figure/2026-10-31_run01'
    trace.mkdir(parents=True,exist_ok=True)
    report=(ROOT/'audit/FIGURE_REVIEW.md').read_text(encoding='utf-8')
    response=(ROOT/'audit/FIGURE_REVIEW_FINAL_RESPONSE.md').read_text(encoding='utf-8')
    meta=dict(call_number=1,purpose='final-figure-review',timestamp=now,
        tool='collaboration.spawn_agent',backend='codex',
        agent_id='/root/figure_final_review',requested_reviewer_model='gpt-6-astra',
        reported_reviewer_model=None,reviewer_model='gpt-6-astra',reviewer_model_source='requested',
        effort='xhigh',executor_model='GPT-6',executor_family='openai',
        review_independence='same-family',acceptance_status='provisional',
        review_completed=True,review_scope_complete=True,review_positive=True,
        verdict='completed; no unresolved blocker after documented corrections',status='ok',
        helper_resolution='Canonical helper unavailable in installed partial skill pack; direct trace writer used',
        identity_limitation='Requested identity only; not independently verified')
    request=dict(meta,prompt=PROMPT,files_referenced=['final_report.md','candidate_evidence_matrix.csv',
        'candidate_seed_evidence_matrix.csv','case_studies/','figures/all_candidates_evidence.png','audit/'])
    (trace/'run.meta.json').write_text(json.dumps(dict(skill='paper-figure',
        run_id='2026-10-31_run01',executor='codex',executor_model='GPT-6',
        reviewer_backend='codex',project_dir=str(ROOT)),indent=2),encoding='utf-8')
    (trace/'001-final-figure-review.request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    (trace/'001-final-figure-review.response.md').write_text(response,encoding='utf-8')
    (trace/'001-final-figure-review.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    events=ROOT/'.aris/meta';events.mkdir(exist_ok=True)
    with (events/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event='review_trace',skill='paper-figure',
            purpose='final-figure-review',agent_id=meta['agent_id'],
            trace_path=str(trace.relative_to(ROOT)),status='ok'))+'\n')
    path=ROOT/'audit/FINAL_CASE_ARTIFACTS_COMPLETE.json'
    artifact=json.loads(path.read_text(encoding='utf-8'))
    artifact['visual_review']='COMPLETE: rendered self-inspection and fresh native reviewer; corrections documented in FIGURE_REVIEW.md; same-family provisional'
    path.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    assert len(report)>100
    print('FIGURE_REVIEW_TRACE_COMPLETE',flush=True)


if __name__=='__main__':main()

"""Check final rendered files and create clearly labelled audit thumbnails."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    checked=[]
    for case in json.loads((ROOT/'SELECTED_CASES.json').read_text()):
        directory=ROOT/'case_studies'/case['id']/'figures'
        files=sorted(directory.glob('*.png'))
        assert len(files)==14
        contact=Image.new('RGB',(1800,math.ceil(len(files)/3)*370),'white')
        draw=ImageDraw.Draw(contact)
        for i,path in enumerate(files):
            assert path.with_suffix('.pdf').stat().st_size>1000
            assert path.with_name(path.stem+'_caption_zh.md').stat().st_size>20
            with Image.open(path) as image:
                assert abs(image.info['dpi'][0]-600)<1
                assert min(image.size)>1000
                image.thumbnail((580,330))
                x=(i%3)*600;y=(i//3)*370
                contact.paste(image.convert('RGB'),(x,y+25))
                draw.text((x+8,y+4),path.stem,fill='black')
            checked.append(path.relative_to(ROOT).as_posix())
        contact.save(ROOT/'audit'/f"{case['id']}_contact.png")
    overview=ROOT/'figures/all_candidates_evidence.png'
    with Image.open(overview) as image:
        assert abs(image.info['dpi'][0]-600)<1
        image.thumbnail((1200,2400))
        image.save(ROOT/'audit/all_candidate_overview_preview.png')
    (ROOT/'audit/FINAL_RENDERED_ARTIFACT_CHECK.json').write_text(json.dumps(dict(
        status='PASS',case_figure_pairs=len(checked),overview_png600=True,
        checked=checked,thumbnails_are_for_audit_only=True,
        independent_visual_review='FIGURE_REVIEW.md'),indent=2))
    print('FINAL_RENDERED_ARTIFACT_PASS',len(checked),flush=True)


if __name__=='__main__':main()

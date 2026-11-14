"""Read-only source-image previews of the active post-freeze review differences."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[2]
NOTES=ROOT/'docs/agent-notes'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status=json.loads((NOTES/'json/current_review_status.json').read_text(encoding='utf-8'))
    table_path=NOTES/'table-csv/remaining_current_review_differences.csv'
    table=pd.read_csv(table_path,dtype={'parent_id':str})
    assert len(table)==status['remaining_review_differences']==26
    assert table.parent_id.is_unique and table.confirmed_for_audit.all()
    table=table.assign(order=table.parent_id.astype(int)).sort_values('order').reset_index(drop=True)
    manifest_path=ROOT/'data/processed/manifest.csv'
    assert sha(manifest_path)==status['manifest_sha256']
    manifest=pd.read_csv(manifest_path,dtype={'parent_id':str})
    manifest=manifest[manifest.split.eq('train')&~manifest.is_crop].set_index('parent_id')
    out=NOTES/'remaining_visual_review';out.mkdir(exist_ok=True)
    pages=[];page_count=math.ceil(len(table)/9)
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white'})
    for page in range(page_count):
        rows=table.iloc[page*9:(page+1)*9]
        fig,axes=plt.subplots(3,3,figsize=(12,14),dpi=130)
        fig.suptitle(f'Sisa 26 perbedaan kategori — lembar {page+1}/{page_count}',fontsize=20,y=.986)
        fig.text(.5,.957,'Urutan nomor menaik · Program aktif dibandingkan dengan review tim',ha='center',fontsize=12,color='#555555')
        for ax in axes.ravel():ax.axis('off')
        for ax,row in zip(axes.ravel(),rows.itertuples()):
            path=(ROOT/row.source_path).resolve();path.relative_to((ROOT/'data').resolve())
            record=manifest.loc[row.parent_id]
            assert sha(path)==record.source_sha256
            assert record.assigned_class==row.current_class and row.current_class!=row.proposed_class
            with Image.open(path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
            ax.imshow(im)
            ax.set_title(f'Train {int(row.parent_id):04d}\nProgram: {row.current_class} | Review: {row.proposed_class}',fontsize=12,pad=10)
        fig.subplots_adjust(left=.025,right=.975,top=.918,bottom=.035,hspace=.32,wspace=.15)
        fig.text(.5,.012,'Review tim adalah pembanding audit; bukan ground truth resmi panitia.',ha='center',fontsize=10,color='#666666')
        path=out/f'remaining_26_page_{page+1}.png'
        fig.savefig(path,facecolor='white');plt.close(fig)
        pages.append({'path':str(path.relative_to(ROOT)),'image_ids':rows.parent_id.tolist(),'sha256':sha(path)})
    (out/'remaining_26_sheets.json').write_text(json.dumps({'active_prefix':status['active_prefix'],'rows':26,'table_sha256':sha(table_path),'manifest_sha256':sha(manifest_path),'pages':pages,'source_images_changed':False,'labels_changed':False},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pages,indent=2))


if __name__=='__main__':main()

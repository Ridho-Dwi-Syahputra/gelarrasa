"""Fetch only pretrained-model bytes from Cornell; never download MINC images.

Range requests retain a reproducible local prefix of the authors' tar archive;
only GoogLeNet weights, deploy config and categories are extracted.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,tarfile,urllib.request,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pretrained/minc15'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'
URL='https://opensurfaces.cs.cornell.edu/static/minc/minc-model.tar.gz'
CHUNK=8*1024*1024

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    parts=OUT/'ranges';parts.mkdir(exist_ok=True)
    previous=OUT/'download.part'
    if previous.exists():
        with previous.open('rb') as f:
            for index in range(previous.stat().st_size//CHUNK):
                target=parts/f'{index:04d}.part'
                data=f.read(CHUNK)
                if not target.exists():target.write_bytes(data)
    started=time.time()
    def fetch(index):
        target=parts/f'{index:04d}.part'
        if target.exists() and target.stat().st_size==CHUNK:return index
        start=index*CHUNK;end=start+CHUNK-1
        request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request,timeout=180) as response:
                    assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                    data=response.read()
                assert len(data)==CHUNK
                target.write_bytes(data);return index
            except Exception:
                if attempt==2:raise
    # GoogLeNet follows AlexNet in the author's archive; VGG16 is not required.
    for mb in [320,400,512,640,712]:
        count=mb//8
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(fetch,i) for i in range(count)]):
                index=future.result()
                print(f'model range {index+1}/{count}, {time.time()-started:.0f}s',flush=True)
        prefix=OUT/'model-prefix.tar.gz'
        with prefix.open('wb') as dest:
            for i in range(count):
                with (parts/f'{i:04d}.part').open('rb') as src:
                    for block in iter(lambda:src.read(CHUNK),b''):dest.write(block)
        selected=[];done=False
        try:
            with tarfile.open(prefix,mode='r|gz') as archive:
                for item in archive:
                    wanted=item.name.endswith(('categories.txt','deploy-googlenet.prototxt','minc-googlenet.caffemodel'))
                    if wanted and item.isfile():
                        target=OUT/'model'/Path(item.name).name;target.parent.mkdir(exist_ok=True)
                        with archive.extractfile(item) as src,target.open('wb') as dest:
                            for block in iter(lambda:src.read(CHUNK),b''):dest.write(block)
                        selected.append(target)
                        print('extracted',target.name,target.stat().st_size,flush=True)
                        if target.name=='minc-googlenet.caffemodel':done=True;break
        except (tarfile.ReadError,EOFError):pass
        if done:
            record=dict(source='Cornell MINC authors, CVPR2015',source_url=URL,
                official_page='https://opensurfaces.cs.cornell.edu/publications/minc/',
                architecture='Caffe GoogLeNet CNN; no language encoder',external_training_images_used=False,
                MINC_image_dataset_downloaded=False,extracted_image_files=0,model_archive_contains_unused_example_image=True,
                pretrained_only=True,local_competition_adaptation_required=True,
                upstream_checksum_published=False,
                files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in selected})
            (REPORT/'material_checkpoint.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
            print('Material CNN weights ready',flush=True);return
    raise RuntimeError('Requested model was not completely recovered; no inference performed.')

if __name__=='__main__':main()

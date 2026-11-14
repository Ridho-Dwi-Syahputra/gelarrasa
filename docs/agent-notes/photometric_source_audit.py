"""Bidirectional registered-pixel duplicate evidence, train-only, no categories."""
import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from skimage.metrics import structural_similarity
from fewshot_train_only_probe import ROOT,NOTES,OLD,sha,dump


def main():
    start=time.time();cv2.setNumThreads(1)
    edges=pd.read_csv(NOTES/'table-csv/local_geometry_edges.csv');edges=edges[edges.accepted]
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);allowed=set(b['keys'])
    needed=sorted(set(edges.key_a)|set(edges.key_b));assert set(needed)<=allowed
    profile=pd.read_csv(OLD/'output/manifests/stage1_v1/image_profile.csv').set_index('key')
    local=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False);li=dict(zip(local['keys'],range(len(local['keys']))))
    plan={'scope':'same-source pixel identity evidence,not material category prediction','train_rows':len(needed),'test_rows':0,
          'candidates':'existing116SIFT/RANSACedges;no reviewcasesread',
          'matching':'600SIFT,ratio.7,RANSAC3px;maxdimension640',
          'verification':'bidirectional fullregisteredRGBSSIM>=.95,objectROI SSIM>=.95,bothcoverage>=.8,RGBMAE<=.03',
          'ROI':'fixedautomaticobjectboxwhenavailable;otherwisecentral80percent;no categoryread',
          'no_category_assignment':True,'source_code_sha256':sha(__file__)}
    dump(NOTES/'json/photometric_source_plan.json',plan)
    sift=cv2.SIFT_create(nfeatures=600,contrastThreshold=.04);images={};matcher=cv2.BFMatcher(cv2.NORM_L2)
    for key in needed:
        row=profile.loc[key];path=ROOT/row.source_path
        if not path.exists():path=ROOT/'data'/row.source_path
        assert sha(path)==row.source_sha256
        with Image.open(path) as source:im=ImageOps.exif_transpose(source).convert('RGB')
        ow,oh=im.size;im.thumbnail((640,640),Image.Resampling.LANCZOS);rgb=np.array(im);h,w=rgb.shape[:2];roi=np.zeros((h,w),np.uint8)
        i=li[key]
        if local['focus_flags'][i]:
            a,b,c,d=local['boxes'][i];a,c=int(a*w/ow),int(np.ceil(c*w/ow));b,d=int(b*h/oh),int(np.ceil(d*h/oh))
            roi[max(0,b):min(h,d),max(0,a):min(w,c)]=1
        else:roi[int(.1*h):max(int(.1*h)+1,int(.9*h)),int(.1*w):max(int(.1*w)+1,int(.9*w))]=1
        kp,des=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None);points=np.array([v.pt for v in kp],np.float32).reshape(-1,2)
        images[key]=(rgb,roi,points,des)
    def compare(a,b,matrix):
        rgb,roi,_,_=a;target,target_roi,_,_=b;h,w=target.shape[:2]
        warped=cv2.warpPerspective(rgb,matrix,(w,h),flags=cv2.INTER_LINEAR)
        valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)
        coverage=float(valid.mean());valid=cv2.erode(valid,np.ones((7,7),np.uint8)).astype(bool)
        mask=valid&target_roi.astype(bool)
        if valid.sum()<100 or mask.sum()<100:return coverage,0.,0.,1.
        _,map_ssim=structural_similarity(warped,target,channel_axis=2,data_range=255,full=True,gaussian_weights=True,sigma=1.5)
        map_ssim=map_ssim.mean(2);global_score=float(map_ssim[valid].mean());object_score=float(map_ssim[mask].mean())
        mae=float(np.abs(warped.astype(np.float32)-target.astype(np.float32))[mask].mean()/255)
        return coverage,global_score,object_score,mae
    records=[]
    for edge in edges.itertuples():
        a,b=images[edge.key_a],images[edge.key_b];pa,da=a[2:];pb,db=b[2:]
        pairs=matcher.knnMatch(da,db,k=2);good=[v[0] for v in pairs if len(v)==2 and v[0].distance<.7*v[1].distance]
        if len(good)<16:continue
        h,mask=cv2.findHomography(pa[[v.queryIdx for v in good]],pb[[v.trainIdx for v in good]],cv2.RANSAC,3.,maxIters=2000,confidence=.995)
        if h is None or abs(np.linalg.det(h))<1e-12:continue
        ab=compare(a,b,h);ba=compare(b,a,np.linalg.inv(h));passed=min(ab[0],ba[0])>=.8 and min(ab[1],ba[1])>=.95 and min(ab[2],ba[2])>=.95 and max(ab[3],ba[3])<=.03
        records.append({'key_a':edge.key_a,'key_b':edge.key_b,'coverage_a_to_b':ab[0],'coverage_b_to_a':ba[0],
                        'full_ssim_min':min(ab[1],ba[1]),'roi_ssim_min':min(ab[2],ba[2]),'roi_rgb_mae_max':max(ab[3],ba[3]),'photometric_duplicate':passed})
    table=pd.DataFrame(records);path=NOTES/'table-csv/photometric_source_edges.csv';table.to_csv(path,index=False)
    proof={'status':'passed','existing_geometric_edges':len(edges),'registered_pairs_checked':len(table),
           'photometric_duplicate_edges':int(table.photometric_duplicate.sum()),'train_source_rows':len(needed),'test_rows':0,'categories_read':False,
           'source_code_sha256':sha(__file__),'evidence_sha256':sha(path),'seconds':time.time()-start}
    dump(NOTES/'json/photometric_source_audit.json',proof);print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()

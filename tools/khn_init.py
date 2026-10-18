# Початкова прив'язка знімків Івана (ref/khn) до старої підкладки: SIFT -> афінне. Пише build/khn_init.json (копія в data/). ozerna прив'язано через khm (див. історію).
import json, numpy as np, cv2
meta=json.load(open('build_v1/sat_map.json')); sat=np.load('build_v1/sat_map.npy',mmap_mode='r')
f=0.32/2.0
R=cv2.resize(np.ascontiguousarray(sat[::2,::2]),None,fx=f*2,fy=f*2,interpolation=cv2.INTER_AREA)
Rg=cv2.cvtColor(R,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create(nfeatures=40000)
kr,dr=sift.detectAndCompute(Rg,None)
fl=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
res={}
for n in ['khm','khm2','ozerna']:
    im=cv2.imread('ref/khn/%s.jpg'%n,cv2.IMREAD_GRAYSCALE)
    for s in (0.3,0.375,0.45):
        t=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        kt,dt=sift.detectAndCompute(t,None)
        g=[a for a,b in fl.knnMatch(dt,dr,k=2) if a.distance<0.8*b.distance]
        if len(g)<10: print(n,s,'few',len(g)); continue
        src=np.float32([kt[a.queryIdx].pt for a in g])/s; dst=np.float32([kr[a.trainIdx].pt for a in g])
        A,inl=cv2.estimateAffine2D(src,dst,ransacReprojThreshold=4)
        ni=int(inl.sum())
        # px->meters
        Am=np.vstack([A,[0,0,1]]); S=np.array([[2,0,meta['x0']],[0,2,meta['y0']],[0,0,1]])
        M=S@Am
        print(n,s,'matches',len(g),'inl',ni,'mpp %.3f %.3f'%(np.hypot(*M[:2,0]),np.hypot(*M[:2,1])),'rot %.2f'%np.degrees(np.arctan2(M[1,0],M[0,0])))
        if ni>res.get(n,(0,))[0]: res[n]=(ni,M.tolist())
json.dump(res,open('build/khn_init.json','w'))

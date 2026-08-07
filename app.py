from __future__ import annotations

import io, math, re, tempfile, zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

APP_TITLE = 'KT3 PDO + PSC Microwell Analyzer'

@dataclass
class Settings:
    well_diameter_um: float = 100.0
    well_rmin: int = 23
    well_rmax: int = 40
    well_spacing: int = 54
    hough_p2: float = 27.0
    green_low: float = 30.0
    green_high: float = 45.0
    pdo_min_area: int = 20
    split_pdos: bool = True
    pdo_peak_distance: int = 18
    psc_peak_threshold: float = 9.0
    psc_red_minimum: float = 12.0
    psc_peak_distance: int = 4
    histogram_bins: int = 12

def fonts(a=22,b=17):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf',a), ImageFont.truetype('DejaVuSans.ttf',b)
    except Exception:
        return ImageFont.load_default(), ImageFont.load_default()

def natural_key(p: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',p.name)]

def infer_series(name: str, fallback: int):
    m=re.search(r'series\s*0*(\d+)',name,re.I)
    return int(m.group(1)) if m else fallback

def detect_wells(rgb,s):
    # OpenCV requires Python integers for minRadius/maxRadius.
    rmin=max(1,int(round(s.well_rmin)))
    rmax=max(rmin+1,int(round(s.well_rmax)))
    spacing=max(1,int(round(s.well_spacing)))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(7,7),1.5)
    c=cv2.HoughCircles(
        blur,cv2.HOUGH_GRADIENT,dp=1.15,minDist=float(spacing),
        param1=75.0,param2=float(s.hough_p2),
        minRadius=int(rmin),maxRadius=int(rmax)
    )
    if c is None:
        return np.empty((0,3),dtype=int)
    c=np.round(c[0]).astype(int)
    kept=[]
    for x,y,r in c[np.argsort(c[:,0])]:
        if all((x-a)**2+(y-b)**2>20**2 for a,b,_ in kept):
            kept.append((int(x),int(y),int(r)))
    return np.asarray(kept,dtype=int)

def cluster(vals,tol=12):
    vals=sorted(map(float,vals))
    if not vals:return []
    g=[[vals[0]]]
    for v in vals[1:]:
        if abs(v-np.mean(g[-1]))<=tol:g[-1].append(v)
        else:g.append([v])
    return [float(np.mean(x)) for x in g]

def grid_index(x,y,xs,ys):
    return int(np.argmin([abs(x-v) for v in xs]))+1,int(np.argmin([abs(y-v) for v in ys]))+1

def green_excess(rgb):
    a=rgb.astype(np.float32);r,g,b=a[...,0],a[...,1],a[...,2]
    return gaussian_filter(g-np.maximum(r,b),0.8)

def segment_pdos(green,s):
    labs=label(green>float(s.green_low));out=[]
    min_area=int(s.pdo_min_area);peak_dist=int(s.pdo_peak_distance)
    for reg in regionprops(labs,intensity_image=green):
        if reg.area<min_area or reg.intensity_max<float(s.green_high):continue
        y0,x0,y1,x1=reg.bbox
        mask=labs[y0:y1,x0:x1]==reg.label;sub=green[y0:y1,x0:x1];pieces=None
        if s.split_pdos:
            peaks=peak_local_max(sub,min_distance=peak_dist,threshold_abs=float(s.green_high),labels=mask.astype(np.uint8),exclude_border=False)
            if len(peaks)>=2:
                markers=np.zeros_like(sub,np.int32)
                for i,(py,px) in enumerate(peaks,1):markers[py,px]=i
                ws=watershed(-sub,markers=markers,mask=mask)
                cand=[p for p in regionprops(ws,intensity_image=sub) if p.area>=min_area and p.intensity_max>=float(s.green_high)]
                if len(cand)>=2:pieces=cand
            if pieces is None:
                dist=distance_transform_edt(mask)
                if float(dist.max())>0:
                    sp=peak_local_max(dist,min_distance=12,threshold_abs=max(7.0,0.55*float(dist.max())),labels=mask.astype(np.uint8),exclude_border=False)
                    if len(sp)==2:
                        vals=[float(dist[tuple(p)]) for p in sp];sep=float(np.linalg.norm(sp[0]-sp[1]))
                        if min(vals)/max(vals)>=0.60 and sep>=18:
                            markers=np.zeros_like(sub,np.int32)
                            for i,(py,px) in enumerate(sp,1):markers[py,px]=i
                            ws=watershed(-dist,markers=markers,mask=mask)
                            cand=[p for p in regionprops(ws,intensity_image=sub) if p.area>=max(220,min_area) and p.intensity_max>=float(s.green_high)]
                            if len(cand)==2:
                                aa=sorted(float(p.area) for p in cand)
                                if aa[0]/aa[1]>=0.35:pieces=cand
        if pieces:
            for p in pieces:
                cy,cx=p.centroid;out.append({'x':x0+float(cx),'y':y0+float(cy),'area':float(p.area)})
        else:
            cy,cx=reg.centroid;out.append({'x':float(cx),'y':float(cy),'area':float(reg.area)})
    return out

def detect_psc(rgb,x,y,r,s):
    R=max(4,int(round(r*0.86)));x0=max(0,x-R);x1=min(rgb.shape[1],x+R+1);y0=max(0,y-R);y1=min(rgb.shape[0],y+R+1)
    sub=rgb[y0:y1,x0:x1].astype(np.float32);yy,xx=np.ogrid[:sub.shape[0],:sub.shape[1]]
    mask=(xx-(x-x0))**2+(yy-(y-y0))**2<=R**2
    score=sub[...,0]-sub[...,2];score=score-gaussian_filter(score,5.0);score[~mask]=-999
    peaks=peak_local_max(score,min_distance=int(s.psc_peak_distance),threshold_abs=float(s.psc_peak_threshold),exclude_border=False)
    f=[]
    for py,px in peaks:
        if sub[py,px,0]-sub[py,px,2]>=float(s.psc_red_minimum):f.append((x0+int(px),y0+int(py),float(score[py,px])))
    return f

def crop_square(rgb,x,y,r,scale=4):
    R=int(round(r*1.75));x0=max(0,x-R);x1=min(rgb.shape[1],x+R);y0=max(0,y-R);y1=min(rgb.shape[0],y+R)
    im=Image.fromarray(rgb[y0:y1,x0:x1]);side=max(im.size);c=Image.new('RGB',(side,side),'black');c.paste(im,((side-im.width)//2,(side-im.height)//2))
    return c.resize((side*scale,side*scale),Image.Resampling.NEAREST)

def labelled_crop(crop,series,well,pdo_n,psc_n,sizes):
    title,body=fonts();lines=[f'Image {series:02d} | Well {well}',f'PDO count: {pdo_n} | PSC count: {psc_n}','PDO size'+('s' if len(sizes)!=1 else '')+': '+', '.join(f'{v:.1f} µm' for v in sizes)]
    dummy=Image.new('RGB',(10,10));d=ImageDraw.Draw(dummy);widths=[d.textbbox((0,0),t,font=(title if i==0 else body))[2] for i,t in enumerate(lines)]
    header=105;W=max(crop.width,max(widths)+28);out=Image.new('RGB',(W,header+crop.height),'white');out.paste(crop,((W-crop.width)//2,header));dr=ImageDraw.Draw(out);dr.rectangle((0,0,W,header),fill='black')
    y=8
    for i,t in enumerate(lines):
        f=title if i==0 else body;dr.text((12,y),t,font=f,fill='white');y+=34 if i==0 else 29
    return out

def indexed_overlay(rgb,wells):
    im=Image.fromarray(rgb).convert('RGB');dr=ImageDraw.Draw(im);_,f=fonts(18,14)
    for w in wells:
        x,y,r=w['x'],w['y'],w['r'];dr.ellipse((x-r,y-r,x+r,y+r),outline='yellow',width=1);dr.text((x-r,y-r-12),w['well'],fill='white',font=f)
    return im

def make_contact(paths,out,cols=5,gap=5):
    ims=[Image.open(p).convert('RGB') for p in paths]
    if not ims:return
    tw=min(620,max(i.width for i in ims));thumbs=[]
    for im in ims:
        sc=tw/im.width;thumbs.append(im.resize((tw,int(im.height*sc)),Image.Resampling.LANCZOS))
    ch=max(i.height for i in thumbs);rows=math.ceil(len(thumbs)/cols);sheet=Image.new('RGB',(cols*tw+(cols+1)*gap,rows*ch+(rows+1)*gap),'white')
    for i,im in enumerate(thumbs):
        rr,cc=divmod(i,cols);sheet.paste(im,(gap+cc*(tw+gap),gap+rr*(ch+gap)))
    sheet.save(out,dpi=(300,300))

def zip_bytes(folder):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(folder))
    return b.getvalue()

def process(files,s,cols):
    root=Path(tempfile.mkdtemp(prefix='kt3_web_'));inp=root/'input';out=root/'results'
    for d in [inp,out/'csv',out/'raw_crops',out/'labelled_crops',out/'indexed_large_images',out/'figures']:d.mkdir(parents=True,exist_ok=True)
    paths=[]
    for uf in files:
        p=inp/Path(uf.name).name;p.write_bytes(uf.getbuffer());paths.append(p)
    paths=sorted(paths,key=natural_key);wells_rows=[];pdo_rows=[];image_rows=[];labelled_paths=[]
    for idx,p in enumerate(paths,1):
        series=infer_series(p.name,idx);rgb=np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8);H,W=rgb.shape[:2]
        circles=detect_wells(rgb,s);circles=np.asarray([c for c in circles if c[0]-c[2]>=2 and c[0]+c[2]<W-2 and c[1]-c[2]>=2 and c[1]+c[2]<H-2],dtype=int)
        if len(circles)==0:raise RuntimeError(f'No fully visible wells detected in {p.name}. Try widening the well-radius range or lowering the well detection sensitivity.')
        xs=cluster(circles[:,0]);ys=cluster(circles[:,1]);umpp=float(s.well_diameter_um)/(2*float(np.median(circles[:,2])));pdos=segment_pdos(green_excess(rgb),s);local=[]
        for x,y,r in circles:
            x,y,r=int(x),int(y),int(r);col,row=grid_index(x,y,xs,ys);well=f'{col},{row}';assigned=[o for o in pdos if (o['x']-x)**2+(o['y']-y)**2<=(0.86*r)**2]
            psc=detect_psc(rgb,x,y,r,s);sizes=[2*math.sqrt(o['area']/math.pi)*umpp for o in assigned]
            local.append({'x':x,'y':y,'r':r,'well':well,'col':col,'row':row,'pdo_n':len(assigned),'psc_n':len(psc),'sizes':sizes})
            wells_rows.append({'image_series':series,'well_index':well,'well_col_index':col,'well_row_index':row,'well_centre_x_px':x,'well_centre_y_px':y,'well_radius_px':r,'um_per_pixel':umpp,'PDO_count':len(assigned),'PSC_like_focus_count':len(psc),'PDO_sizes_um':'; '.join(f'{v:.4f}' for v in sizes)})
            for n,(obj,size) in enumerate(zip(assigned,sizes),1):pdo_rows.append({'image_series':series,'well_index':well,'PDO_number_in_well':n,'PDO_count_in_well':len(assigned),'centroid_x_px':obj['x'],'centroid_y_px':obj['y'],'projected_area_px2':obj['area'],'equivalent_circular_diameter_um':size,'PSC_like_focus_count_in_well':len(psc)})
            if assigned:
                crop=crop_square(rgb,x,y,r);base=f'series_{series:02d}_well_{col}_{row}';rp=out/'raw_crops'/f'{base}.png';lp=out/'labelled_crops'/f'{base}_labelled.png';crop.save(rp,dpi=(300,300));labelled_crop(crop,series,well,len(assigned),len(psc),sizes).save(lp,dpi=(300,300));labelled_paths.append(lp)
        indexed_overlay(rgb,local).save(out/'indexed_large_images'/f'series_{series:02d}_indexed.png',dpi=(300,300))
        image_rows.append({'image_series':series,'source_image':p.name,'fully_visible_wells':len(local),'PDO_containing_wells':sum(w['pdo_n']>0 for w in local),'PDO_count':sum(w['pdo_n'] for w in local),'PSC_like_foci_all_wells':sum(w['psc_n'] for w in local),'um_per_pixel':umpp})
    wdf=pd.DataFrame(wells_rows);pdf=pd.DataFrame(pdo_rows);idf=pd.DataFrame(image_rows);wdf.to_csv(out/'csv'/'well_raw_data.csv',index=False);pdf.to_csv(out/'csv'/'PDO_raw_data.csv',index=False);idf.to_csv(out/'csv'/'image_summary.csv',index=False)
    if len(pdf):
        d=pdf['equivalent_circular_diameter_um'].astype(float);mean=float(d.mean());sd=float(d.std(ddof=1)) if len(d)>1 else 0.0
        fig,ax=plt.subplots(figsize=(6.2,4.6));ax.hist(d,bins=int(s.histogram_bins),edgecolor='black');ax.axvline(mean,ls='--',label=f'Mean = {mean:.1f} µm');ax.axvline(mean-sd,ls=':',label=f'±1 SD = {sd:.1f} µm');ax.axvline(mean+sd,ls=':');ax.set(xlabel='PDO equivalent circular diameter (µm)',ylabel='Number of PDOs');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'figures'/'PDO_size_distribution.png',dpi=300);plt.close(fig)
        fr=pdf['PSC_like_focus_count_in_well'].value_counts().sort_index().rename_axis('PSC_like_focus_count').reset_index(name='PDO_count');fr['percentage_of_PDOs']=100*fr.PDO_count/len(pdf);fr.to_csv(out/'csv'/'PSC_count_frequency_across_PDOs.csv',index=False)
        fig,ax=plt.subplots(figsize=(6.2,4.6));bars=ax.bar(fr.PSC_like_focus_count,fr.PDO_count,edgecolor='black')
        for b,c,pct in zip(bars,fr.PDO_count,fr.percentage_of_PDOs):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f'{int(c)}\n({pct:.1f}%)',ha='center',va='bottom',fontsize=8)
        ax.set(xlabel='PSC-like fluorescent foci in the same well',ylabel='Number of PDOs');ax.set_xticks(fr.PSC_like_focus_count);fig.tight_layout();fig.savefig(out/'figures'/'PSC_count_frequency_across_PDOs.png',dpi=300);plt.close(fig)
        summary=pd.DataFrame([{'images_processed':len(idf),'fully_visible_wells':len(wdf),'PDO_containing_wells':int((wdf.PDO_count>0).sum()),'PDO_count':len(pdf),'mean_PDO_diameter_um':mean,'median_PDO_diameter_um':float(d.median()),'SD_PDO_diameter_um':sd,'min_PDO_diameter_um':float(d.min()),'max_PDO_diameter_um':float(d.max()),'PSC_like_foci_all_detected_wells':int(wdf.PSC_like_focus_count.sum()),'PSC_like_foci_in_PDO_wells':int(wdf.loc[wdf.PDO_count>0,'PSC_like_focus_count'].sum())}])
    else:
        summary=pd.DataFrame([{'images_processed':len(idf),'fully_visible_wells':len(wdf),'PDO_containing_wells':0,'PDO_count':0,'PSC_like_foci_all_detected_wells':int(wdf.PSC_like_focus_count.sum()),'PSC_like_foci_in_PDO_wells':0}])
    summary.to_csv(out/'csv'/'overall_summary.csv',index=False)
    if labelled_paths:make_contact(labelled_paths,out/'figures'/'PDO_well_contact_sheet_compact.png',cols=int(cols),gap=5)
    return root,out,summary,idf

st.set_page_config(page_title=APP_TITLE,page_icon='🔬',layout='wide')
st.title(APP_TITLE)
st.caption('Upload RGB fluorescence microscopy images, analyze PDOs and PSC-like foci, inspect QC outputs, and download the complete result package.')
with st.expander('Measurement notes'):
    st.markdown('''- Microwell diameter is used as the calibration ruler (default **100 µm**).\n- PDO size is **2D equivalent circular diameter** from projected green area, not true 3D diameter.\n- PSC counts are automated **PSC-like red fluorescent foci**, not definitive individual-cell counts.\n- Always visually QC the indexed images and crops before thesis/publication use.''')
files=st.file_uploader('Upload large microscopy images',type=['png','jpg','jpeg','tif','tiff','bmp'],accept_multiple_files=True)
with st.sidebar:
    st.header('Analysis settings');st.caption('Defaults are tuned to the KT3 day-7 dataset.')
    well=st.number_input('Microwell diameter (µm)',1.0,1000.0,100.0,1.0);split=st.checkbox('Split touching PDOs',True);bins=st.slider('Histogram bins',5,30,12);cols=st.slider('Contact-sheet columns',3,10,5)
    with st.expander('Advanced thresholds'):
        rmin=st.number_input('Minimum well radius (px)',5,200,23,step=1);rmax=st.number_input('Maximum well radius (px)',6,300,40,step=1);spacing=st.number_input('Minimum well spacing (px)',10,500,54,step=1);hp2=st.number_input('Well detection sensitivity',1.0,100.0,27.0,1.0)
        gl=st.number_input('PDO green low threshold',0.0,255.0,30.0,1.0);gh=st.number_input('PDO green high threshold',0.0,255.0,45.0,1.0);amin=st.number_input('Minimum PDO area (px²)',1,100000,20,step=1);pdist=st.number_input('PDO split peak distance (px)',1,100,18,step=1)
        pt=st.number_input('PSC focus threshold',0.0,255.0,9.0,0.5);prm=st.number_input('PSC red-minus-blue minimum',0.0,255.0,12.0,0.5);ppd=st.number_input('PSC focus minimum spacing (px)',1,100,4,step=1)
run=st.button('Run analysis',type='primary',use_container_width=True,disabled=not files)
if run:
    s=Settings(float(well),int(rmin),int(rmax),int(spacing),float(hp2),float(gl),float(gh),int(amin),bool(split),int(pdist),float(pt),float(prm),int(ppd),int(bins))
    bar=st.progress(5,text='Starting analysis…')
    try:
        root,out,summary,idf=process(files,s,int(cols));bar.progress(100,text='Complete');st.session_state['zip']=zip_bytes(out);st.session_state['out']=str(out);st.session_state['summary']=summary.to_dict('records');st.session_state['idf']=idf.to_dict('records');st.success('Analysis complete.')
    except Exception as e:
        bar.empty();st.error(f'Analysis stopped: {e}')
if 'zip' in st.session_state:
    out=Path(st.session_state['out']);summary=pd.DataFrame(st.session_state['summary']);idf=pd.DataFrame(st.session_state['idf']);st.divider();st.subheader('Results')
    if len(summary):
        r=summary.iloc[0];c=st.columns(5);c[0].metric('Images',int(r.get('images_processed',0)));c[1].metric('Visible wells',int(r.get('fully_visible_wells',0)));c[2].metric('PDO wells',int(r.get('PDO_containing_wells',0)));c[3].metric('PDOs',int(r.get('PDO_count',0)));c[4].metric('Mean PDO diameter',f"{float(r['mean_PDO_diameter_um']):.1f} µm" if 'mean_PDO_diameter_um' in r and pd.notna(r['mean_PDO_diameter_um']) else '—')
    st.download_button('Download complete results ZIP',st.session_state['zip'],'KT3_PDO_PSC_analysis_results.zip','application/zip',type='primary',use_container_width=True)
    fdir=out/'figures';cc=st.columns(3)
    for col,(lab,name) in zip(cc,[('PDO size distribution','PDO_size_distribution.png'),('PSC frequency','PSC_count_frequency_across_PDOs.png'),('Cropped PDO wells','PDO_well_contact_sheet_compact.png')]):
        p=fdir/name
        if p.exists():col.image(str(p),caption=lab,use_container_width=True)
    with st.expander('Image-by-image summary'):st.dataframe(idf,use_container_width=True,hide_index=True)
    inds=sorted((out/'indexed_large_images').glob('*.png'))
    if inds:
        with st.expander('Indexed large-image QC overlays'):
            for i in range(0,len(inds),3):
                cs=st.columns(3)
                for col,p in zip(cs,inds[i:i+3]):col.image(str(p),caption=p.stem,use_container_width=True)

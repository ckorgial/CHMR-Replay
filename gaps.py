"""Secondary protocol: first masked frame must have no accepted association."""
from kitti import extract
from scientific_config import cache_directory, selected_suites
from models import replay,combine
import pandas as pd
import numpy as np
from pathlib import Path
out=Path('supplement')
out.mkdir(exist_ok=True)
if 'KITTI' not in selected_suites():
    raise SystemExit('KITTI is not in the selected suites; the gap protocol is not applicable')
y,r,seq=extract(cache_directory(),out,gap=True)
test=(seq%3!=0)
rows=[]
if test.any():
    stats=replay(y[test]);ref=r[test,60:]
    for mode in ['pooled','matched','one','equal']:
        p=combine(stats,10.,mode=mode)[0] if mode!='equal' else stats[2]+.5*stats[3]
        rows.append([mode,test.sum(),len(np.unique(seq[test])),np.sqrt(np.mean((p-ref)**2))])
pd.DataFrame(rows,columns=['method','windows','sequences','rmse']).to_csv(out/'gap_scores.csv',index=False)
print('Development and test gap windows',sum(~test),sum(test))

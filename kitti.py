"""Recover depth histories from published detections and annotated track IDs."""
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve
import hashlib, zipfile, csv, json
import time, os
from urllib.error import URLError
import numpy as np

COMMIT = '61f3bd72574093e367916c757b4747ca445f978c'
LABEL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip'
ROOT = f'https://raw.githubusercontent.com/xinshuoweng/AB3DMOT/{COMMIT}/data/KITTI/detection/pointrcnn_Car_val'

def get(path, url):
    if not path.exists():
        if os.environ.get('CHMR_ALLOW_DOWNLOAD') != '1' or os.environ.get('CHMR_OFFLINE') == '1':
            raise FileNotFoundError(f'Missing KITTI data {path}. Offline mode prevents downloading {url}')
        temporary=path.with_suffix('.partial')
        for attempt in range(3):
            try:
                urlretrieve(url, temporary)
                temporary.replace(path)
                break
            except (OSError, URLError):
                temporary.unlink(missing_ok=True)
                if attempt==2:
                    raise ConnectionError(f'Cannot obtain {path} from {url}. Connect to the Internet or pre-populate the cache.')
                time.sleep(1+attempt)
    return path

def extract(cache, out, gap=False):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    cache = Path(cache)
    cache.mkdir(exist_ok=True, parents=True)
    labels = get(cache/'labels.zip', LABEL)
    assert hashlib.sha256(labels.read_bytes()).hexdigest() == '6ad1fa0125e3cddcbe4d182e41cd12a854a11b0c733b26c706595f6db5c254e0'
    rows, history, truth = [], [], []
    checksums=json.loads(Path(__file__).with_name('detector_blobs.json').read_text())
    with zipfile.ZipFile(labels) as z:
        for seq in range(21):
            name = f'{seq:04d}.txt'
            det_path=get(cache/name, f'{ROOT}/{name}')
            raw=det_path.read_bytes()
            blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
            assert blob==checksums[name], f'Unexpected detector content: {name}'
            det = np.loadtxt(det_path, delimiter=',', ndmin=2)
            frames, tracks, measured = defaultdict(list), defaultdict(dict), defaultdict(dict)
            for line in z.read('training/label_02/'+name).decode().splitlines():
                p = line.split()
                if p[2] not in ('Car','Van'): continue
                t, k = map(int,p[:2])
                tracks[k][t] = (float(p[13]),float(p[15]),p[2])
                frames[t].append((k,np.array(p[6:10],float)))
            for t, objects in frames.items():
                candidates = det[det[:,0]==t]
                if not len(candidates): continue
                boxes = candidates[:,2:6]
                pairs = []
                for k,b in objects:
                    inter = np.maximum(0,np.minimum(b[2:],boxes[:,2:])-np.maximum(b[:2],boxes[:,:2])).prod(axis=1)
                    iou = inter/((b[2:]-b[:2]).prod()+(boxes[:,2:]-boxes[:,:2]).prod(axis=1)-inter)
                    pairs.extend((-iou[i],k,i) for i in np.flatnonzero(iou>=0.5))
                used = set()
                for _,k,i in sorted(pairs):
                    if i not in used and t not in measured[k]:
                        measured[k][t] = candidates[i,12]
                        used.add(i)
            for k, track in sorted(tracks.items()):
                next_start = -1
                for j in sorted(track):
                    start = j-59
                    if start<next_start or abs(track[j][0])>1.6 or not 5<=track[j][1]<=50: continue
                    if not all(t in measured[k] for t in range(start,j+1)): continue
                    if not all(t in track for t in range(j+1,j+11)): continue
                    if gap and j+1 in measured[k]: continue
                    history.append([measured[k][t] for t in range(start,j+1)])
                    truth.append([track[t][1] for t in range(start,j+11)])
                    rows.append((seq,k,start,j,j+10,track[j][2]))
                    next_start=j+11
    with open(out/('gap_tracks.csv' if gap else 'tracks.csv'),'w') as f:
        w=csv.writer(f); w.writerow(('sequence','track','first','loss','last','class'));w.writerows(rows)
    return np.array(history).reshape(-1,60), np.array(truth).reshape(-1,70), np.array([r[0] for r in rows])

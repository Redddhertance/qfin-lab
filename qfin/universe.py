import os
import time

import bs4
import pandas as pd
import requests

#point-in-time s&p500 membership, rebuilt from the revision history of the wikipedia
#constituents page. taking the page as it stood on a past date gives the index as it
#actually was then, including names that have since been acquired or dropped, which is
#what stops the backtest quietly selecting for survivors
API = 'https://en.wikipedia.org/w/api.php'
TITLE = 'List of S&P 500 companies'
UA = {'User-Agent': 'qfin-lab research (point-in-time index membership reconstruction)'}
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'sp500_membership.csv')

def get_json(session, params, retries: int = 6, backoff: float = 5.0):
    #wikipedia throttles bursts, answering 429 or an html error page rather than json, so
    #treat a decode failure the same as a bad status. honour Retry-After when it sends one
    last = None
    for attempt in range(retries):
        try:
            resp = session.get(API, params=params, headers=UA, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            last = f'status {resp.status_code}'
            wait = float(resp.headers.get('Retry-After', 0)) or backoff * (2 ** attempt)
        except Exception as exc:
            last = str(exc)
            wait = backoff * (2 ** attempt)
        time.sleep(min(wait, 120.0))
    raise RuntimeError(f'wikipedia api failed after {retries} attempts: {last}')

def revision_at(timestamp: str, session=None):
    #newest revision at or before the timestamp
    s = session or requests
    params = {'action': 'query', 'prop': 'revisions', 'titles': TITLE, 'rvlimit': 1,
              'rvdir': 'older', 'rvstart': timestamp, 'rvprop': 'ids|timestamp', 'format': 'json'}
    page = next(iter(get_json(s, params)['query']['pages'].values()))
    revs = page.get('revisions')
    if not revs:
        return None, None
    return revs[0]['revid'], revs[0]['timestamp']

def members_at(revid: int, session=None):
    #first column of the constituents table is the ticker. wikipedia writes class B share
    #classes with a dot, yahoo wants a dash, so normalise here rather than at every call site
    s = session or requests
    payload = get_json(s, {'action': 'parse', 'oldid': revid, 'prop': 'text', 'format': 'json'})
    table = bs4.BeautifulSoup(payload['parse']['text']['*'], 'html.parser').find('table', class_='wikitable')
    if table is None:
        return []
    out = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
        ticker = cells[0].get_text(strip=True).replace('.', '-').upper()
        if ticker and len(ticker) <= 6 and ticker.replace('-', '').isalpha():
            out.append(ticker)
    return out

def build_membership(start: str, end: str, freq: str = 'QE', cache_path: str = DEFAULT_CACHE,
                     refresh: bool = False, pause: float = 2.0):
    #one snapshot per period. quarterly is enough, the index only reconstitutes a handful
    #of names at a time and a stale month costs far less accuracy than survivorship does.
    #saves after every snapshot so a rate limit part way through doesn't lose the run,
    #just call it again and it picks up the snapshots it's still missing
    dates = pd.date_range(start=start, end=end, freq=freq)
    if len(dates) == 0 or dates[0] > pd.Timestamp(start):
        dates = pd.DatetimeIndex([pd.Timestamp(start)]).append(dates)
    dates = pd.DatetimeIndex([d.normalize() for d in dates])

    done = pd.DataFrame(columns=['snapshot', 'ticker', 'revid'])
    if os.path.exists(cache_path) and not refresh:
        done = load_membership(cache_path)
    have = set(pd.DatetimeIndex(done['snapshot']).normalize()) if len(done) else set()
    todo = [d for d in dates if d not in have]
    if not todo:
        return done
    print(f'{len(have)} snapshots cached, fetching {len(todo)} more')

    rows = done.to_dict('records')
    with requests.Session() as session:
        for d in todo:
            revid, when = revision_at(d.strftime('%Y-%m-%dT%H:%M:%SZ'), session)
            if revid is None:
                continue
            tickers = members_at(revid, session)
            print(f'  {d:%Y-%m-%d}: revision {revid} from {when[:10]}, {len(tickers)} members')
            rows.extend({'snapshot': d, 'ticker': t, 'revid': revid} for t in tickers)
            frame = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            frame.to_csv(cache_path, index=False) #checkpoint after each snapshot
            time.sleep(pause) #be polite to the api, this is someone else's server

    return pd.DataFrame(rows)

def load_membership(cache_path: str = DEFAULT_CACHE):
    frame = pd.read_csv(cache_path, parse_dates=['snapshot'])
    return frame

def all_tickers(membership: pd.DataFrame):
    #every name that was ever a member across the sample, this is what we download
    return sorted(membership['ticker'].unique())

def membership_mask(membership: pd.DataFrame, index: pd.DatetimeIndex, columns):
    #expand the sparse snapshots into a daily bool frame. a name counts as a member from the
    #snapshot it first appears in until the snapshot it drops out of, forward filled between
    snaps = sorted(membership['snapshot'].unique())
    wide = pd.DataFrame(False, index=pd.DatetimeIndex(snaps), columns=columns)
    for snap, group in membership.groupby('snapshot'):
        present = [t for t in group['ticker'] if t in wide.columns]
        wide.loc[snap, present] = True
    #reindex onto trading days, carrying each snapshot forward until the next one
    daily = wide.astype('boolean').reindex(wide.index.union(index)).ffill().reindex(index)
    return daily.fillna(False).astype(bool)

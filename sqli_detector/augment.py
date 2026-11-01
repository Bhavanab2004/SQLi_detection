"""Synthetic API-style samples.

The Kaggle-style dataset contains full SQL statements, but an API guard sees
individual user-supplied values ("john", "laptop 15 inch", "1' OR '1'='1").
We therefore (a) generate realistic benign values and (b) cut the injected
payload out of the malicious statements. Templates here are intentionally
different from the hand-written hold-out set in tests/holdout.py.
"""
import random
import re

FIRST = ["john", "priya", "amit", "sara", "li", "carlos", "fatima", "rahul", "emma", "noah",
         "ananya", "kiran", "meera", "arjun", "olivia", "ravi", "sneha", "david", "zoe", "imran"]
LAST = ["smith", "sharma", "o'brien", "d'souza", "patel", "kumar", "nguyen", "garcia", "khan",
        "reddy", "iyer", "o'connor", "lee", "brown", "singh", "rao", "mcdonald", "o'neil"]
DOMAINS = ["gmail.com", "example.com", "yahoo.in", "outlook.com", "company.org", "mca.edu"]
PRODUCTS = ["laptop", "wireless mouse", "usb-c cable", "running shoes", "coffee beans", "notebook a5",
            "phone case", "desk lamp", "yoga mat", "water bottle 1l", "4k monitor", "mechanical keyboard"]
PHRASES = [
    "please select a size", "union bank of india", "drop me a line", "order by date", "how to select a good laptop",
    "update my address", "delete my account please", "insert coin to continue", "where is my order",
    "and then we went home", "or maybe tomorrow", "table for two", "create a new playlist", "sleep 8 hours a day",
    "benchmark results 2024", "from bangalore to delhi", "the user's guide", "it's a great product!",
    "5 star - would buy again", "meeting at 10:30 -- room 4", "cost is $5 + tax", "100% cotton", "50 % off",
    "see you @ the cafe #1", "hello; how are you?", "what's the difference between sql and nosql",
    "admin panel access request", "password reset needed", "order #12345 status", "select all that apply",
]
STREETS = ["12 mg road", "4th cross, indiranagar", "221b baker street", "45 park avenue", "7 church st."]
SQL_VERBS = ["select", "union", "insert", "update", "drop", "delete", "order", "where", "and", "or"]


_LAST_K = 0


def _benign_one(r: random.Random) -> str:
    global _LAST_K
    k = r.randint(0, 11)
    _LAST_K = k
    f, l = r.choice(FIRST), r.choice(LAST)
    if k == 0: return f
    if k == 1: return f"{f} {l}"
    if k == 2: return f"{f}.{l.replace(chr(39), '')}{r.randint(1, 999)}@{r.choice(DOMAINS)}"
    if k == 3: return str(r.randint(1, 99999))
    if k == 4: return r.choice(PRODUCTS)
    if k == 5: return f"{r.choice(PRODUCTS)} {r.choice(['under', 'below', 'above'])} {r.randint(5, 900)}"
    if k == 6: return r.choice(PHRASES)
    if k == 7: return f"{r.randint(2019, 2026)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
    if k == 8: return r.choice(STREETS)
    if k == 9: return f"{f}{r.randint(1, 9999)}"
    if k == 10: return f"{r.choice(SQL_VERBS)} {r.choice(['the', 'a', 'my', 'your'])} {r.choice(PRODUCTS)}"
    return f"{f}_{l.replace(chr(39), '')}!{r.randint(10, 99)}"


def make_benign(n: int, seed: int = 0):
    r = random.Random(seed)
    return [_benign_one(r) for _ in range(n)]


def make_benign_with_ids(n: int, seed: int = 0):
    """Same stream as make_benign (same seed -> same texts) plus the generator-template id of each."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        t = _benign_one(r)
        out.append((t, _LAST_K))
    return out


# Attack templates for common textbook payload shapes (mixed with dataset fragments)
_TEMPLATES = [
    "{q}{c} OR {a}={a}{t}", "{q}{c} OR {q2}{a}{q2}={q2}{a}{q2}{t}", "{q} OR 1=1{t}", "admin{q}{t}",
    "{n}{q} UNION SELECT {cols} FROM {tb}{t}", "{n} UNION ALL SELECT NULL,NULL,{n2}{t}",
    "{n}{q}; DROP TABLE {tb}{t}", "{n}; SELECT * FROM {tb}{t}", "{n}{q} AND SLEEP({s}){t}",
    "{n}; WAITFOR DELAY '0:0:{s}'{t}", "{n}{q} AND (SELECT COUNT(*) FROM {tb})>0{t}",
    "{n} AND extractvalue(1,concat(0x7e,(select version()))){t}",
    "{n}{q} OR (SELECT SUBSTRING(password,1,1) FROM users WHERE username='admin')='a'{t}",
    "{n} ORDER BY {n2}{t}", "{n}{q} HAVING 1=1{t}", "{n}{q} AND 1=CONVERT(int,(SELECT @@version)){t}",
    "{n} OR pg_sleep({s}){t}", "{n}{q}) OR ({a}={a}{t}", "{n} AND (SELECT table_name FROM information_schema.tables LIMIT 1) LIKE 'u%'{t}",
    "{n}'; EXEC xp_cmdshell('dir'){t}", "{n} OR benchmark(5000000,md5(1)){t}",
]


def make_malicious_templates(n: int, seed: int = 0, with_ids: bool = False):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        tpl = r.choice(_TEMPLATES)
        a = r.randint(1, 9999)
        out.append(tpl.format(
            q=r.choice(["'", '"', "')", "'"]), q2=r.choice(["'", ""]), c=r.choice(["", " "]), a=a,
            n=r.choice(["1", "0", "-1", "admin", "john", "test", str(r.randint(1, 500))]),
            n2=r.randint(1, 30), cols=r.choice(["1,2,3", "username,password", "NULL,NULL", "user(),database()"]),
            tb=r.choice(["users", "accounts", "admin", "orders", "information_schema.tables"]),
            s=r.randint(2, 10), t=r.choice(["--", "-- ", "#", "/*", "", ";--", "-- -"]),
        ))
        if with_ids:
            out[-1] = (out[-1], _TEMPLATES.index(tpl))
    return out


_WHERE = re.compile(r"\bWHERE\s+\w+\s*(?:=|<=|>=|<|>|LIKE|IN|BETWEEN)\s*(.+)$", re.I | re.S)


def extract_fragment(query: str):
    """Cut the value part out of 'SELECT ... WHERE id = <payload>'."""
    m = _WHERE.search(query)
    if not m:
        return None
    frag = m.group(1).strip().rstrip("')")  # drop wrapper of EXEC('...') style rows
    return frag if len(frag) > 1 else None

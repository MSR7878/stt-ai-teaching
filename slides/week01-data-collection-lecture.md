---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Fira+Code&display=swap');
  @import 'custom.css';
---

<!-- _class: lead -->

# Week 1: Data Collection for Machine Learning

**CS 203: Software Tools and Techniques for AI**

Prof. Nipun Batra
IIT Gandhinagar

---

# The Netflix Movie Recommendation Problem

**Scenario**: You work at Netflix as a data scientist.

**The Task**: "Predict which movies will be successful to decide our next acquisitions."

**The Bottleneck**: We have no data.

**Today's Focus**: How do we build the dataset to solve this problem?

---

# The ML Pipeline

![width:900px](../figures/data_pipeline_flow.png)
*[diagram-generators/data_pipeline_flow.py](../diagram-generators/data_pipeline_flow.py)*

**Garbage In, Garbage Out**:
- 80% of ML work is data engineering.
- Sophisticated models cannot fix broken data.
- **Goal**: Automate the collection of high-quality data.

---

# Data Sources Strategy

We need features: *Title, Budget, Revenue, Reviews, Cast*.

| Source Type | Example | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Public APIs** | OMDb, TMDb | Structured, Reliable | Rate limits, Cost |
| **Web Scraping** | IMDb, Rotten Tomatoes | Free, Flexible | Fragile, IP bans |
| **Datasets** | Kaggle, Hugging Face | Clean, Ready | Static, Generic |

**Plan**: Use OMDb API for base data + Scraping for reviews.

---

# Part 1: The Web Protocol (HTTP)

How browsers (and scripts) talk to servers.

---

# Client-Server Architecture

![width:700px](../figures/http_request_sequence.png)
*[diagram-generators/http_request_sequence.py](../diagram-generators/http_request_sequence.py)*

---

# Understanding HTTP: The Foundation

**HTTP (HyperText Transfer Protocol)** is an application-layer protocol.

**Key characteristics**:
- **Stateless**: Each request is independent (no memory of previous requests)
- **Request-Response**: Client initiates, server responds
- **Text-based**: Human-readable headers and methods
- **Port 80** (HTTP) or **Port 443** (HTTPS - encrypted)

**Why it matters for ML**:
- Most APIs use HTTP/HTTPS
- Understanding requests helps debug data collection issues
- Rate limiting, caching, and errors are HTTP concepts

---

# Anatomy of a URL

**URL**: `https://api.omdbapi.com:443/search?apikey=123&t=Inception#results`

Breaking it down:
- **Protocol**: `https://` - Secure HTTP
- **Domain**: `api.omdbapi.com` - Server location
- **Port**: `:443` - Usually implicit (80 for HTTP, 443 for HTTPS)
- **Path**: `/search` - Resource location on server
- **Query String**: `?apikey=123&t=Inception` - Parameters (key=value pairs)
- **Fragment**: `#results` - Client-side anchor (not sent to server)

**Query parameters** are how we pass data in GET requests.

---

# HTTP Methods (Verbs)

Methods define the **action** to perform on a resource.

| Method | Purpose | Safe? | Idempotent? | Has Body? |
| :--- | :--- | :---: | :---: | :---: |
| **GET** | Retrieve data | Yes | Yes | No |
| **POST** | Create resource | No | No | Yes |
| **PUT** | Update/replace | No | Yes | Yes |
| **PATCH** | Partial update | No | No | Yes |
| **DELETE** | Remove resource | No | Yes | No |

**Safe**: Doesn't modify server state
**Idempotent**: Multiple identical requests = same result as one request

*For data collection, we mostly use GET.*

---

# HTTP Request Structure

A request has three parts:

**1. Request Line**:
```http
GET /search?q=movies HTTP/1.1
```

**2. Headers** (metadata):
```http
Host: api.omdbapi.com
User-Agent: Mozilla/5.0
Accept: application/json
Authorization: Bearer abc123
```

**3. Body** (optional, for POST/PUT):
```json
{"title": "Inception", "year": 2010}
```

Headers provide context about the request and client.

---

# HTTP Response Structure

A response also has three parts:

**1. Status Line**:
```http
HTTP/1.1 200 OK
```

**2. Headers**:
```http
Content-Type: application/json
Content-Length: 1234
Cache-Control: max-age=3600
```

**3. Body** (the actual data):
```json
{"Title": "Inception", "Year": "2010", ...}
```

---

# HTTP Status Codes (Theory)

Status codes are grouped by category:

**1xx - Informational**: Request received, processing continues
**2xx - Success**: Request successfully processed
**3xx - Redirection**: Further action needed to complete request
**4xx - Client Error**: Request has an error (your fault)
**5xx - Server Error**: Server failed to process valid request (their fault)

Understanding these helps debug data collection failures.

---

# Common Status Codes for Data Collection

**Success**:
- `200 OK`: Request succeeded
- `201 Created`: Resource created (POST)
- `204 No Content`: Success, but no response body

**Client Errors** (fix your code):
- `400 Bad Request`: Malformed request
- `401 Unauthorized`: Missing/invalid authentication
- `403 Forbidden`: Authenticated but not authorized
- `404 Not Found`: Resource doesn't exist
- `429 Too Many Requests`: Rate limit exceeded

**Server Errors** (retry later):
- `500 Internal Server Error`: Server crash
- `502 Bad Gateway`: Upstream server failed
- `503 Service Unavailable`: Server overloaded

---

# REST API Principles

**REST (Representational State Transfer)** is an architectural style.

**Core principles**:
1. **Stateless**: No session stored on server
2. **Resource-based**: URLs represent resources (nouns, not verbs)
3. **HTTP Methods**: Use standard verbs (GET, POST, PUT, DELETE)
4. **Standard formats**: JSON or XML responses
5. **HATEOAS**: Responses include links to related resources

**Example**:
- **Good**: `GET /movies/123` (resource-oriented)
- **Bad**: `GET /getMovie?id=123` (action-oriented)

---

# API Authentication Methods

Most APIs require authentication to track usage and prevent abuse.

**Common methods**:

1. **API Key** (simplest):
   - `?apikey=abc123` or `X-API-Key: abc123` header
   - Easy to use but less secure

2. **Bearer Token (OAuth)**:
   - `Authorization: Bearer abc123`
   - More secure, time-limited

3. **Basic Auth**:
   - `Authorization: Basic base64(username:password)`
   - Simple but requires HTTPS

4. **OAuth 2.0** (most secure):
   - Complex flow with authorization servers
   - Used by Google, Twitter APIs

---

# Rate Limiting: Theory

**Why rate limiting exists**:
- Prevent abuse and DoS attacks
- Ensure fair resource allocation
- Protect server infrastructure
- Monetization (pay for higher limits)

**Common approaches**:
1. **Fixed window**: 100 requests per hour (resets at :00)
2. **Sliding window**: 100 requests in any 60-minute period
3. **Token bucket**: Accumulate tokens, spend on requests
4. **Concurrent requests**: Max N simultaneous connections

**Headers to watch**:
- `X-RateLimit-Limit`: Total allowed requests
- `X-RateLimit-Remaining`: Requests left in period
- `X-RateLimit-Reset`: When limit resets (Unix timestamp)

---

# Part 2: CLI Tools (curl & jq)

Test APIs before writing code.

---

# curl: The HTTP Swiss Army Knife

**Fetch data**:
```bash
curl "http://www.omdbapi.com/?apikey=$KEY&t=Inception"
```

**Inspect headers (`-I`)**:
```bash
curl -I "https://google.com"
# HTTP/2 200
# content-type: text/html
```

**Why use curl?**
- Language agnostic.
- Instant debugging.
- "Copy as curl" from Chrome DevTools.

---

# jq: JSON Processor

Raw JSON is unreadable. `jq` makes it useful.

**Pretty print**:
```bash
curl ... | jq
```

**Filter fields**:
```bash
# Get just the title and rating
curl ... | jq '{Title, imdbRating}'
```

**Filter array elements**:
```bash
# Get titles of movies created after 2010
cat movies.json | jq '.[] | select(.Year > 2010) | .Title'
```

---

# Part 3: Python `requests`

Automating the process.

---

# The Synchronous Pattern

`requests` is **blocking**. The program stops until the server responds.

```python
import requests

def get_movie(title):
    url = "http://www.omdbapi.com/"
    params = {"apikey": "SECRET", "t": title}
    
    try:
        # Block here until response arrives
        resp = requests.get(url, params=params)
        resp.raise_for_status() # Check for 4xx/5xx
        return resp.json()
    except Exception as e:
        print(f"Failed: {e}")
        return None
```

---

# Advanced: Async IO (Conceptual)

**Problem**: Fetching 1,000 movies sequentially is slow.
**Solution**: Asynchronous Requests (`aiohttp`, `httpx`).

![width:800px](../figures/sync_vs_async_timing.png)
*[diagram-generators/sync_vs_async_timing.py](../diagram-generators/sync_vs_async_timing.py)*

*We will implement Async in Week 10 (FastAPI).*

---

# Handling Rate Limits: Exponential Backoff

**Strategy**: When rate limited, wait increasingly longer between retries.

```python
import time

def fetch_with_retry(url, retries=3):
    for i in range(retries):
        resp = requests.get(url)
        if resp.status_code == 429: # Rate limit
            wait = 2 ** i  # 1s, 2s, 4s...
            time.sleep(wait)
            continue
        return resp
```

**Why exponential?**
- Gives server time to recover
- Prevents thundering herd problem
- Standard practice in distributed systems

---

# Advanced: Retry Strategies

**Retry decision tree**:

| Status Code | Should Retry? | Strategy |
| :--- | :--- | :--- |
| `429 Too Many Requests` | Yes | Exponential backoff |
| `500 Internal Server Error` | Yes | Fixed delay, limited retries |
| `502/503 Service Error` | Yes | Short delay, many retries |
| `400 Bad Request` | No | Fix your request |
| `401/403 Auth Error` | No | Check credentials |
| `404 Not Found` | No | Resource doesn't exist |

**Implementation**: Use libraries like `tenacity` or `backoff` for production code.

---

# JSON Response Parsing

**JSON (JavaScript Object Notation)** is the standard API response format.

**Why JSON?**
- Human-readable and machine-parseable
- Nested structure (objects, arrays)
- Language-agnostic
- Smaller than XML

**Python mapping**:
- JSON object `{}` → Python dict
- JSON array `[]` → Python list
- JSON string `"text"` → Python str
- JSON number `42` → Python int/float
- JSON boolean `true/false` → Python bool
- JSON `null` → Python None

---

# Error Handling: Network Failures

**Common network errors**:

1. **Connection timeout**: Server unreachable
2. **Read timeout**: Server too slow to respond
3. **DNS failure**: Domain name doesn't resolve
4. **SSL certificate error**: Invalid/expired certificate
5. **Connection reset**: Server closed connection

**Always handle**:
```python
try:
    resp = requests.get(url, timeout=10)
except requests.exceptions.Timeout:
    # Server too slow
except requests.exceptions.ConnectionError:
    # Network problem
except requests.exceptions.RequestException:
    # Catch-all for other issues
```

---

# Ethical Scraping: robots.txt

**robots.txt** is a file that tells crawlers what they can access.

**Example**: `https://www.imdb.com/robots.txt`
```
User-agent: *
Disallow: /search/
Allow: /title/

Crawl-delay: 1
```

**Interpretation**:
- All bots (`*`) allowed
- Don't crawl `/search/` endpoints
- Can crawl `/title/` pages
- Wait 1 second between requests

**Check before scraping**:
```python
from urllib.robotparser import RobotFileParser

rp = RobotFileParser()
rp.set_url("https://example.com/robots.txt")
rp.read()

if rp.can_fetch("MyBot", "https://example.com/page"):
    # OK to scrape
```

---

# User-Agent Headers

**User-Agent** identifies your client to the server.

**Browser User-Agent**:
```
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
```

**Python requests default**:
```
python-requests/2.28.1
```

**Best practice** - Identify yourself:
```python
headers = {
    'User-Agent': 'MovieCollectorBot/1.0 (student@university.edu)'
}
requests.get(url, headers=headers)
```

**Why it matters**:
- Helps admins contact you if issues arise
- Some sites block generic User-Agents
- Ethical transparency

---

# Part 4: Web Scraping (BeautifulSoup)

When there is no API.

---

# When to Scrape vs Use APIs

**Prefer APIs when available**:
- Structured, reliable data
- Official support and documentation
- Stable endpoints
- Legal/ToS compliant

**Scraping is needed when**:
- No API exists
- API is too expensive
- API missing needed data
- API has restrictive rate limits

**Trade-offs**: Scraping is fragile (breaks when HTML changes).

---

# HTML Structure: The DOM

**DOM (Document Object Model)** is a tree structure.

```html
<html>
  <body>
    <div class="container">
      <div class="movie-card">
        <h1>Inception</h1>
        <span class="rating">8.8</span>
      </div>
    </div>
  </body>
</html>
```

**Tree representation**:
```
html
└── body
    └── div.container
        └── div.movie-card
            ├── h1 ("Inception")
            └── span.rating ("8.8")
```

---

# HTML Elements Anatomy

Each element has:

**Tag**: `<div>`, `<span>`, `<a>`, etc.
**Attributes**: `class="rating"`, `id="main"`, `href="/movie/123"`
**Content**: Text or child elements

```html
<a href="/movie/123" class="link" id="inception-link">
  Inception
</a>
```

**For scraping**: Find elements by tag, class, id, or attributes.

---

# CSS Selectors (Theory)

**CSS selectors** are patterns to select elements.

| Selector | Meaning | Example |
| :--- | :--- | :--- |
| `tag` | Element by tag name | `div`, `span`, `a` |
| `.class` | Element by class | `.rating`, `.movie-card` |
| `#id` | Element by ID | `#main`, `#header` |
| `tag.class` | Tag with class | `div.movie-card` |
| `parent > child` | Direct child | `div > span` |
| `parent descendant` | Any descendant | `div span` |
| `[attr=value]` | By attribute | `[href="/home"]` |

**BeautifulSoup uses these to find elements.**

---

# Parsing with BeautifulSoup

**BeautifulSoup** converts HTML to navigable Python objects.

**Basic pattern**:
```python
from bs4 import BeautifulSoup

html = "<div class='movie'><h1>Inception</h1></div>"
soup = BeautifulSoup(html, 'html.parser')

# Find by tag
title = soup.find('h1')  # First <h1>
print(title.text)  # "Inception"

# Find by class
movie = soup.find('div', class_='movie')

# Find all matching elements
all_divs = soup.find_all('div')
```

---

# Navigating the Tree

**BeautifulSoup navigation methods**:

**Children** (one level down):
```python
parent.find('child-tag')        # First child
parent.find_all('child-tag')    # All children
```

**Parents** (one level up):
```python
element.parent          # Direct parent
element.find_parent()   # Find ancestor
```

**Siblings** (same level):
```python
element.next_sibling
element.previous_sibling
```

**Useful for**: Complex layouts where structure matters.

---

# Static vs Dynamic Websites

**Static HTML**: Content in the initial HTML response
- Works with `requests + BeautifulSoup`
- Fast and simple
- Example: Wikipedia, simple blogs

**Dynamic JavaScript**: Content loaded after page loads
- Requires browser automation (`Selenium`, `Playwright`)
- Slower, heavier
- Example: Twitter, Facebook, modern SPAs

**Test**: `curl URL` and check if data is in the HTML source.

---

# Scraping Strategies Compared

| Approach | Tool | Speed | Difficulty | Use Case |
| :--- | :--- | :---: | :---: | :--- |
| **Static parsing** | BeautifulSoup | Fast | Easy | Server-rendered HTML |
| **Browser automation** | Playwright | Slow | Medium | JavaScript-heavy sites |
| **API inspection** | DevTools + requests | Fast | Medium | Hidden APIs in SPAs |
| **Vision models** | GPT-4V | Slow | Easy | Complex/inaccessible layouts |

**Rule**: Start simple (BeautifulSoup), escalate if needed.

---

# Anti-Scraping Measures

Websites use techniques to block scrapers:

1. **Rate limiting**: Block IPs with too many requests
2. **User-Agent filtering**: Block non-browser agents
3. **CAPTCHAs**: Require human interaction
4. **Session tracking**: Detect automated patterns
5. **Dynamic content**: Render with JavaScript
6. **IP blocking**: Ban suspicious IPs

**Countermeasures** (ethical):
- Respect `robots.txt`
- Use delays between requests
- Rotate User-Agents (within reason)
- Use proxies sparingly

---

# Data Licensing & Ethics

**Can I use this data?**

**Creative Commons licenses**:
1. **Public Domain (CC0)**: Free to use for anything
2. **Attribution (CC-BY)**: Must credit the source
3. **Non-Commercial (NC)**: Academic OK, commercial NO
4. **No Derivatives (ND)**: Can't modify the data

**Copyright**: "All Rights Reserved"
- **Fair Use**: Small excerpts for research may be OK (legal gray area)
- **Scraping**: Generally legal for public data (US: hiQ v LinkedIn)
- **ToS violation**: Can get you banned, but rarely legal consequences

---

# Legal Considerations

**Key court cases**:
- **hiQ Labs v. LinkedIn (2019)**: Scraping public data is legal in the US
- **Meta v. Bright Data (ongoing)**: Scraping vs ToS

**Best practices**:
1. **Check ToS**: Understand what's prohibited
2. **Respect robots.txt**: It's the web standard
3. **Rate limit yourself**: Don't overload servers
4. **Don't bypass paywalls**: That's clearly wrong
5. **Academic use**: Usually safer than commercial

**When in doubt**: Ask the website owner or use public datasets.

---

# Data Quality from Scraping

**Challenges**:
- **Inconsistent formatting**: Different pages, different structures
- **Missing data**: Not all fields present
- **Dirty data**: Extra whitespace, special characters
- **Broken HTML**: Unclosed tags, invalid structure

**Solutions**:
- Defensive programming (check if element exists)
- Data validation (Week 2 topic)
- Regular expressions for cleaning
- Fallback values for missing data

---

# Summary

1.  **APIs > Scraping**: Always look for an API first (stable, legal).
2.  **Tools**: `curl` for quick checks, `requests` for scripts.
3.  **Robustness**: Handle errors, retries, and rate limits.
4.  **Ethics**: Respect `robots.txt` and server load.

**Next Up**: Now that we have data, it's probably messy. **Week 2: Data Validation**.

"""
Manual single-URL web crawl (Phase 10).

Safety rules enforced before any fetch:
  1. URL must be http(s) and well-formed.
  2. Domain must be in the configurable allowlist (config.CRAWL_ALLOWED_DOMAINS).
  3. robots.txt must permit our user-agent for that path.

No autonomous multi-page crawling — exactly one page is fetched, cleaned
(nav/footer/scripts/ads stripped) and returned as plain text for the normal
chunk+embed pipeline.
"""
from urllib.parse import urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

import config


class CrawlError(Exception):
    pass


def _domain(url):
    return (urlparse(url).hostname or "").lower()


def is_allowed_domain(url):
    host = _domain(url)
    if not host:
        return False
    if config.CRAWL_ALLOW_ALL:
        return True  # explicit opt-in to crawl any domain
    if not config.CRAWL_ALLOWED_DOMAINS:
        return False  # nothing allowed until configured
    return any(host == d or host.endswith("." + d) for d in config.CRAWL_ALLOWED_DOMAINS)


def robots_allows(url):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots.txt cannot be fetched, be conservative and disallow.
        return False
    return rp.can_fetch(config.CRAWL_USER_AGENT, url)


def clean_html(html):
    """Strip non-content elements and return readable text (Phase 10.4)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "form", "noscript", "iframe", "svg"]):
        tag.decompose()
    # Drop obvious ad / nav containers by class/id hints.
    for el in soup.find_all(attrs={"class": True}):
        cls = " ".join(el.get("class", [])).lower()
        if any(h in cls for h in ("advert", "ad-", "-ad", "cookie", "banner",
                                  "sidebar", "menu", "navbar", "social")):
            el.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    # Collapse blank lines.
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return title, cleaned


def fetch_url(url):
    """Validate, check robots + allowlist, fetch and clean a single URL.

    Returns (title, cleaned_text). Raises CrawlError on any policy failure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CrawlError("Invalid URL — must start with http:// or https://")
    if not is_allowed_domain(url):
        allowed = ", ".join(config.CRAWL_ALLOWED_DOMAINS) or "(none configured)"
        raise CrawlError(
            f"Domain '{_domain(url)}' is not allowed. Allowed: {allowed}. "
            "Add it to the 'crawl_allowed_domains' secret (comma-separated), "
            "or set 'crawl_allow_all = true' to permit any domain."
        )
    if not robots_allows(url):
        raise CrawlError("Blocked by robots.txt for this URL.")

    try:
        resp = requests.get(
            url, timeout=config.CRAWL_TIMEOUT,
            headers={"User-Agent": config.CRAWL_USER_AGENT},
            stream=True,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise CrawlError(f"Fetch failed: {e}")

    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype and "text" not in ctype:
        raise CrawlError(f"Unsupported content type: {ctype or 'unknown'}")

    content = resp.content[:config.CRAWL_MAX_BYTES]
    title, cleaned = clean_html(content.decode(resp.encoding or "utf-8", errors="ignore"))
    if not cleaned.strip():
        raise CrawlError("No readable text extracted from the page.")
    return title, cleaned

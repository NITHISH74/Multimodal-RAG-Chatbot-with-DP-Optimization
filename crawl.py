"""
Web crawling via Crawl4AI (Phase 10, v2).

Replaces the old requests+BeautifulSoup fetcher. Crawl4AI drives a headless
Chromium (Playwright), so JavaScript-rendered sites work, and emits clean
markdown that flows into the normal chunk -> embed -> pgvector pipeline.

Safety gates (unchanged from v1):
  1. URL must be http(s) and well-formed.
  2. Domain must pass the allowlist (config.CRAWL_ALLOWED_DOMAINS /
     CRAWL_ALLOW_ALL).
  3. robots.txt is respected for every page fetched.

Modes:
  * "single" — exactly one page.
  * "site"   — breadth-first crawl that stays on the start URL's host,
               dedupes URLs, respects robots.txt and stops at max_pages.

Crawl4AI (and its Chromium) is imported lazily so the rest of the app works
even where the browser stack isn't installed.
"""
import asyncio
import re
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urlparse, urldefrag

import config


class CrawlError(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────
#  URL / domain / robots helpers
# ──────────────────────────────────────────────────────────────────────
def _domain(url):
    return (urlparse(url).hostname or "").lower()


def normalize_url(url):
    """Canonical form for dedup: strip fragment + trailing slash, lower host."""
    url, _frag = urldefrag(url.strip())
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if p.port:
        host = f"{host}:{p.port}"
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = f"?{p.query}" if p.query else ""
    return f"{p.scheme}://{host}{path}{query}"


def is_allowed_domain(url):
    host = _domain(url)
    if not host:
        return False
    if config.CRAWL_ALLOW_ALL:
        return True  # explicit opt-in to crawl any domain
    if not config.CRAWL_ALLOWED_DOMAINS:
        return False  # nothing allowed until configured
    return any(host == d or host.endswith("." + d) for d in config.CRAWL_ALLOWED_DOMAINS)


_ROBOTS_CACHE = {}


def robots_state(url):
    """Per-host cached robots.txt check.

    Returns "allow", "deny", or "unreachable" (robots.txt could not be
    fetched — likely the whole site is down/unresolvable; we conservatively
    do not crawl in that case either)."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _ROBOTS_CACHE.get(base)
    if rp is None:
        rp = robotparser.RobotFileParser()
        try:
            rp.set_url(f"{base}/robots.txt")
            rp.read()
        except Exception:
            rp = False  # cache the failure too
        _ROBOTS_CACHE[base] = rp
    if rp is False:
        return "unreachable"
    return "allow" if rp.can_fetch(config.CRAWL_USER_AGENT, url) else "deny"


def robots_allows(url):
    return robots_state(url) == "allow"


# File-ish links we never enqueue during a site crawl.
_SKIP_EXT_RE = re.compile(
    r"\.(pdf|png|jpe?g|gif|svg|webp|ico|css|js|mjs|zip|gz|tar|rar|7z|exe|dmg|"
    r"mp[34]|avi|mov|woff2?|ttf|eot|xml|rss|atom)$", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────
#  Markdown post-processing
# ──────────────────────────────────────────────────────────────────────
def tidy_markdown(md):
    """Light cleanup of Crawl4AI markdown before chunking: drop repeated
    nav/cookie remnant lines, collapse blank runs, cap size."""
    if not md:
        return ""
    md = md.replace("\r\n", "\n").replace("​", "")
    out, prev = [], None
    for ln in md.split("\n"):
        s = ln.rstrip()
        if s and s == prev:          # consecutive duplicate lines
            continue
        out.append(s)
        prev = s
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md[:config.CRAWL_MAX_BYTES].strip()


def _markdown_of(result):
    """Extract the best markdown from a Crawl4AI result across versions.
    Prefers the content-filtered 'fit' markdown unless it pruned too much."""
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    fit = getattr(md, "fit_markdown", None)
    raw = getattr(md, "raw_markdown", None)
    if fit and len(fit.strip()) >= 200:
        return fit
    if raw and raw.strip():
        return raw
    return md if isinstance(md, str) else (str(md) or "")


# ──────────────────────────────────────────────────────────────────────
#  Crawl4AI plumbing (lazy import + one-time browser install)
# ──────────────────────────────────────────────────────────────────────
def _crawl4ai():
    try:
        import crawl4ai as c4
    except ImportError:
        raise CrawlError(
            "Crawl4AI is not installed. Run: pip install crawl4ai && "
            "python -m playwright install chromium")
    return c4


def _run_cfg(c4):
    try:
        md_gen = c4.DefaultMarkdownGenerator(content_filter=c4.PruningContentFilter())
    except AttributeError:  # older crawl4ai layouts
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
        from crawl4ai.content_filter_strategy import PruningContentFilter
        md_gen = DefaultMarkdownGenerator(content_filter=PruningContentFilter())
    return c4.CrawlerRunConfig(
        cache_mode=c4.CacheMode.BYPASS,
        markdown_generator=md_gen,
        excluded_tags=["nav", "footer", "header", "aside", "form"],
        remove_overlay_elements=True,    # cookie banners / modals
        exclude_external_links=True,
        page_timeout=config.CRAWL_TIMEOUT * 1000,
        word_count_threshold=5,
        verbose=False,
    )


# Remembered after the first successful launch so later crawls skip the probe.
_BROWSER_KW = None


async def _start_crawler(c4):
    """Launch a crawler, falling back from bundled Chromium to the system
    Chrome/Edge (Playwright 'channel'). The bundled build can be broken on
    some Windows hosts (side-by-side config error) while a system browser
    works fine."""
    global _BROWSER_KW
    candidates = ([_BROWSER_KW] if _BROWSER_KW is not None
                  else [{},
                        {"chrome_channel": "chrome", "channel": "chrome"},
                        {"chrome_channel": "msedge", "channel": "msedge"}])
    last_exc = None
    for kw in candidates:
        try:
            cfg = c4.BrowserConfig(headless=True, user_agent=config.CRAWL_USER_AGENT,
                                   verbose=False, **kw)
        except TypeError:        # crawl4ai too old for 'channel'
            continue
        crawler = c4.AsyncWebCrawler(config=cfg)
        try:
            await crawler.start()
            _BROWSER_KW = kw
            return crawler
        except Exception as e:
            last_exc = e
            try:
                await crawler.close()
            except Exception:
                pass
    raise last_exc if last_exc else CrawlError("No usable browser found.")


_install_lock = threading.Lock()
_install_attempted = False


def _is_browser_missing(exc):
    s = str(exc)
    return ("Executable doesn't exist" in s or "playwright install" in s
            or "BrowserType.launch" in s)


def _install_chromium():
    """One-time best-effort Chromium download (first run / fresh deploys)."""
    global _install_attempted
    with _install_lock:
        if _install_attempted:
            return False
        _install_attempted = True
        try:
            r = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True, timeout=600)
            return r.returncode == 0
        except Exception:
            return False


def _run_async(coro):
    """Run a coroutine from Streamlit's (sync) script thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box = {}
    def _runner():
        box["v"] = asyncio.run(coro)
    t = threading.Thread(target=_runner, daemon=True)
    t.start(); t.join()
    return box["v"]


# ──────────────────────────────────────────────────────────────────────
#  Core crawl (async BFS, single page == max_pages 1 without link-follow)
# ──────────────────────────────────────────────────────────────────────
def _internal_links(result):
    links = getattr(result, "links", None) or {}
    for item in links.get("internal", []):
        href = item.get("href") if isinstance(item, dict) else item
        if href:
            yield href


async def _acrawl(start_url, mode, max_pages, progress_cb):
    c4 = _crawl4ai()
    run_cfg = _run_cfg(c4)
    start_host = _domain(start_url)
    queue = deque([normalize_url(start_url)])
    seen = {queue[0]}
    pages, skipped = [], []

    crawler = await _start_crawler(c4)
    try:
        while queue and len(pages) < max_pages:
            url = queue.popleft()
            if progress_cb:
                progress_cb(len(pages), max_pages, url)
            state = robots_state(url)
            if state != "allow":
                skipped.append((url, "blocked by robots.txt" if state == "deny"
                                else "site unreachable (robots.txt could not be fetched)"))
                continue
            try:
                result = await crawler.arun(url=url, config=run_cfg)
            except Exception as e:
                skipped.append((url, f"fetch error: {e}"))
                continue
            if not getattr(result, "success", False):
                skipped.append((url, getattr(result, "error_message", None) or "fetch failed"))
                continue

            md = tidy_markdown(_markdown_of(result))
            title = ((getattr(result, "metadata", None) or {}).get("title") or "").strip()
            if md:
                pages.append({
                    "url": url,
                    "title": title or url,
                    "markdown": md,
                    "domain": start_host,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                })
            else:
                skipped.append((url, "no readable text extracted"))

            if mode == "site":
                for href in _internal_links(result):
                    try:
                        nu = normalize_url(href)
                    except Exception:
                        continue
                    if (nu in seen or _domain(nu) != start_host
                            or urlparse(nu).scheme not in ("http", "https")
                            or _SKIP_EXT_RE.search(urlparse(nu).path)
                            or not is_allowed_domain(nu)):
                        continue
                    seen.add(nu)
                    if len(queue) < max_pages * 5:   # bound queue memory
                        queue.append(nu)
    finally:
        try:
            await crawler.close()
        except Exception:
            pass
    return pages, skipped


# ──────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────
def crawl_pages(url, mode="single", max_pages=None, progress_cb=None):
    """Crawl one page ("single") or a same-domain site ("site").

    Returns (pages, skipped):
      pages   — [{url, title, markdown, domain, crawled_at}, ...]
      skipped — [(url, reason), ...]
    Raises CrawlError on policy failures (bad URL, allowlist, robots on the
    start page, browser unavailable) with user-friendly messages.
    """
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CrawlError("Invalid URL — must start with http:// or https://")
    if not is_allowed_domain(url):
        allowed = ", ".join(config.CRAWL_ALLOWED_DOMAINS) or "(none configured)"
        raise CrawlError(
            f"Domain '{_domain(url)}' is not allowed. Allowed: {allowed}. "
            "Add it to the 'crawl_allowed_domains' secret (comma-separated), "
            "or set 'crawl_allow_all = true' to permit any domain.")
    if max_pages is None or mode == "single":
        max_pages = 1 if mode == "single" else config.CRAWL_MAX_PAGES_DEFAULT
    max_pages = max(1, min(int(max_pages), config.CRAWL_MAX_PAGES_LIMIT))

    try:
        pages, skipped = _run_async(_acrawl(url, mode, max_pages, progress_cb))
    except CrawlError:
        raise
    except Exception as e:
        if _is_browser_missing(e) and _install_chromium():
            pages, skipped = _run_async(_acrawl(url, mode, max_pages, progress_cb))
        elif _is_browser_missing(e):
            raise CrawlError(
                "Headless browser unavailable. Run 'python -m playwright "
                "install chromium' (on Streamlit Cloud also add the system "
                "packages from packages.txt) and restart the app.")
        else:
            raise CrawlError(f"Crawl failed: {e}")

    if not pages and mode == "single":
        reason = skipped[0][1] if skipped else "no readable text extracted"
        if reason == "blocked by robots.txt":
            reason = "Blocked by robots.txt for this URL."
        raise CrawlError(reason)
    return pages, skipped

from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests


USER_AGENT = "AI-Daily-Stock-Analysis-Bot/1.0"


def _strip_html(value: str) -> str:
    value = html.unescape(value or "")
    return re.sub(r"<[^>]+>", "", value).strip()


def fetch_news(ticker: str, company_name: str | None = None, limit: int = 10) -> list[dict]:
    query = company_name or ticker
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles = []

    for item in root.findall(".//item")[:limit]:
        title = _strip_html(item.findtext("title", default=""))
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        source = item.findtext("source", default="Unknown")

        articles.append(
            {
                "headline": title,
                "url": link,
                "published": pub_date,
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return articles

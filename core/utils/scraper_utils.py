from typing import List
import requests
from bs4 import BeautifulSoup
import json, logging, time
import os
from datetime import datetime, timezone, timedelta

HEADERS = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    }

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# ✅ 新闻类定义
class News:
    def __init__(self, language: str, title: str, author: str, source: str, link: str, published_time: datetime, caption: str, content: str, summary: str):
        self.language = language
        self.title = title
        self.author = author
        self.source = source
        self.link = link
        self.published_time = published_time
        self.caption = caption
        self.content = caption + "\n" + content
        self.summary = summary
        self.category = "None"
        self.scraped_at = datetime.utcnow()

    def __str__(self):
        return f"""
Language: {self.language}
Title: {self.title}
Author: {self.author}
Source: {self.source}
Link: {self.link}
Published Time: {self.published_time}
Scraped At: {self.scraped_at}
Category: {self.category}
Content:\n{self.content}

Summary: {self.summary}
======================================="""

    def to_dict(self):
        return {
            "language": self.language,
            "title": self.title,
            "author": self.author,
            "source": self.source,
            "link": self.link,
            "published_time": self.published_time.isoformat(),
            "scraped_at": self.scraped_at.isoformat(),
            "content": self.content,
            "summary": self.summary,
            "category": self.category
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            language=data.get("language", ""),
            title=data.get("title", ""),
            author=data.get("author", ""),
            source=data.get("source", ""),
            link=data.get("link", ""),
            published_time=datetime.fromisoformat(data.get("published_time")),
            caption="",  # 可选：原始 dict 没有 caption 可置空
            content=data.get("content", ""),
            summary=data.get("summary", "")
        )

# ✅ 带重试的网络请求函数
def get_with_retries(url: str) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url=url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if 400 <= e.response.status_code < 500:
                logging.error(f"Client error {e.response.status_code} for {url}. Not retrying.")
                raise e
            logging.warning(f"Server error {e.response.status_code} for {url}. Retrying ({attempt + 1}/{MAX_RETRIES})...")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logging.warning(f"Network error ({type(e).__name__}) for {url}. Retrying ({attempt + 1}/{MAX_RETRIES})...")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            logging.error(f"All {MAX_RETRIES} retries failed for {url}.")
            raise e

# ✅ 抓取新闻内容
def the_edge_scrape(link: str, is_already_english: bool):
    try:
        response = get_with_retries(link)
        soup = BeautifulSoup(response.text, "html.parser")

        if not is_already_english:
            content_divs = soup.find_all("div", class_="newsTextDataWrapInner")
            english_link = None
            for div in content_divs:
                em_tag = div.find("em")
                a_tag = div.find("a", href=True)
                if em_tag:
                    em_text = em_tag.get_text(strip=True).replace(u'\xa0', u' ')
                    if "English version" in em_text:
                        english_link = a_tag["href"]
                        break

            if english_link:
                logging.info(f"Found English version: {english_link}. Scraping English content.")
                response = get_with_retries(english_link)
                soup = BeautifulSoup(response.text, "html.parser")
                link = english_link
            else:
                logging.warning(f"Article not marked as English and no English version link found for {link}. Skipping.")
                return None, "", "", None, ""

        script_tag = soup.find("script", id="__NEXT_DATA__")
        english_title = ""
        caption = ""
        published_time = None
        if script_tag:
            json_data = json.loads(script_tag.string)
            english_title = json_data.get("props", {}).get("pageProps", {}).get("data", {}).get("title", "")
            caption = json_data.get("props", {}).get("pageProps", {}).get("data", {}).get("caption", "")
            try:
                timestamp = json_data.get("props", {}).get("pageProps", {}).get("data", {}).get("created")
                if timestamp:
                    published_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone(timedelta(hours=8)))
            except (ValueError, KeyError):
                logging.warning("Could not parse the published time.")
        
        content_divs = soup.find_all("div", class_="newsTextDataWrapInner")
        
        all_paragraphs = []
        for div in content_divs:
            # Replace <br> tags with newline characters to preserve paragraph breaks
            for br in div.find_all('br'):
                br.replace_with('\n')
            
            # Get text and split by our newlines, then clean up
            text_content = div.get_text()
            paragraphs = [p.strip() for p in text_content.split('\n') if p.strip()]
            all_paragraphs.extend(paragraphs)

        clean_paragraphs = []
        for p in all_paragraphs:
            if p.strip().startswith("Read also:"):
                break
            clean_paragraphs.append(p)
        
        content = "\n".join(clean_paragraphs)
        
        return published_time, caption, content, link, english_title
    except Exception as e:
        logging.error(f"❌ Error scraping {link}: {e}", exc_info=True)
        return None, "", "", None, ""

# ✅ 抓取新闻列表
def fetch_news(offset: int, section: str, api_type: str = 'category') -> List[dict]:
    try:
        response = get_with_retries(f"https://theedgemalaysia.com/api/{api_type_to_endpoint(api_type)}?offset={offset}&{api_type_to_param(api_type)}={section}")
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        logging.error(f"❌ Failed to fetch news list for section {section}: {e}", exc_info=True)
        return []

def api_type_to_endpoint(api_type: str) -> str:
    if api_type == 'category': return "loadMoreCategories"
    if api_type == 'option': return "loadMoreOption"
    if api_type == 'flash': return "loadMoreFlashCategories"
    return ""

def api_type_to_param(api_type: str) -> str:
    if api_type == 'category': return "categories"
    if api_type == 'option': return "option"
    if api_type == 'flash': return "flash"
    return ""

# ✅ 单篇新闻处理器
def process_news_article(news, section=None):
    news_link = f"https://theedgemalaysia.com/{news['alias']}"
    is_english_from_api = news.get('language', '').lower() == 'english'

    published_time, caption, content, final_news_link, english_title = the_edge_scrape(news_link, is_already_english=is_english_from_api)

    if not final_news_link:
        return None

    # Normalize the link to a consistent domain to prevent duplicates
    final_news_link = final_news_link.replace("www.theedgemarkets.com", "theedgemalaysia.com")

    title = news['title']
    author = news['author']
    source = news['source']

    if not published_time:
        # Fallback to current time, ensuring it's offset-aware to match other timestamps
        published_time = datetime.now(timezone(timedelta(hours=8)))
        logging.warning(f"Published time could not be fetched for {final_news_link}. Using current time.")
    summary = news.get('summary', '')

    language = "English"

    # Create the News object
    news_object = News(language, english_title or title, author, source, final_news_link, published_time, caption, content, summary)
    
    # Assign the section to the section field
    if section:
        news_object.section = section
    # Explicitly set the category to None for later classification
    news_object.category = None
        
    return news_object

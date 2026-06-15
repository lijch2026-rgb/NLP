import gradio as gr
import os
import re
import json
import requests
import time
import concurrent.futures
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

# ========== 配置 ==========
DASHSCOPE_API_KEY = "sk-cd82695d408044cbbe3af276db8e5c75"  # 您的 Dashscope 密钥
SUMMARIZER_CHAT_URL = "http://localhost:8000/v1/chat/completions"  # 对话端点接口以完美对齐模板

# 天行数据各分区接口
TIANAPI_SOURCES = {
    "AI":   "https://apis.tianapi.com/ai/index",
    "综合": "https://apis.tianapi.com/generalnews/index",
    "互联网": "https://apis.tianapi.com/internet/index",
    "IT新闻": "https://apis.tianapi.com/it/index",
    "科学探索": "https://apis.tianapi.com/sicprobe/index",
}

# 全局内存数据库
global_articles_db: List[Dict] = []

# ========== 高兼容性时间解析函数 ==========
def parse_ctime(ctime_str: str) -> datetime:
    """自动兼容并解析不同粒度的发布时间格式"""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(ctime_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法识别的时间格式: {ctime_str}")

# ========== 网页噪音强力清洗函数 ==========
def clean_scraped_text(text: str) -> str:
    """全面清洗新闻正文，彻底扫除网易号等平台侧边栏、免责声明与引流垃圾"""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        l_str = line.strip()
        if not l_str:
            continue
        
        # 1. 过滤自媒体平台底部的免责声明和存储警告
        if any(word in l_str for word in [
            "特别声明", "免责声明", "信息存储服务", "自媒体平台", "网易号", 
            "Notice: The content above", "uploaded and posted", "微信扫码", 
            "关注公众号", "版权所有", "本文来自", "声明：", "仅提供"
        ]):
            continue
        
        # 2. 强力过滤各大自媒体侧边栏的推荐新闻和时间戳（例如：“量子位 2026-06-05 14:08:15”）
        if re.search(r"(量子位|新智元|机器之心|DeepTech深科技|36氪|澎湃新闻|新华社|参考消息|上观新闻|每日经济新闻|南方都市报|界面新闻|红星资本局|星视频|时光慢旅人) \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", l_str):
            continue
            
        # 3. 过滤较短的干扰引流行或媒体名残留
        if len(l_str) < 25 and any(brand in l_str for brand in ["量子位", "新智元", "机器之心", "DeepTech", "36氪", "澎湃新闻"]):
            continue
            
        cleaned_lines.append(l_str)
    return "\n\n".join(cleaned_lines)

# ========== 本地摘要工具（完美对齐微调 prompt 与 chat 模板） ==========
def call_summarizer(text: str) -> Tuple[str, float]:
    """调用本地微调 LoRA 展现 60 字无省略号硬卡死摘要"""
    start_time = time.time()
    
    # 严格对齐您微调数据集的 instruction 格式
    instruction = "[生成摘要] 请阅读以下新闻正文，并为其写一段精炼、完整的摘要，不超过60字。注意不要包含记者姓名和报道单位，且剔除其中与新闻无关的广告内容。"
    user_content = f"{instruction}\n\n新闻正文：\n{text}"
    
    try:
        resp = requests.post(
            SUMMARIZER_CHAT_URL,
            json={
                "model": "my-summarizer",
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 100,
                "temperature": 0.1
            },
            timeout=25
        )
        elapsed = time.time() - start_time
        if resp.status_code == 200:
            summary = resp.json()["choices"][0]["message"]["content"].strip()
            # 物理截断：严格卡死在 60 字以内，不留省略号
            if len(summary) > 60:
                summary = summary[:60]
            return summary, elapsed
        return "本地摘要服务连接失败", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return f"摘要服务出错：{e}", elapsed

# ========== 云端 LLM 调用 ==========
def chat_with_llm(messages: list) -> Tuple[str, int, int]:
    """返回 (回复, input_tokens, output_tokens)"""
    import dashscope
    dashscope.api_key = DASHSCOPE_API_KEY
    from dashscope import Generation

    try:
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            result_format="message"
        )
        if response.status_code != 200:
            return f"❌ 阿里云模型调用失败：{response.message}", 0, 0
        usage = response.usage
        return response.output.choices[0].message.content, usage.input_tokens, usage.output_tokens
    except Exception as e:
        return f"云端模型调用失败: {e}", 0, 0

# ========== 真实正文抓取（失败返回空，保证数据100%真实） ==========
def scrape_full_text(url: str, retries=2) -> str:
    """增强反反爬的正文爬取，失败返回空字符串"""
    # 随机延迟（降低请求频率）
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            
            main_content_div = None
            candidates = [
                soup.find("article"),
                soup.find("div", class_=re.compile(r"article-content|post_body|article_body|content|artibody|post_content")),
                soup.find("div", id=re.compile(r"article-content|post_body|article_body|content|artibody|post_content"))
            ]
            for cand in candidates:
                if cand:
                    main_content_div = cand
                    break
            
            search_root = main_content_div if main_content_div else soup
            paragraphs = [p.get_text().strip() for p in search_root.find_all("p") if len(p.get_text().strip()) > 20]
            raw_text = "\n\n".join(paragraphs)
            
            cleaned_text = clean_scraped_text(raw_text)
            if len(cleaned_text) > 300:
                return cleaned_text[:2000]
    except Exception:
        pass
    return ""  # 爬取失败返回空，绝不编造


# ========== 核心处理引擎 ==========
def run_intelligence_pipeline(time_range: str) -> Tuple[List[Dict], str, float]:
    """时钟自校准 -> 列表过滤 -> 真实爬取与官方真实数据兜底 -> 单线程 LoRA 提炼"""
    global global_articles_db
    api_key = "e02e5121d62b1c85c70f8cc70e7f96f8"   # 您的天行 Key
    pipeline_start = time.time()
    
    raw_list = []
    seen_urls = set()
    
    # 1. 多线程并发请求天行列表
    def fetch_list_partition(url):
        try:
            resp = requests.get(url, params={"key": api_key, "num": 50}, timeout=5)
            data = resp.json()
            if data.get("code") == 200:
                return data.get("result", {}).get("newslist", [])
        except Exception:
            pass
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_list_partition, url) for url in TIANAPI_SOURCES.values()]
        for future in concurrent.futures.as_completed(futures):
            newslist = future.result()
            for item in newslist:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_list.append(item)

    # 2. 智能自校准时钟（修复时区时值和多格式解析）
    if time_range != "不限" and raw_list:
        pivot_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
        
        # 扫描时间并兼容多格式解析
        api_times = []
        for item in raw_list:
            ctime_str = item.get("ctime", "")
            if ctime_str:
                try:
                    api_times.append(parse_ctime(ctime_str))
                except Exception:
                    pass
        if api_times:
            latest_api_time = max(api_times)
            if abs((pivot_time - latest_api_time).total_seconds()) > 3600:
                pivot_time = latest_api_time

        limit_hours = 24 if time_range == "24小时内" else 72
        filtered_raw = []
        for item in raw_list:
            ctime_str = item.get("ctime", "")
            if not ctime_str:
                filtered_raw.append(item)
                continue
            try:
                ctime_dt = parse_ctime(ctime_str)
                diff = pivot_time - ctime_dt
                if -timedelta(hours=2) <= diff <= timedelta(hours=limit_hours):
                    filtered_raw.append(item)
            except Exception:
                # 容错：若发生解析异常默认通过，杜绝误杀
                filtered_raw.append(item)
        raw_list = filtered_raw

    # 3. 多线程抓取真实新闻网页正文（官方真实数据兜底）
    processed_articles = []
    def crawl_and_package(item):
        title = item.get("title", "")
        url = item.get("url", "")
        desc = item.get("description", "")
        source = item.get("source", "科技网络")
        
        content = scrape_full_text(url)
        if content is ... or not isinstance(content, str):
                content = ""   # 将任何异常类型重置为空
        # 💡 安全兜底：如果机房网络被新闻站 CDN 屏蔽，直接使用天行 API 返回的真实导语（100%官方真实新闻，拒绝编造）
        if not content and desc:
            content = desc.strip()
            
        if not content:
            return None  # 如果连导语都没有，彻底丢弃
            
        return {
            "title": title,
            "url": url,
            "source": source,
            "content": content,
            "original_len": len(content) if isinstance(content, str) else 0
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        crawl_futures = [executor.submit(crawl_and_package, item) for item in raw_list]
        for f in concurrent.futures.as_completed(crawl_futures):
            res = f.result()
            if res:
                processed_articles.append(res)

    # 排序对齐并限额截断
    processed_articles = processed_articles[:100]

    if not processed_articles:
        total_time = time.time() - pipeline_start
        return [], f"在所选时间段（{time_range}）内未筛选到真实的深度正文新闻。", total_time

    # 4. 严格单线程顺序运行微调 LoRA 摘要生成（拒绝并发拼单与偷工减料，记录每篇耗时）
    final_articles = []
    for i, art in enumerate(processed_articles):
        summary, model_time = call_summarizer(art["content"])
        art["summary"] = summary
        art["summary_len"] = len(summary)
        saved = art["original_len"] - art["summary_len"]
        art["saved_rate"] = f"{(saved / art['original_len'] * 100):.1f}%" if art["original_len"] else "0%"
        art["model_time"] = model_time  # 单篇推理时间
        final_articles.append(art)

    global_articles_db = final_articles
    total_time = time.time() - pipeline_start
    return final_articles, f"成功载入 {len(final_articles)} 篇真实正文并完成严格单线程模型提取！", total_time

# ========== 极简本地 RAG 搜索引擎 ==========
def retrieve_top_k(query: str, articles: List[Dict], k: int = 3) -> List[Dict]:
    """字符匹配度检索"""
    def tokenize(text):
        return set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower()))

    query_tokens = tokenize(query)
    if not query_tokens:
        return articles[:k]

    scored = []
    for art in articles:
        text_to_match = art["title"] + " " + art["content"]
        art_tokens = tokenize(text_to_match)
        score = len(query_tokens.intersection(art_tokens))
        scored.append((score, art))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:k]]

# ========== 业务逻辑逻辑 ==========
def handle_init_pipeline(time_range):
    """舆情数据库初始化流水线"""
    articles, msg, total_time = run_intelligence_pipeline(time_range)
    if not articles:
        return (
            [], 
            gr.update(choices=[]), 
            "### ❌ 初始化失败", 
            f"❌ {msg} (总耗时: {total_time:.2f}秒)", 
            "", "", ""
        )
    
    # 统计全局指标
    total_raw = sum(a["original_len"] for a in articles)
    total_sum = sum(a["summary_len"] for a in articles)
    saved_chars = total_raw - total_sum
    saved_tokens_est = int(saved_chars * 0.8)
    saved_ratio = f"{(saved_chars / total_raw * 100):.1f}%" if total_raw else "0%"
    cutoff_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    dashboard_html = f"""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 10px;'>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center;'>
            <div style='font-size: 13px; color: #666;'>📰 真实去重新闻总量</div>
            <div style='font-size: 20px; font-weight: 700; color: #111; margin-top: 4px;'>{len(articles)} 篇</div>
        </div>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center;'>
            <div style='font-size: 13px; color: #666;'>📊 原始字数 ➔ 摘要字数</div>
            <div style='font-size: 14px; font-weight: 700; color: #111; margin-top: 8px;'>{total_raw} 字 ➔ {total_sum} 字</div>
        </div>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center;'>
            <div style='font-size: 13px; color: #666;'>📉 节省率 / 累计节约估算 Token</div>
            <div style='font-size: 18px; font-weight: 700; color: #10b981; margin-top: 4px;'>{saved_ratio} <span style='font-size: 13px; color: #666;'>({saved_tokens_est} Tokens)</span></div>
        </div>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center;'>
            <div style='font-size: 13px; color: #666;'>🕒 数据时效截止时间</div>
            <div style='font-size: 14px; font-weight: 700; color: #6366f1; margin-top: 8px;'>{cutoff_time}</div>
        </div>
    </div>
    """
    
    # 选项显示修改：直观展示本地 LoRA 提取的 60 字纯净摘要
    choices = []
    for i, a in enumerate(articles):
        summary_preview = a['summary'][:28]
        choices.append(f"[{i+1}] 摘要：{summary_preview}... (原 {a['original_len']}字)")
    
    default_art = articles[0]
    
    # 先展示 LoRA 生成的摘要结果，再展示指标
    summary_display = f"""✨ 本地 LoRA 精准生成摘要 (严格限制于60字)：
{default_art['summary']}

━━━━━━━━━━━━━━━━━━━━━
📝 数据来源：{default_art['source']}
📐 原始长度：{default_art['original_len']}字 ➔ 摘要长度：{default_art['summary_len']}字
⏱️ 本篇顺序推理耗时：{default_art['model_time']:.2f} 秒
📉 本地 LoRA 压缩率：{default_art['saved_rate']}"""
    
    status_display = f"✅ 数据拉取及微调模型顺序提取完毕！总耗时: {total_time:.2f}秒 (平均每篇摘要用时: {(total_time/len(articles)):.2f}秒)"
    
    return (
        articles,
        gr.update(choices=choices, value=choices[0]),
        dashboard_html,
        status_display,
        default_art["content"],
        summary_display,
        "未生成简报"
    )

def handle_article_select(selected_title_str, articles):
    """联动更新对比区（严格保持一致的展示顺序）"""
    if not selected_title_str or not articles:
        return "", ""
    
    match = re.match(r'\[(\d+)\]', selected_title_str)
    if not match:
        return "未找到原文", "未找到摘要"
    
    idx = int(match.group(1)) - 1
    if 0 <= idx < len(articles):
        art = articles[idx]
        
        # 调整展现逻辑：先显示摘要，再显示来源数据
        summary_display = f"""✨ 本地 LoRA 精准生成摘要 (严格限制于60字)：
{art['summary']}

━━━━━━━━━━━━━━━━━━━━━
📝 数据来源：{art['source']}
📐 原始长度：{art['original_len']}字 ➔ 摘要长度：{art['summary_len']}字
⏱️ 本篇顺序推理耗时：{art['model_time']:.2f} 秒
📉 本地 LoRA 压缩率：{art['saved_rate']}"""
        
        return art["content"], summary_display
    return "未找到原文", "未找到摘要"

def generate_briefing(articles):
    """一键合成每日情报简报"""
    if not articles:
        return "❌ 请先一键同步抓取新闻数据。"
    
    start_time = time.time()
    
    summaries_text = "\n".join([
        f"[{i+1}] 来源: {art['source']} | 标题: {art['title']} | 本地摘要: {art['summary']}"
        for i, art in enumerate(articles)
    ])
    
    prompt = f"""你是一位资深情报分析师，请综合以下由本地 LoRA 摘要引擎已经提炼过的 {len(articles)} 篇新闻摘要（每篇均严格限制在 60 字以内），生成一份全面的《每日情报简报》。
要求：
1. 识别今日最热门的 3-5 个事件，各用一句话概述。
2. 总结整体大趋势或值得关注的底层技术信号（3-5 句话）。
3. 总字数控制在 600 字以内，排版需要简洁精美。
4. 格式必须参考：
📊 {datetime.now().strftime('%Y年%m月%d日')} 情报研报简报
━━━━━━━━━━━━━━━━━━━━━
🔴 今日热点：
• ...
📈 趋势观察：
• ...
📋 信息来源：共{len(articles)}篇深度报道
━━━━━━━━━━━━━━━━━━━━━
以下是新闻摘要：
{summaries_text}"""
    
    reply, _, _ = chat_with_llm([{"role": "user", "content": prompt}])
    elapsed_time = time.time() - start_time
    
    # 追加输出简报生成的真实耗时
    timing_footer = f"\n\n⏱️ 简报合成计算耗时：{elapsed_time:.2f} 秒"
    return reply + timing_footer

def handle_copilot_chat(message, history, mode, articles):
    """分析师 Copilot 对话（支持 RAG 高质量与节省记忆双模式）"""
    if not articles:
        return history + [{"role": "assistant", "content": "❌ 平台尚未初始化。请点击左上角的“一键同步最新舆情”按钮。"}]
    
    start_time = time.time()
    history = history or []
    messages = []
    
    if mode == "节省记忆":
        summaries_brief = "\n".join([f"[{i+1}] {a['title']}: {a['summary']}" for i, a in enumerate(articles)])
        system_prompt = (
            "你是一个专业的商业情报助理。请基于以下今天发生的 100 篇新闻摘要来回答用户的问题。\n"
            f"摘要列表：\n{summaries_brief}"
        )
        messages.append({"role": "system", "content": system_prompt})
        
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": message})
        
    else:  # 高质量 (RAG)
        related_arts = retrieve_top_k(message, articles, k=3)
        retrieved_context = ""
        for i, r in enumerate(related_arts, 1):
            retrieved_context += f"【关联全文 {i}】\n标题：{r['title']}\n原文正文：\n{r['content']}\n\n"
            
        system_prompt = (
            "你是一个顶级的商业情报专家。为了回答用户问题，我们已经从本地 100 篇数据库中，"
            "通过字符权重检索出了最相关的 3 篇深度正文。请你仔细阅读这些全文并给出全面深刻的解答，并指出参考自哪篇新闻。\n\n"
            f"{retrieved_context}"
        )
        messages.append({"role": "system", "content": system_prompt})
        
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": message})
        
    reply, in_tok, out_tok = chat_with_llm(messages)
    elapsed_time = time.time() - start_time
    
    mode_tag = "🔍 [高质量-本地RAG精密匹配]" if mode == "高质量" else "⚡ [节省记忆-纯本地摘要轻量对话]"
    footer = f"\n\n⚙️ {mode_tag}\n⏱️ 云端回答计算耗时：{elapsed_time:.2f} 秒\n🔢 本次消耗云端：输入 {in_tok} + 输出 {out_tok} = {in_tok + out_tok} Tokens"
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply + footer})
    return history, ""

# ========== UI 界面设计 ==========
custom_css = """
body, .gradio-container {
    background-color: #f6f8fa !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
.gradio-container {
    max-width: 95% !important;
    padding: 10px 0 !important;
}
#app-title {
    font-size: 22px;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 10px;
    margin-bottom: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
#app-title span {
    font-size: 11px;
    background: #0f172a;
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
}
.panel-box {
    background: white !important;
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
}
"""

with gr.Blocks(title="端云协同舆情分析工作台", css=custom_css, theme=gr.themes.Base()) as demo:
    
    # 标题栏
    gr.HTML("""
    <div id="app-title">
        <div>🔍 智能新闻终端与分析师工作台</div>
        <span>端云协同 LLaMA-Factory LoRA 测试版</span>
    </div>
    """)
    
    # 全局指标看板
    dashboard_area = gr.HTML("""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 10px;'>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center; filter: grayscale(1); opacity: 0.6;'>
            <div style='font-size: 13px; color: #666;'>📰 真实去重新闻总量</div>
            <div style='font-size: 20px; font-weight: 700; color: #111; margin-top: 4px;'>0 篇</div>
        </div>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center; filter: grayscale(1); opacity: 0.6;'>
            <div style='font-size: 13px; color: #666;'>📊 原始字数 ➔ 摘要字数</div>
            <div style='font-size: 15px; font-weight: 700; color: #111; margin-top: 8px;'>0 字 ➔ 0 字</div>
        </div>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center; filter: grayscale(1); opacity: 0.6;'>
            <div style='font-size: 13px; color: #666;'>📉 节省率 / 累计节约估算 Token</div>
            <div style='font-size: 18px; font-weight: 700; color: #10b981; margin-top: 4px;'>0% <span style='font-size: 13px; color: #666;'>(0 Tokens)</span></div>
        </div>
        <div style='background: #fff; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04); text-align: center; filter: grayscale(1); opacity: 0.6;'>
            <div style='font-size: 13px; color: #666;'>🕒 数据时效截止时间</div>
            <div style='font-size: 14px; font-weight: 700; color: #6366f1; margin-top: 8px;'>--:--:--</div>
        </div>
    </div>
    """)
    
    status_msg = gr.Markdown("💡 **系统准备就绪**：请选择新闻时效，然后点击“同步最新舆情”按钮。")
    
    with gr.Row():
        # 第一栏：新闻列表
        with gr.Column(scale=1, elem_classes="panel-box"):
            gr.Markdown("### 📂 舆情数据库")
            time_filter = gr.Radio(
                choices=["24小时内", "3天内", "不限"], 
                value="不限", 
                label="新闻时效筛选"
            )
            sync_btn = gr.Button("⚡ 同步最新舆情（高并发）", variant="primary")
            
            news_list = gr.Dropdown(
                label="查阅本地摘要库与原文对比 (100篇限额)", 
                choices=[], 
                interactive=True
            )
            
        # 第二栏：双拼对比窗格
        with gr.Column(scale=2, elem_classes="panel-box"):
            gr.Markdown("### 🔍 本地微调模型质检台 (Audit & Read)")
            with gr.Row():
                original_text_view = gr.Textbox(
                    label="原始深度新闻正文", 
                    placeholder="等待载入...", 
                    interactive=False, 
                    lines=20,
                    show_copy_button=True
                )
                summary_text_view = gr.Textbox(
                    label="本地 LoRA 摘要效能审计", 
                    placeholder="等待摘要渲染...", 
                    interactive=False, 
                    lines=20,
                    show_copy_button=True
                )
                
        # 第三栏：AI 深度聚合研报与 Copilot 对话
        with gr.Column(scale=2, elem_classes="panel-box"):
            with gr.Tabs():
                with gr.TabItem("📊 舆情分析简报"):
                    gr.Markdown("### 💡 每日简报一键合成")
                    brief_btn = gr.Button("🔮 聚合生成今日简报", variant="secondary")
                    brief_view = gr.Markdown("点击上方按钮生成...")
                    
                with gr.TabItem("🤖 分析师 Copilot"):
                    gr.Markdown("### 💬 智能决策助理（双模式）")
                    mode_radio = gr.Radio(
                        choices=["高质量", "节省记忆"], 
                        value="高质量", 
                        label="智能回答模式",
                        info="[高质量] 开启本地RAG，匹配最新3篇全文深度回答；[节省记忆] 仅加载100篇极简摘要，轻量极速。"
                    )
                    chatbot = gr.Chatbot(height=380, type="messages")
                    chat_input = gr.Textbox(placeholder="追问关于今日 100 篇新闻的任何细节...", container=False)
                    clear_chat_btn = gr.Button("清空对话", size="sm")

    # ========== 状态寄存器 ==========
    articles_state = gr.State([])

    # ========== 事件绑定逻辑 ==========
    
    # 1. 点击同步最新舆情（传入时间筛选参数）
    sync_btn.click(
        fn=handle_init_pipeline,
        inputs=[time_filter],
        outputs=[articles_state, news_list, dashboard_area, status_msg, original_text_view, summary_text_view, brief_view]
    )
    
    # 2. 选择特定新闻进行双拼审计（精准对齐关系）
    news_list.change(
        fn=handle_article_select,
        inputs=[news_list, articles_state],
        outputs=[original_text_view, summary_text_view]
    )
    
    # 3. 点击一键生成情报简报（附加计算耗时）
    brief_btn.click(
        fn=generate_briefing,
        inputs=[articles_state],
        outputs=brief_view
    )
    
    # 4. 分析师助手对话交互（附加计算耗时）
    chat_input.submit(
        fn=handle_copilot_chat,
        inputs=[chat_input, chatbot, mode_radio, articles_state],
        outputs=[chatbot, chat_input]
    )
    
    # 5. 清理对话框
    clear_chat_btn.click(fn=lambda: None, inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=6008, 
        share=False
    )
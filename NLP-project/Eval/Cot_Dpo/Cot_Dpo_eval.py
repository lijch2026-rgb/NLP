import os
import json
import numpy as np
import torch
import re
import warnings
import threading
from typing import List
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

from bert_score import score as bert_score
from rouge_score import rouge_scorer

print_lock = threading.Lock()

class AcademicCharTokenizer:
    def tokenize(self, text):
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        return [char for char in cleaned_text if char.strip()]

class NewsSummaryEvaluator:
    def __init__(self):
        print("⏳ 初始化评测系统...")
        # 单一 DeepSeek 客户端，用于调用 flash 和 pro 两个模型
        self.ds_client = OpenAI(
            api_key="sk-7839d2f420404408a55eb59691f90581",
            base_url="https://api.deepseek.com"
        )
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=False,
            tokenizer=AcademicCharTokenizer()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("✅ DeepSeek Flash+Pro 双评分矩阵及学术级词汇评测器就绪！")

    def _call_deepseek(self, prompt: str, sample_id: int, model_name: str, call_index: int = 1, max_retries: int = 3) -> float:
        for attempt in range(max_retries):
            try:
                resp = self.ds_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}} if "pro" in model_name else {}
                )
                content = resp.choices[0].message.content.strip()
                match = re.search(r'<score>\s*(\d+)\s*</score>', content)
                return float(match.group(1)) if match else 0.0
            except Exception as e:
                wait_time = 2 ** attempt
                with print_lock:
                    print(f"\n⚠️ [样本 {sample_id} {model_name} 第{call_index}次] 异常 (尝试 {attempt+1}): {e}")
                    print(f"   ↳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        with print_lock:
            print(f"\n❌ [样本 {sample_id}] {model_name} 重试 {max_retries} 次后失败！")
        return None

    def evaluate_factuality(self, source: str, generated: str, sample_id: int) -> float:
        prompt = f"""
        你是严谨的资深新闻事实核查专家。请严格对比【原新闻】和【生成摘要】，检查摘要是否存在事实幻觉。

        核查维度：
        1. 实体一致性：人名、机构名、地名是否与原文一致？
        2. 数据准确性：时间、金额、比例等数值是否正确？
        3. 逻辑与因果：是否存在因果倒置、张冠李戴、无中生有的情节？
        4. 其他关键要素：是否遗漏了决定新闻性质的重大事实，或存在其他形式的语义偏离等？

        请先简明扼要地列出你的核查分析过程（找出具体的不一致之处）。然后，请根据你对新闻事实重要性的理解，自行判断这些错误的严重程度并决定扣分幅度。
        最后一行独立使用 <score>最终分数</score> 给出打分（0-100整数）。

        【原新闻】：{source}
        【生成摘要】：{generated}
        """
        score_flash = self._call_deepseek(prompt, sample_id, "deepseek-v4-flash", call_index=1)
        score_pro   = self._call_deepseek(prompt, sample_id, "deepseek-v4-pro", call_index=2)

        valid_scores = [s for s in [score_flash, score_pro] if s is not None]
        if len(valid_scores) == 2:
            return sum(valid_scores) / 2.0
        elif len(valid_scores) == 1:
            return valid_scores[0]
        else:
            with print_lock:
                print(f"🚨 [样本 {sample_id}] Flash+Pro 全部请求失败！兜底 50 分。")
            return 50.0

    def evaluate_semantic_batch(self, references: List[str], generateds: List[str]) -> List[float]:
        local_model_path = "/root/models/bge-m3"
        P, R, F1 = bert_score(generateds, references, lang="zh", model_type=local_model_path, num_layers=24, batch_size=32, verbose=False, device=self.device)
        return [min(f.item() * 100, 100.0) for f in F1]

    def evaluate_lexical(self, reference: str, generated: str) -> float:
        if not generated.strip():
            return 0.0
        scores = self.rouge_scorer.score(reference, generated)
        r1 = scores['rouge1'].fmeasure * 100
        r2 = scores['rouge2'].fmeasure * 100
        rl = scores['rougeL'].fmeasure * 100
        return min((r1 + r2 + rl) / 3.0, 100.0)

    def evaluate_compliance(self, generated: str, max_length: int = 60) -> float:
        chars = re.findall(r'[\u4e00-\u9fffa-zA-Z0-9]', generated)
        return 1.0 if len(chars) <= max_length else 0.0

if __name__ == "__main__":
    PREDICT_FILE = "/root/autodl-tmp/LLaMA-Factory/outputs/dpo/generated_predictions.jsonl"
    evaluator = NewsSummaryEvaluator()
    news_sources, references, generateds = [], [], []
    with open(PREDICT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompt_text = data.get("prompt", "") or data.get("source", "")
            news_sources.append(prompt_text)
            raw_label = data.get("label", "") or data.get("reference", "")
            if "【标题】：" in raw_label:
                clean_label = raw_label.split("【标题】：")[-1].strip()
            else:
                clean_label = raw_label.strip()
            if clean_label.startswith("标题："):
                clean_label = clean_label[3:].strip()
            references.append(clean_label)
            raw_predict = data.get("predict", "") or data.get("generated", "")
            if "【标题】：" in raw_predict:
                clean_predict = raw_predict.split("【标题】：")[-1].strip()
            else:
                clean_predict = raw_predict.strip()
            if clean_predict.startswith("标题："):
                clean_predict = clean_predict[3:].strip()
            generateds.append(clean_predict)
    total_samples = len(generateds)
    print("🧠 正在批量计算语义等价性 (BERTScore)...")
    semantic_scores = evaluator.evaluate_semantic_batch(references, generateds)
    final_scores = [0.0] * total_samples
    fact_scores = [0.0] * total_samples
    lex_scores = [0.0] * total_samples
    compliance_scores = [0.0] * total_samples
    detailed_results = [None] * total_samples
    global_counter = 0
    compliance_count = 0

    def process_single_sample(i):
        global global_counter, compliance_count
        src, ref, gen = news_sources[i], references[i], generateds[i]
        ref = ref.strip()
        gen = gen.strip()
        match = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', src, flags=re.DOTALL)
        if match:
            clean_src = match.group(1).strip()
        else:
            clean_src = re.sub(r'<\|.*?\|>', '', src).replace('system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.', '').replace('user\n', '').strip()
        comp_score = evaluator.evaluate_compliance(gen)
        sem_score = semantic_scores[i]
        sample_id = i + 1
        if comp_score == 0:
            fact_score, lex_score, final_score = 0.0, 0.0, 0.0
            detailed_results[i] = {
                "id": sample_id, "source": clean_src, "reference": ref, "generated": gen,
                "metrics": {"Compliance": 0.0, "Semantic": round(sem_score, 2), "Factuality": 0.0, "Lexical": 0.0},
                "final_score": 0.0
            }
        else:
            fact_score = evaluator.evaluate_factuality(clean_src, gen, sample_id)
            lex_score = evaluator.evaluate_lexical(ref, gen)
            final_score = (fact_score * 0.5 + sem_score * 0.3 + lex_score * 0.2) * comp_score
            detailed_results[i] = {
                "id": sample_id, "source": clean_src, "reference": ref, "generated": gen,
                "metrics": {"Compliance": 1.0, "Semantic": round(sem_score, 2), "Factuality": round(fact_score, 2), "Lexical": round(lex_score, 2)},
                "final_score": round(final_score, 2)
            }
        fact_scores[i] = fact_score
        lex_scores[i] = lex_score
        final_scores[i] = final_score
        compliance_scores[i] = comp_score
        with print_lock:
            global_counter += 1
            compliance_count += comp_score
            if global_counter <= 3:
                print(f"\n\n{'='*15} 🔎 抽查样本 {global_counter} (索引: {sample_id}) {'='*15}")
                print(f"📝【原新闻前100字】: {clean_src[:100]}...")
                print(f"🎯【参考摘要】: '{ref}'")
                print(f"🤖【生成摘要】: '{gen}'")
                print(f"📊【得分】: 合规: {comp_score} | 语义: {sem_score:.2f} | 词汇: {lex_score:.2f} | 忠实(Flash+Pro): {fact_score:.2f} => 总分: {final_score:.2f}")
                print("="*45 + "\n")
            if global_counter % 100 == 0:
                valid_finals = [final_scores[idx] for idx in range(total_samples) if detailed_results[idx] is not None]
                valid_facts = [fact_scores[idx] for idx in range(total_samples) if detailed_results[idx] is not None and compliance_scores[idx] > 0]
                valid_sems = [semantic_scores[idx] for idx in range(total_samples) if detailed_results[idx] is not None and compliance_scores[idx] > 0]
                valid_lexs = [lex_scores[idx] for idx in range(total_samples) if detailed_results[idx] is not None and compliance_scores[idx] > 0]
                print(f"\n\n{'-'*15} 🕒 实时观测点：已完成 {global_counter}/{total_samples} 条 {'-'*15}")
                print(f"📏 合规率: {(compliance_count / global_counter) * 100:.2f}%")
                print(f"🧠 平均忠实: {np.mean(valid_facts) if valid_facts else 0:.2f}")
                print(f"📝 平均语义: {np.mean(valid_sems) if valid_sems else 0:.2f}")
                print(f"🎯 平均词汇: {np.mean(valid_lexs) if valid_lexs else 0:.2f}")
                print(f"⭐ 当前均分: {np.mean(valid_finals) if valid_finals else 0:.2f}")
                print(f"{'-' * 60}\n")

    print("📊 全异步调用 DeepSeek Flash+Pro 评分中...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # 降低并发，防止限流
        list(tqdm(executor.map(process_single_sample, range(total_samples)), total=total_samples, desc="DPO 评测进度"))

    print("\n" + "="*50)
    print(f"🏆 DPO 模型评测报告 (样本数: {total_samples})")
    print(f"📏 合规率: {(compliance_count/total_samples)*100:.2f}%")
    valid_fact = [f for f, c in zip(fact_scores, compliance_scores) if c > 0]
    valid_sem = [s for s, c in zip(semantic_scores, compliance_scores) if c > 0]
    valid_lex = [l for l, c in zip(lex_scores, compliance_scores) if c > 0]
    print(f"🧠 平均忠实度 (Flash+Pro): {np.mean(valid_fact) if valid_fact else 0:.2f}")
    print(f"📝 平均语义: {np.mean(valid_sem) if valid_sem else 0:.2f}")
    print(f"🎯 平均词汇: {np.mean(valid_lex) if valid_lex else 0:.2f}")
    print(f"⭐ 最终综合得分: {np.mean(final_scores):.2f}")
    print("="*50)

    save_dir = os.path.dirname(PREDICT_FILE)
    radar_data = {
        "metrics": {
            "Factuality": round(np.mean(fact_scores) if fact_scores else 0, 2),
            "Semantic": round(np.mean(semantic_scores) if semantic_scores else 0, 2),
            "Lexical": round(np.mean(lex_scores) if lex_scores else 0, 2),
            "Compliance": round((compliance_count/total_samples)*100, 2)
        },
        "final_score": round(np.mean(final_scores) if final_scores else 0, 2)
    }
    with open(os.path.join(save_dir, "radar_chart_data.json"), 'w', encoding='utf-8') as f:
        json.dump(radar_data, f, ensure_ascii=False, indent=4)
    detailed_path = os.path.join(save_dir, "detailed_evaluation_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"💾 结果已保存至 {save_dir}")

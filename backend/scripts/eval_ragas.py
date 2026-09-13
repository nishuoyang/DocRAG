"""RAGAS 自动化评估脚本：参数化 CLI + Markdown/JSON 报告。

运行（在 backend/ 下，用独立 venv 避免污染主环境）：
    ./.venv-ragas/Scripts/python.exe scripts/eval_ragas.py [参数]

流程：逐条走真实检索链路生成回答（结果缓存到 .ragas_cache.json，重跑直接复用），
再对每条执行 4 项指标判分（faithfulness / answer_relevancy / context_precision / context_recall），
输出带时间戳的 Markdown 报告 + JSON 原始数据到 reports/（可用 --report-dir 覆盖）。

主要参数（--help 查看全部）：
    --testset PATH        测试集文件（默认 docs/test.py，相对项目根；RAGAS_TESTSET 可覆盖）
    --top-k N             检索返回块数（默认读 .env TOP_K，即 5）
    --no-transform        关闭 Query Transformation 改写
    --no-hyde             关闭 HYDE 假想答案检索
    --no-rerank           关闭 Rerank 重排序
    --no-cache            忽略缓存强制重新生成（换文档库/检索参数后必须）
    --report-dir DIR      报告输出目录（默认 reports/，相对项目根）
    --nan-handling MODE   NaN 处理策略：keep（默认，统计跳过 NaN 并标注）/ drop / zero
    --judge-raw           判分 LLM 不包装 LangchainLLMWrapper（回退旧行为，用于对比验证）

NaN 处理说明：
    ragas 0.4.3 下 NaN 的主要来源（已按本项目链路核对源码）：
    1) 判分 LLM（DeepSeek）输出非严格 JSON（```json 围栏/夹杂解释/字段缺失）→ PydanticPrompt
       对 LangChain LLM 直连分支不重试直接解析失败 → evaluate(raise_exceptions=False) 吞成 NaN。
       对策：默认把 judge 包装成 LangchainLLMWrapper，走 extract_json + fix_output_format 重试；
       仍失败时用 --judge-raw 与旧行为对比，或换更稳的判分模型（RAGAS_LLM_MODEL）。
    2) 回答为空 / 兜底回答 → faithfulness 提取不出 statements → NaN。生成阶段已拦截空回答。
    3) ground_truth 为空 → context_recall/precision 判分异常 → NaN。已跳过并计数。
    4) 缓存脏数据（空 response/contexts）→ 自动识别并强制重新生成。
    5) answer_relevancy 依赖 embedding：gen_questions 全空或 embedding 异常 → NaN。同属 LLM
       JSON 解析问题，包装 judge 后显著减少；仍出现时检查 SiliconFlow embedding 是否限流/超时。
    脚本在判分后输出逐指标 NaN 统计与涉及题号（含回答摘要），结合 ragas 的 warning 日志
    （"No statements were generated" / "did not return a valid classification" 等）即可定位根因。
    keep 口径下均值用 pandas skipna（NaN 不计入），报告 meta 的 nan_counts 记录每指标 NaN 条数。

Judge LLM 与 Embedding 均复用 .env 配置（DeepSeek + SiliconFlow bge-m3），不改 .env。
兼容环境变量（CLI 优先）：
    RAGAS_TESTSET / RAGAS_CACHE / RAGAS_NO_CACHE / RAGAS_LLM_MODEL / RAGAS_LLM_BASE_URL / RAGAS_LLM_API_KEY
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# 显示 ragas 判分日志（"No statements were generated" / "did not return a valid classification"
# / "Invalid response format" 等 warning 是定位 NaN 的第一手线索）
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(name)s: %(message)s",
)

# 把 backend/ 加入 sys.path，复用项目代码与 .env 配置
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# 加载 backend/.env（项目未安装 python-dotenv 自动加载机制，脚本需手动注入环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from langchain_core.documents import Document  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import SingleTurnSample, EvaluationDataset  # noqa: E402
# 单例指标对象（旧版 MetricWithLLM 实例，evaluate 类型校验可通过）
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from pymilvus import connections  # noqa: E402

from config import get_settings  # noqa: E402
from core import retrieval  # noqa: E402
from core.eval_cache import (  # noqa: E402
    build_cache_fingerprint,
    load_cache_items,
    wrap_cache_items,
)
from db import milvus  # noqa: E402

# 建立 Milvus 连接（正常由 main.py 启动时建立，脚本需手动；必须用 uri 形式，
# pymilvus 2.5.18 只传 host/port 会走环境变量分支报 ConnLackConf）
settings = get_settings()
connections.connect(alias="default", uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

# 检索开关与默认参数（与 .env / 前端默认 Top-K 对应）
DEFAULT_TESTSET = BACKEND_DIR.parent / "docs" / "test.py"
DEFAULT_CACHE = BACKEND_DIR / ".ragas_cache.json"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGAS 自动化评估：逐条走真实检索链路生成回答并判分，输出 Markdown + JSON 报告"
    )
    parser.add_argument("--testset", default=None, help="测试集文件路径（默认 docs/test.py，相对项目根）")
    parser.add_argument("--top-k", type=int, default=None, help="检索返回块数（默认读 .env TOP_K，即 5）")
    parser.add_argument("--no-transform", action="store_true", help="关闭 Query Transformation 改写")
    parser.add_argument("--no-hyde", action="store_true", help="关闭 HYDE 假想答案检索")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 Rerank 重排序")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存，强制重新走检索链路生成")
    parser.add_argument("--report-dir", default="reports", help="报告输出目录（默认 reports/，相对项目根）")
    parser.add_argument(
        "--nan-handling",
        choices=["keep", "drop", "zero"],
        default="keep",
        help="NaN 处理策略（默认 keep）：keep=聚合统计跳过 NaN（skipna）并在报告中标注；"
        "drop=只统计无 NaN 的行；zero=把 NaN 视为 0 分计入。逐条明细始终全量展示，NaN 显示为 -",
    )
    parser.add_argument(
        "--judge-raw",
        action="store_true",
        help="判分 LLM 不包装 LangchainLLMWrapper（回退旧行为）。默认包装后走 ragas 的 "
        "extract_json + fix_output_format 重试分支，能显著降低 DeepSeek 返回非严格 JSON 导致的 NaN",
    )
    return parser.parse_args()


def _resolve_project_path(path: str) -> Path:
    """相对路径基于项目根解析（与 RAGAS_TESTSET 默认值一致）。"""
    p = Path(path)
    return p if p.is_absolute() else BACKEND_DIR.parent / p


def load_testset(testset_path: Path) -> tuple[list[str], list[list[str]]]:
    """从测试集文件加载 questions / ground_truths 列表（Python 文件 exec 加载）。"""
    if not testset_path.exists():
        raise FileNotFoundError(f"测试集文件不存在: {testset_path}")
    ns: dict = {}
    exec(compile(testset_path.read_text(encoding="utf-8"), str(testset_path), "exec"), ns)
    questions = ns["questions"]
    ground_truths = ns["ground_truths"]
    assert len(questions) == len(ground_truths), "questions 与 ground_truths 数量不一致"
    return questions, ground_truths


def generate_answer(
    query: str,
    top_k: int | None,
) -> tuple[str, list[Document], dict]:
    """走与 /chat 完全相同的链路：检索 → 拼上下文 → LLM 生成。"""
    retrieval_start = time.perf_counter()
    docs = retrieval._retrieve(query, top_k)
    retrieval_s = time.perf_counter() - retrieval_start
    if not docs:
        return "资料库中尚未检索到相关内容。", [], {
            "retrieval_s": round(retrieval_s, 4),
            "generation_s": 0.0,
            "context_chars": 0,
            "source_count": 0,
        }
    messages = retrieval._build_messages(query, docs)
    generation_start = time.perf_counter()
    answer = retrieval._generate_answer(messages).content
    generation_s = time.perf_counter() - generation_start
    return answer, docs, {
        "retrieval_s": round(retrieval_s, 4),
        "generation_s": round(generation_s, 4),
        "context_chars": sum(len(doc.page_content) for doc in docs),
        "source_count": len(docs),
    }


def load_cache(fingerprint: str) -> dict:
    """读取上次生成的缓存（question → {response, contexts}）。"""
    path = Path(os.getenv("RAGAS_CACHE", str(DEFAULT_CACHE)))
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return load_cache_items(raw, fingerprint)
    return {}


def save_cache(cache: dict, fingerprint: str) -> None:
    path = Path(os.getenv("RAGAS_CACHE", str(DEFAULT_CACHE)))
    path.write_text(
        json.dumps(wrap_cache_items(cache, fingerprint), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def build_run_fingerprint(top_k: int | None) -> tuple[str, dict]:
    """Fingerprint all non-secret inputs that can change generated eval answers."""
    settings_obj = get_settings()
    docs = milvus.get_all_documents()
    manifest = {
        "collection": milvus.get_collection_name(),
        "schema_version": settings_obj.COLLECTION_SCHEMA_VERSION,
        "embedding_model": settings_obj.EMBEDDING_MODEL,
        "embedding_dim": settings_obj.EMBEDDING_DIM,
        "rerank_enabled": settings_obj.RERANK_ENABLED,
        "rerank_model": settings_obj.RERANK_MODEL if settings_obj.RERANK_ENABLED else "",
        "query_transform": settings_obj.QUERY_TRANSFORM,
        "hyde": settings_obj.HYDE,
        "retrieval_mode": settings_obj.RETRIEVAL_MODE,
        "top_k": top_k or settings_obj.TOP_K,
        "retrieval_candidate_k": settings_obj.RETRIEVAL_CANDIDATE_K,
        "retrieval_max_parents": settings_obj.RETRIEVAL_MAX_PARENTS,
        "child_chunk_size": settings_obj.CHILD_CHUNK_SIZE,
        "parent_chunk_size": settings_obj.PARENT_CHUNK_SIZE,
        "parent_context_max_chars": settings_obj.PARENT_CONTEXT_MAX_CHARS,
        "corpus": {
            "chunks": len(docs),
            "pk_digest": build_cache_fingerprint(
                {"pks": sorted(str(d.metadata.get("pk") or "") for d in docs)}
            ),
            "max_upload_time": max(
                (int(d.metadata.get("upload_time") or 0) for d in docs),
                default=0,
            ),
        },
    }
    return build_cache_fingerprint(manifest), manifest


def override_settings(args: argparse.Namespace) -> dict:
    """运行时覆盖检索开关（get_settings 单例，进程级生效），返回原始值便于恢复。"""
    settings_obj = get_settings()
    originals = {}
    for field, flag in [
        ("QUERY_TRANSFORM", args.no_transform),
        ("HYDE", args.no_hyde),
        ("RERANK_ENABLED", args.no_rerank),
    ]:
        if flag:
            originals[field] = getattr(settings_obj, field)
            setattr(settings_obj, field, False)
    return originals


def restore_and_warn(originals: dict) -> None:
    """恢复被覆盖的开关值（保持进程内状态干净），并提示 .env 未变。"""
    settings_obj = get_settings()
    for field, old_val in originals.items():
        setattr(settings_obj, field, old_val)
    if originals:
        print("注意：检索开关仅本次脚本运行时覆盖，.env 和服务器配置未改变。")


def _aggregate_run_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summary = {}
    for key in ("retrieval_s", "generation_s", "context_chars", "source_count"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": round(float(np.mean(values)), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
        }
    return summary


def run_generation(
    questions: list[str],
    ground_truths: list[list[str]],
    top_k: int | None,
    no_cache: bool,
    fingerprint: str,
) -> tuple[list[SingleTurnSample], dict, dict]:
    """逐条生成回答（缓存命中复用，否则走检索链路），返回 (samples, cache, 统计)。"""
    cache = {} if no_cache else load_cache(fingerprint)
    hit = 0 if no_cache else sum(q in cache for q in questions)
    print(f"加载 {len(questions)} 条测试题（缓存命中 {hit} 条" + ("，--no-cache 强制重新生成" if no_cache else "") + "）\n")

    samples = []
    failures = 0
    latency_rows: list[dict] = []
    for i, (q, refs) in enumerate(zip(questions, ground_truths), 1):
        # ground_truth 为空 → context_recall/context_precision 判分必然异常（NaN），直接跳过并说明
        ref = refs[0] if isinstance(refs, list) else refs
        if not isinstance(ref, str) or not ref.strip():
            failures += 1
            print(f"  [{i}/{len(questions)}] 跳过：ground_truth 为空（该题无法评估 context_recall/precision）")
            continue

        from_cache = q in cache
        try:
            if from_cache:
                entry = cache[q]
                answer, contexts = entry.get("response"), entry.get("contexts")
                metrics = entry.get("metrics") or {}
                # 缓存数据无效（空回答 / 空上下文，常见于旧缓存或判分失败后回填）→ 强制重新生成
                if not answer or not answer.strip() or not contexts:
                    print(f"  [{i}/{len(questions)}] 缓存数据无效（空 response/contexts），重新生成")
                    from_cache = False
            if not from_cache:
                answer, docs, metrics = generate_answer(q, top_k)
                if not docs:
                    raise RuntimeError("检索为空（资料库缺对应内容）")
                if not answer or not answer.strip():
                    raise RuntimeError("LLM 返回空回答")
                contexts = [d.page_content for d in docs]
                cache[q] = {
                    "response": answer,
                    "contexts": contexts,
                    "metrics": metrics,
                }
                save_cache(cache, fingerprint)
            if metrics:
                latency_rows.append(metrics)
            samples.append(
                SingleTurnSample(
                    user_input=q,
                    response=answer,
                    reference=ref,
                    retrieved_contexts=contexts,
                )
            )
            print(f"  [{i}/{len(questions)}] 完成" + ("（缓存）" if from_cache else ""))
        except Exception:
            failures += 1
            print(f"  [{i}/{len(questions)}] 失败: {traceback.format_exc(limit=1).strip().splitlines()[-1]}")
    print(f"\n生成/加载完毕，成功 {len(samples)} 条 / 失败 {failures} 条\n")

    stats = {"hit": hit, "failed": failures, "metrics": _aggregate_run_metrics(latency_rows)}
    return samples, cache, stats


def run_evaluation(samples: list[SingleTurnSample], judge_raw: bool = False):
    """RAGAS 判分 4 项指标，返回 (DataFrame, 指标列表)。"""
    # Judge LLM / Embedding：复用项目 .env 配置（ChatOpenAI + 项目 Embeddings，已验证可跑通）
    settings_obj = get_settings()
    raw_llm = ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", settings_obj.LLM_MODEL),
        api_key=os.getenv("RAGAS_LLM_API_KEY", settings_obj.LLM_API_KEY),
        base_url=os.getenv("RAGAS_LLM_BASE_URL", settings_obj.llm_base_url),
        temperature=0,
    )
    if judge_raw:
        judge_llm = raw_llm
        print("判分 LLM：直连 ChatOpenAI（--judge-raw，无 JSON 修复重试）")
    else:
        # 包装成 BaseRagasLLM：ragas 0.4.3 的 PydanticPrompt 对 LangChain LLM 直连分支
        # 直接用 model_validate_json 解析、不重试（见 ragas/prompt/pydantic_prompt.py），
        # DeepSeek 输出 ```json 围栏 / 夹杂解释文字 / 字段缺失时解析失败 → 该行该指标被
        # evaluate(raise_exceptions=False) 吞成 NaN。包装后走 extract_json + fix_output_format 重试。
        # bypass_n=True：DeepSeek 不支持 OpenAI n>1（answer_relevancy 需 n=3，改为发 n 次单请求）；
        # bypass_temperature=True：保持构造时的 temperature=0，判分确定性不受 wrapper 覆盖。
        from ragas.llms import LangchainLLMWrapper  # noqa: E402

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            judge_llm = LangchainLLMWrapper(
                raw_llm, bypass_n=True, bypass_temperature=True
            )
        print("判分 LLM：LangchainLLMWrapper 包装（extract_json + fix_output_format 重试，可显著降低 NaN）")

    from core.embeddings import get_embeddings  # noqa: F401

    # 单例指标注入 LLM / Embeddings
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = get_embeddings()

    print("开始 RAGAS 评估（每指标逐条判分，耗时较长）…")
    import ragas

    result = ragas.evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        raise_exceptions=False,
    )
    return result.to_pandas(), metrics


def _safe_round(v, ndigits: int = 4):
    """NaN → None（JSON 序列化为 null），否则四舍五入。修复 float('nan') 被 json.dumps 输出为非法 NaN。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, ndigits)


def _fmt(v, ndigits: int = 3) -> str:
    """NaN → '-'，否则格式化浮点。修复 Markdown 里 :.3f 输出 'nan' 字符串的坑。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{f:.{ndigits}f}" if not np.isnan(f) else "-"


def diagnose_nan(df) -> dict:
    """统计各指标 NaN 条数，并逐条打印（题号 + 问题 + 回答摘要），返回 {metric: count}。"""
    nan_counts = {}
    for name in METRIC_NAMES:
        mask = df[name].isna()
        cnt = int(mask.sum())
        nan_counts[name] = cnt
        if cnt:
            print(f"  [NaN x{cnt}] {name}")
            for i, row in df[mask].iterrows():
                q = str(row.get("user_input", ""))[:60].replace("\n", " ")
                resp = str(row.get("response", ""))[:40].replace("\n", " ")
                print(f"      #{i + 1} 问题: {q}\n          回答: {resp}…")
    return nan_counts


def apply_nan_policy(df, mode: str):
    """按 --nan-handling 对全局统计口径做处理（逐条明细不受影响，NaN 仍显示为 -）。"""
    if mode == "zero":
        return df.fillna(0.0)
    if mode == "drop":
        n = int(df[METRIC_NAMES].isna().any(axis=1).sum())
        if n:
            print(f"[nan-handling=drop] 剔除含 NaN 的行 {n} 条，剩余 {len(df) - n} 条参与统计")
        return df.dropna(subset=METRIC_NAMES).copy()
    return df


def build_json_report(meta: dict, df) -> dict:
    """序列化评估结果为 JSON 结构（meta / global_metrics / details），NaN 一律输出 null。"""
    stats = df.describe()
    global_metrics = {
        name: {
            stat: _safe_round(stats.loc[stat, name], 4)
            for stat in ["mean", "min", "25%", "50%", "75%", "max"]
        }
        for name in METRIC_NAMES
    }
    details = []
    for i, row in df.iterrows():
        item = {"index": i + 1, "question": str(row.get("user_input", ""))}
        for name in METRIC_NAMES:
            item[name] = _safe_round(row[name])
        details.append(item)
    return {"meta": meta, "global_metrics": global_metrics, "details": details}


def build_markdown_report(meta: dict, df, json_filename: str) -> str:
    """渲染 Markdown 报告（纯字符串拼接，不引入模板库）。"""
    stats = df.describe()
    rows = []
    for name in METRIC_NAMES:
        s = stats[name]
        rows.append(
            f"| {name} | {_fmt(s['mean'])} | {_fmt(s['min'])} | {_fmt(s['25%'])} | "
            f"{_fmt(s['50%'])} | {_fmt(s['75%'])} | {_fmt(s['max'])} |"
        )
    overall = df[METRIC_NAMES].mean().mean()

    lines = [
        "# RAGAS 评估报告",
        "",
        f"**生成时间**：{meta['generated_at']}",
        f"**测试集**：{meta['testset']}（{meta['total_questions']} 题，成功 {meta['successful']} / 失败 {meta['failed']}）",
        "**运行参数**：",
        f"- Top-K: {meta['top_k']}",
        f"- Query Transformation: {'开启' if meta['query_transform'] else '关闭'}",
        f"- HYDE: {'开启' if meta['hyde'] else '关闭'}",
        f"- Retrieval Mode: {meta['retrieval_mode']}",
        f"- Rerank: {'开启' if meta['rerank'] else '关闭'}",
        f"- 缓存: 命中 {meta['cache_hits']} 条" + ("（缓存启用）" if meta["cache_enabled"] else "（缓存关闭）"),
        f"- 耗时: {meta['generation_duration_s']}s（生成） + {meta['evaluation_duration_s']}s（评估）",
        "",
    ]
    run_metrics = meta.get("run_metrics") or {}
    if run_metrics:
        lines += [
            "## 运行指标",
            "",
            "| 指标 | 均值 | P50 | P95 |",
            "|------|------|-----|-----|",
        ]
        for name, values in run_metrics.items():
            lines.append(
                f"| {name} | {_fmt(values.get('mean'))} | "
                f"{_fmt(values.get('p50'))} | {_fmt(values.get('p95'))} |"
            )
        lines.append("")
    lines += [
        "## 全局指标",
        "",
        "| 指标 | 均值 | 最小值 | 25% | 中位数 | 75% | 最大值 |",
        "|------|------|--------|-----|--------|-----|--------|",
        *rows,
        "",
        f"**综合评分**：{_fmt(overall)}（四指标均值）",
        "",
        "**NaN 统计**（判分失败/无效输出的样本数，不参与均值）：",
        "",
        "| 指标 | NaN 条数 |",
        "|------|---------|",
        *[f"| {name} | {meta['nan_counts'].get(name, 0)} |" for name in METRIC_NAMES],
        "",
        "## 逐条明细",
        "",
        "| # | 问题（摘要） | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
        "|---|-------------|-------------|-----------------|-------------------|----------------|",
    ]
    for i, row in df.iterrows():
        question = str(row.get("user_input", ""))
        q_short = (question[:50] + "…") if len(question) > 50 else question
        q_short = q_short.replace("|", "\\|")
        lines.append(
            f"| {i + 1} | {q_short} | {_fmt(row['faithfulness'])} | {_fmt(row['answer_relevancy'])} | "
            f"{_fmt(row['context_precision'])} | {_fmt(row['context_recall'])} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "> 可对比不同运行参数（Top-K / 检索开关）下的报告结论。",
        f"> 原始数据见 `{json_filename}`",
        "",
    ]
    return "\n".join(lines)


def write_report(report_dir: Path, df, questions: list[str], meta: dict) -> tuple[Path, Path]:
    """写 Markdown + JSON 报告（同名时间戳），返回两个文件路径。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = report_dir / f"ragas_report_{timestamp}.md"
    json_path = report_dir / f"ragas_report_{timestamp}.json"

    json_path.write_text(
        json.dumps(build_json_report(meta, df), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(build_markdown_report(meta, df, json_path.name), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    # ---- 阶段 0：参数解析 & 早期校验 ----
    args = parse_args()
    report_dir = _resolve_project_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---- 阶段 1：加载测试集 ----
    testset_path = _resolve_project_path(args.testset or os.getenv("RAGAS_TESTSET", str(DEFAULT_TESTSET)))
    questions, ground_truths = load_testset(testset_path)

    # ---- 阶段 2：覆盖检索开关，并记录本次实际生效值（供报告 meta 使用）----
    no_cache = args.no_cache or os.getenv("RAGAS_NO_CACHE") == "1"
    originals = override_settings(args)
    effective = {
        "top_k": args.top_k or get_settings().TOP_K,
        "query_transform": get_settings().QUERY_TRANSFORM,
        "hyde": get_settings().HYDE,
        "retrieval_mode": get_settings().RETRIEVAL_MODE,
        "rerank": get_settings().RERANK_ENABLED,
    }
    try:
        cache_fingerprint, corpus_manifest = build_run_fingerprint(args.top_k)
        # ---- 阶段 3：批量生成回答（检索链路 + LLM）----
        t_gen_start = time.time()
        samples, _cache, gen_stats = run_generation(
            questions,
            ground_truths,
            args.top_k,
            no_cache,
            cache_fingerprint,
        )
        t_gen_end = time.time()
    finally:
        # ---- 阶段 4：恢复开关（保持进程内状态干净）----
        restore_and_warn(originals)

    # ---- 阶段 5：RAGAS 评估（指标判分）----
    if not samples:
        print("无可评估样本，退出。")
        sys.exit(1)
    t_eval_start = time.time()
    df, _metrics = run_evaluation(samples, judge_raw=args.judge_raw)
    t_eval_end = time.time()

    # ---- 阶段 5.5：NaN 诊断与处理策略 ----
    print("\nNaN 诊断（判分失败/LLM 输出无效的样本）：")
    nan_counts = diagnose_nan(df)
    total_nan = sum(nan_counts.values())
    if total_nan:
        print(
            "提示：NaN 的常见根因与对策见脚本注释（--judge-raw 回退对比 / 换判分模型 / "
            "检查 ground_truth 与回答是否为空）；--nan-handling 可控制统计口径（keep/drop/zero）。"
        )
    else:
        print("  无 NaN，全部样本判分成功。")
    df = apply_nan_policy(df, args.nan_handling)

    # ---- 阶段 6：生成报告 ----
    testset_display = testset_path.resolve().relative_to(BACKEND_DIR.parent) if testset_path.is_relative_to(BACKEND_DIR.parent) else str(testset_path)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "testset": str(testset_display),
        "top_k": effective["top_k"],
        "query_transform": effective["query_transform"],
        "hyde": effective["hyde"],
        "retrieval_mode": effective["retrieval_mode"],
        "rerank": effective["rerank"],
        "judge_raw": args.judge_raw,
        "nan_handling": args.nan_handling,
        "nan_counts": nan_counts,
        "cache_enabled": not no_cache,
        "cache_fingerprint": cache_fingerprint,
        "corpus": corpus_manifest["corpus"],
        "cache_hits": gen_stats["hit"],
        "run_metrics": gen_stats["metrics"],
        "total_questions": len(questions),
        "successful": len(samples),
        "failed": gen_stats["failed"],
        "generation_duration_s": round(t_gen_end - t_gen_start, 1),
        "evaluation_duration_s": round(t_eval_end - t_eval_start, 1),
    }
    md_path, json_path = write_report(report_dir, df, questions, meta)

    # ---- 阶段 7：控制台摘要 ----
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(df.describe().loc[["mean", "min", "25%", "50%", "75%", "max"]].round(3).to_string())
    print(f"\n平均分：{df[METRIC_NAMES].mean().round(3).to_dict()}")
    print(f"\n报告已生成：\n  Markdown: {md_path}\n  JSON:     {json_path}")


if __name__ == "__main__":
    main()

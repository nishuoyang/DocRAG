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

Judge LLM 与 Embedding 均复用 .env 配置（DeepSeek + SiliconFlow bge-m3），不改 .env。
兼容环境变量（CLI 优先）：
    RAGAS_TESTSET / RAGAS_CACHE / RAGAS_NO_CACHE / RAGAS_LLM_MODEL / RAGAS_LLM_BASE_URL / RAGAS_LLM_API_KEY
"""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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


def generate_answer(query: str, top_k: int | None) -> tuple[str, list[Document]]:
    """走与 /chat 完全相同的链路：检索 → 拼上下文 → LLM 生成。"""
    docs = retrieval._retrieve(query, top_k)
    if not docs:
        return "资料库中尚未检索到相关内容。", []
    messages = retrieval._build_messages(query, docs)
    answer = retrieval.llm.get_llm().invoke(messages).content
    return answer, docs


def load_cache() -> dict:
    """读取上次生成的缓存（question → {response, contexts}）。"""
    path = Path(os.getenv("RAGAS_CACHE", str(DEFAULT_CACHE)))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    path = Path(os.getenv("RAGAS_CACHE", str(DEFAULT_CACHE)))
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


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


def run_generation(
    questions: list[str], ground_truths: list[list[str]], top_k: int | None, no_cache: bool
) -> tuple[list[SingleTurnSample], dict, dict]:
    """逐条生成回答（缓存命中复用，否则走检索链路），返回 (samples, cache, 统计)。"""
    cache = {} if no_cache else load_cache()
    hit = 0 if no_cache else sum(q in cache for q in questions)
    print(f"加载 {len(questions)} 条测试题（缓存命中 {hit} 条" + ("，--no-cache 强制重新生成" if no_cache else "") + "）\n")

    samples = []
    failures = 0
    for i, (q, refs) in enumerate(zip(questions, ground_truths), 1):
        from_cache = q in cache
        try:
            if from_cache:
                entry = cache[q]
                answer, contexts = entry["response"], entry["contexts"]
            else:
                answer, docs = generate_answer(q, top_k)
                if not docs:
                    raise RuntimeError("检索为空（资料库缺对应内容）")
                contexts = [d.page_content for d in docs]
                cache[q] = {"response": answer, "contexts": contexts}
                save_cache(cache)
            samples.append(
                SingleTurnSample(
                    user_input=q,
                    response=answer,
                    reference=refs[0] if isinstance(refs, list) else refs,
                    retrieved_contexts=contexts,
                )
            )
            print(f"  [{i}/{len(questions)}] 完成" + ("（缓存）" if from_cache else ""))
        except Exception:
            failures += 1
            print(f"  [{i}/{len(questions)}] 失败: {traceback.format_exc(limit=1).strip().splitlines()[-1]}")
    print(f"\n生成/加载完毕，成功 {len(samples)} 条 / 失败 {failures} 条\n")

    stats = {"hit": hit, "failed": failures}
    return samples, cache, stats


def run_evaluation(samples: list[SingleTurnSample]):
    """RAGAS 判分 4 项指标，返回 (DataFrame, 指标列表)。"""
    # Judge LLM / Embedding：复用项目 .env 配置（ChatOpenAI + 项目 Embeddings，已验证可跑通）
    settings_obj = get_settings()
    judge_llm = ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", settings_obj.LLM_MODEL),
        api_key=os.getenv("RAGAS_LLM_API_KEY", settings_obj.LLM_API_KEY),
        base_url=os.getenv("RAGAS_LLM_BASE_URL", settings_obj.llm_base_url),
        temperature=0,
    )
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


def build_json_report(meta: dict, df) -> dict:
    """序列化评估结果为 JSON 结构（meta / global_metrics / details）。"""
    stats = df.describe()
    global_metrics = {
        name: {
            stat: round(float(stats.loc[stat, name]), 4)
            for stat in ["mean", "min", "25%", "50%", "75%", "max"]
        }
        for name in METRIC_NAMES
    }
    details = []
    for i, row in df.iterrows():
        item = {"index": i + 1, "question": str(row.get("user_input", ""))}
        for name in METRIC_NAMES:
            item[name] = round(float(row[name]), 4)
        details.append(item)
    return {"meta": meta, "global_metrics": global_metrics, "details": details}


def build_markdown_report(meta: dict, df, json_filename: str) -> str:
    """渲染 Markdown 报告（纯字符串拼接，不引入模板库）。"""
    stats = df.describe()
    rows = []
    for name in METRIC_NAMES:
        s = stats[name]
        rows.append(
            f"| {name} | {s['mean']:.3f} | {s['min']:.3f} | {s['25%']:.3f} | "
            f"{s['50%']:.3f} | {s['75%']:.3f} | {s['max']:.3f} |"
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
        f"- Rerank: {'开启' if meta['rerank'] else '关闭'}",
        f"- 缓存: 命中 {meta['cache_hits']} 条" + ("（缓存启用）" if meta["cache_enabled"] else "（缓存关闭）"),
        f"- 耗时: {meta['generation_duration_s']}s（生成） + {meta['evaluation_duration_s']}s（评估）",
        "",
        "## 全局指标",
        "",
        "| 指标 | 均值 | 最小值 | 25% | 中位数 | 75% | 最大值 |",
        "|------|------|--------|-----|--------|-----|--------|",
        *rows,
        "",
        f"**综合评分**：{overall:.3f}（四指标均值）",
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
            f"| {i + 1} | {q_short} | {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | "
            f"{row['context_precision']:.3f} | {row['context_recall']:.3f} |"
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
        "rerank": get_settings().RERANK_ENABLED,
    }
    try:
        # ---- 阶段 3：批量生成回答（检索链路 + LLM）----
        t_gen_start = time.time()
        samples, _cache, gen_stats = run_generation(questions, ground_truths, args.top_k, no_cache)
        t_gen_end = time.time()
    finally:
        # ---- 阶段 4：恢复开关（保持进程内状态干净）----
        restore_and_warn(originals)

    # ---- 阶段 5：RAGAS 评估（指标判分）----
    if not samples:
        print("无可评估样本，退出。")
        sys.exit(1)
    t_eval_start = time.time()
    df, _metrics = run_evaluation(samples)
    t_eval_end = time.time()

    # ---- 阶段 6：生成报告 ----
    testset_display = testset_path.resolve().relative_to(BACKEND_DIR.parent) if testset_path.is_relative_to(BACKEND_DIR.parent) else str(testset_path)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "testset": str(testset_display),
        "top_k": effective["top_k"],
        "query_transform": effective["query_transform"],
        "hyde": effective["hyde"],
        "rerank": effective["rerank"],
        "cache_enabled": not no_cache,
        "cache_hits": gen_stats["hit"],
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

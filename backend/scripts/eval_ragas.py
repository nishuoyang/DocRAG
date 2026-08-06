"""RAGAS 评估脚本：用真实检索链路（_retrieve）跑 docs/test.py 的 50 题，输出 4 项指标。

运行（在 backend/ 下，用独立 venv 避免污染主环境）：
    ./.venv-ragas/Scripts/python.exe scripts/eval_ragas.py

流程：先逐条走真实检索链路生成回答（结果缓存到 .ragas_cache.json，重跑直接复用），
再对每条执行 4 项指标判分（faithfulness / answer_relevancy / context_precision / context_recall）。

Judge LLM 与 Embedding 均复用 .env 配置（DeepSeek + SiliconFlow bge-m3），不改 .env。
可选环境变量（默认复用 .env 的 LLM）：
    RAGAS_LLM_MODEL   判分模型（默认与 LLM_MODEL 相同）
    RAGAS_LLM_BASE_URL
    RAGAS_LLM_API_KEY
    RAGAS_TESTSET     测试集文件路径（默认 docs/test.py；相对路径基于项目根目录）
    RAGAS_CACHE       缓存文件路径（默认 backend/.ragas_cache.json）
    RAGAS_NO_CACHE=1  忽略已有缓存，强制重新走检索链路生成（换文档库/检索参数后必须）
"""
import os
import sys
import time
import traceback
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

N_TOP = 5  # 与前端默认 Top-K 一致


def load_testset() -> tuple[list[str], list[list[str]]]:
    """从测试集文件加载 questions / ground_truths 列表（RAGAS_TESTSET 覆盖，默认 docs/test.py）。"""
    test_py = Path(os.getenv("RAGAS_TESTSET", str(BACKEND_DIR.parent / "docs" / "test.py")))
    if not test_py.is_absolute():
        test_py = BACKEND_DIR.parent / test_py
    if not test_py.exists():
        raise FileNotFoundError(f"测试集文件不存在: {test_py}")
    ns: dict = {}
    exec(compile(test_py.read_text(encoding="utf-8"), str(test_py), "exec"), ns)
    questions = ns["questions"]
    ground_truths = ns["ground_truths"]
    assert len(questions) == len(ground_truths), "questions 与 ground_truths 数量不一致"
    return questions, ground_truths


def generate_answer(query: str) -> tuple[str, list[Document]]:
    """走与 /chat 完全相同的链路：检索 → 拼上下文 → LLM 生成。"""
    docs = retrieval._retrieve(query, N_TOP)
    if not docs:
        return "资料库中尚未检索到相关内容。", []
    messages = retrieval._build_messages(query, docs)
    answer = retrieval.llm.get_llm().invoke(messages).content
    return answer, docs


def load_cache() -> dict:
    """读取上次生成的缓存（question → {response, contexts}）。"""
    path = Path(os.getenv("RAGAS_CACHE", str(BACKEND_DIR / ".ragas_cache.json")))
    if path.exists():
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    import json
    path = Path(os.getenv("RAGAS_CACHE", str(BACKEND_DIR / ".ragas_cache.json")))
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    questions, ground_truths = load_testset()
    no_cache = os.getenv("RAGAS_NO_CACHE") == "1"
    cache = {} if no_cache else load_cache()
    hit = 0 if no_cache else sum(q in cache for q in questions)
    print(f"加载 {len(questions)} 条测试题（缓存命中 {hit} 条" + ("，RAGAS_NO_CACHE 强制重新生成" if no_cache else "") + "）\n")

    samples = []
    failures = 0
    t0 = time.time()
    for i, (q, refs) in enumerate(zip(questions, ground_truths), 1):
        try:
            if q in cache:
                entry = cache[q]
                answer, contexts = entry["response"], entry["contexts"]
            else:
                answer, docs = generate_answer(q)
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
            print(f"  [{i}/{len(questions)}] 完成" + ("（缓存）" if i and q in cache else ""))
        except Exception:
            failures += 1
            print(f"  [{i}/{len(questions)}] 失败: {traceback.format_exc(limit=1).strip().splitlines()[-1]}")
    print(f"\n生成/加载完毕，成功 {len(samples)} 条 / 失败 {failures} 条，耗时 {time.time() - t0:.0f}s\n")

    if not samples:
        sys.exit(1)

    # ---- Judge LLM / Embedding：复用项目 .env 配置（ChatOpenAI + 项目 Embeddings，已验证可跑通） ----
    settings = get_settings()
    judge_llm = ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", settings.LLM_MODEL),
        api_key=os.getenv("RAGAS_LLM_API_KEY", settings.LLM_API_KEY),
        base_url=os.getenv("RAGAS_LLM_BASE_URL", settings.llm_base_url),
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
    df = result.to_pandas()
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(df.describe().loc[["mean", "min", "25%", "50%", "75%", "max"]].round(3).to_string())
    print("\n逐条明细：")
    for i, row in df.iterrows():
        print(
            f"  #{i + 1:>2} 忠实度={row['faithfulness']:.2f} 相关性={row['answer_relevancy']:.2f} "
            f"上下文精度={row['context_precision']:.2f} 上下文召回={row['context_recall']:.2f}  {questions[i][:30]}"
        )
    print(f"\n平均分：{df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()

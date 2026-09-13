"""RAGAS 测试集自动生成：从 Milvus 文档库生成测试集文件（questions / ground_truths）。

运行（在 backend/ 下，用独立 venv 避免污染主环境）：
    ./.venv-ragas/Scripts/python.exe scripts/generate_testset.py [--size 20] [--strategy single-hop] [--out docs/test_generated.py]

生成后的测试集直接用评估脚本跑：
    ./.venv-ragas/Scripts/python.exe scripts/eval_ragas.py --testset docs/test_generated.py

原理：读取 Milvus 中已上传的全部文档块（与检索链路同一来源）→ 构建知识图谱 →
按策略抽取实体/建边 → ragas TestsetGenerator 合成 questions + reference（参考答案）
→ 落盘为 Python 文件（questions / ground_truths 列表，与 docs/test.py 同格式）。

策略（--strategy）：
    single-hop   仅 NER 抽取实体，生成单跳具体问题（快：约 2s/块，620 块约 20 分钟）
    multi-hop    NER + 实体重叠建边，额外生成多跳问题（较慢，约 30-40 分钟）
    full         用 ragas 默认 transforms（摘要/主题/NER/过滤，最全但最慢，约 1 小时+）

注意：
- 生成过程会大量调用 LLM/Embedding API（按 size 与文档块数计费），首次运行较慢
- 知识图谱与 personas 缓存到 backend/.ragas_kg_cache.json / .ragas_personas.json（默认），
  重跑同一策略直接复用；RAGAS_KG_CACHE / RAGAS_PERSONAS_CACHE 可改路径
- 文档库为空（未上传文档）时直接报错退出
- 生成结果基于当前文档库内容；换库/重传文档后需重新生成（或删缓存）
- 提速选项：RAGAS_FAST_LLM_MODEL 给 Summary/NER 抽取指定快模型（可独立配
  RAGAS_FAST_LLM_BASE_URL / RAGAS_FAST_LLM_API_KEY，未配时复用 .env 主配置）；
  --max-chunks 限制抽样块数（大库建议 100-200）；--workers 调并发（默认 32）；
  --style clean 排除 POOR_GRAMMAR/MISSPELLED 错别字风格（仅 single-hop/multi-hop）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 把 backend/ 加入 sys.path，复用项目代码与 .env 配置
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# ragas 内部多次 asyncio.run()（Executor），每次创建新循环并关闭旧的；
# openai 异步 client 会绑定首次请求时的循环，后续复用报 Event loop is closed。
# 用 nest_asyncio 包一层外层循环，让内部全部复用同一活循环（ragas 为 Jupyter 设计的模式）。
import nest_asyncio  # noqa: E402
nest_asyncio.apply()  # noqa: E402

# 加载 backend/.env（项目未安装 python-dotenv 自动加载机制，脚本需手动注入环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from langchain_core.documents import Document  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402

from config import get_settings  # noqa: E402
from db import milvus  # noqa: E402

DEFAULT_OUT = BACKEND_DIR.parent / "docs" / "test_generated.py"
KG_CACHE = BACKEND_DIR / ".ragas_kg_cache.json"
PERSONAS_CACHE = BACKEND_DIR / ".ragas_personas.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGAS 测试集自动生成：从 Milvus 文档库合成 questions / ground_truths 测试集文件"
    )
    parser.add_argument("--size", type=int, default=20, help="生成题目数（默认 20）")
    parser.add_argument(
        "--strategy",
        choices=["single-hop", "multi-hop", "full"],
        default="single-hop",
        help="生成策略（默认 single-hop）：single-hop 仅 NER 最快；multi-hop 加建边支持多跳问题；full 用 ragas 默认 transforms 最全最慢",
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="测试集语言（默认 zh）：zh 用中文 instruction 生成中文问题/答案（匹配中文文档库），en 用 ragas 默认英文",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出测试集文件路径（默认 docs/test_generated.py）")
    parser.add_argument("--no-cache", action="store_true", help="忽略知识图谱/personas 缓存，强制重新抽取")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="最多用多少块（0=全部；大库建议 100-200，抽取耗时与块数成正比）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="并发 LLM 调用数（默认 32；Obsidian 的 as_completed 并发上限。调高可提速但可能触发 API 限流）",
    )
    parser.add_argument(
        "--style",
        choices=["all", "clean"],
        default="all",
        help="问题风格（默认 all 用 ragas 全部 4 种；clean 排除 POOR_GRAMMAR/MISSPELLED 错别字风格，仅 PERFECT_GRAMMAR + WEB_SEARCH_LIKE）",
    )
    return parser.parse_args()


def load_documents(max_chunks: int) -> list[Document]:
    """从 Milvus 读文档块（可选抽样），供生成器构建知识图谱。"""
    docs = milvus.get_all_documents()
    if not docs:
        sys.exit("Milvus 文档库为空，请先通过 /documents/upload 上传文档再生成测试集。")
    non_empty = [d for d in docs if d.page_content.strip()]
    if len(non_empty) != len(docs):
        print(f"  跳过 {len(docs) - len(non_empty)} 个空块（ragas 抽取器对空块崩溃）")
    if max_chunks and len(non_empty) > max_chunks:
        import random

        non_empty = random.sample(non_empty, max_chunks)
        print(f"  --max-chunks 抽样: 随机取 {max_chunks} 块（测试集仅覆盖抽样块）")
    files = sorted({d.metadata.get("filename", "?") for d in non_empty})
    print(f"从 Milvus 加载 {len(non_empty)} 个文档块，来源文件：{files}")
    return non_empty


def build_transforms(strategy: str, llm_w, fast_llm_w, emb_w) -> list:
    """按策略构建知识图谱 transforms（返回列表，传给 apply_transforms）。

    persona 生成硬性依赖节点 summary_embedding（default_filter 要求），
    单跳合成器依赖 entities（NER），因此轻量策略至少需要
    SummaryExtractor + EmbeddingExtractor(summary) + NERExtractor。

    Summary/NER 是机械抽取任务，用 fast_llm（RAGAS_FAST_LLM_MODEL）跑，提速且省钱；
    合成阶段仍用主 LLM 保证问题质量。
    """
    from ragas.testset.transforms.default import (
        EmbeddingExtractor,
        NERExtractor,
        SummaryExtractor,
    )

    summary_extractor = SummaryExtractor(llm=fast_llm_w)
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=emb_w,
        property_name="summary_embedding",
        embed_property_name="summary",
    )
    ner_extractor = NERExtractor(llm=fast_llm_w)

    if strategy == "single-hop":
        return [summary_extractor, summary_emb_extractor, ner_extractor]
    if strategy == "multi-hop":
        from ragas.testset.transforms.default import OverlapScoreBuilder

        return [summary_extractor, summary_emb_extractor, ner_extractor, OverlapScoreBuilder(threshold=0.01)]
    # full：ragas 默认预切分 transforms（摘要/主题/NER/过滤 + 建边，最全最慢）
    from ragas.testset.transforms.default import default_transforms_for_prechunked

    return default_transforms_for_prechunked(llm=fast_llm_w, embedding_model=emb_w)


def apply_language(lang: str) -> None:
    """按 --lang 覆盖合成器 prompt 的 instruction（类属性共享，全局生效）。

    ragas 0.4.3 默认 instruction 为英文，生成的问题/答案是英文；中文文档库
    下英文 reference 与检索到的中文上下文匹配度低，会拉低 context_recall 判分。
    这里把单跳/多跳合成器的生成 prompt 改为中文指令。
    """
    if lang != "zh":
        return
    zh_instruction = (
        "根据指定的条件（角色、主题、风格、长度）和提供的上下文，生成一个中文的问题及答案。"
        "问题必须结合上下文中的主题，答案只能使用提供的上下文内容，不能添加未包含的信息，"
        "回答语言必须与上下文语言一致（中文）。"
    )
    from ragas.testset.synthesizers.single_hop.prompts import QueryAnswerGenerationPrompt as SingleHopPrompt

    SingleHopPrompt.instruction = zh_instruction
    try:
        from ragas.testset.synthesizers.multi_hop.prompts import QueryAnswerGenerationPrompt as MultiHopPrompt

        MultiHopPrompt.instruction = zh_instruction
    except ImportError:
        pass  # 0.4.3 多跳 prompt 路径可能不同，跳过


def _build_llm(model_override_env: str, temperature: float):
    """创建 ChatOpenAI，支持按环境变量覆盖模型/base_url/api_key。

    - 主 LLM：RAGAS_LLM_MODEL / RAGAS_LLM_BASE_URL / RAGAS_LLM_API_KEY（未设置回落 .env）
    - fast LLM（transforms 用）：RAGAS_FAST_LLM_MODEL / RAGAS_FAST_LLM_BASE_URL /
      RAGAS_FAST_LLM_API_KEY（未设置回落 .env 主配置；模型未设置时回落 LLM_MODEL）
    """
    settings = get_settings()
    prefix = "RAGAS_FAST_" if model_override_env == "RAGAS_FAST_LLM_MODEL" else "RAGAS_"
    model = os.getenv(model_override_env, settings.LLM_MODEL)
    api_key = os.getenv(prefix + "LLM_API_KEY", settings.LLM_API_KEY)
    base_url = os.getenv(prefix + "LLM_BASE_URL", settings.llm_base_url)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def load_personas_cache() -> list | None:
    """读取 personas 缓存（重跑省 persona 生成；persona 只依赖图谱 summary 聚类，同一库稳定）。"""
    path = Path(os.getenv("RAGAS_PERSONAS_CACHE", str(PERSONAS_CACHE)))
    if not path.exists():
        return None
    try:
        from ragas.testset.synthesizers.base import Persona

        return [Persona(**p) for p in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as exc:
        print(f"personas 缓存解析失败（{exc}），重新生成")
        return None


def save_personas_cache(personas: list) -> None:
    path = Path(os.getenv("RAGAS_PERSONAS_CACHE", str(PERSONAS_CACHE)))
    path.write_text(
        json.dumps([p.model_dump() for p in personas], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def _build_clean_synthesizer(cls, llm):
    """构造只含 PERFECT_GRAMMAR / WEB_SEARCH_LIKE 风格的合成器子类。

    ragas 默认场景风格含全部 QueryStyle（含 POOR_GRAMMAR / MISSPELLED 错别字风格），
    这类问题与检索上下文匹配差、拉低 answer_relevancy 且观感差。子类覆盖
    prepare_combinations 只保留正常风格。
    """
    from ragas.testset.synthesizers.base import QueryLength, QueryStyle

    class CleanStyle(cls):
        def prepare_combinations(self, node, terms, personas, persona_concepts):
            valid_personas = []
            personas_by_name = {persona.name: persona for persona in personas}
            for persona_name, concepts in persona_concepts.items():
                normalized_name = persona_name.split(" (", 1)[0].strip()
                persona = personas_by_name.get(persona_name)
                if persona is None:
                    persona = next(
                        (
                            candidate
                            for candidate in personas
                            if candidate.name.split(" (", 1)[0].strip() == normalized_name
                        ),
                        None,
                    )
                lowered_concepts = [str(concept).lower() for concept in concepts]
                if persona is not None and any(
                    str(term).lower() in lowered_concepts for term in terms
                ):
                    valid_personas.append(persona)

            # The matching prompt can hallucinate persona names. Fall back to
            # the generated personas rather than failing the whole testset.
            if not valid_personas:
                valid_personas = list(personas)

            return [
                {
                    "terms": terms,
                    "node": node,
                    "personas": valid_personas,
                    "styles": [
                        QueryStyle.PERFECT_GRAMMAR,
                        QueryStyle.WEB_SEARCH_LIKE,
                    ],
                    "lengths": list(QueryLength),
                }
            ]

    return CleanStyle(llm=llm)


def build_query_distribution(strategy: str, llm, style: str, knowledge_graph=None):
    """按策略构造 (synthesizer, prob) 列表；style=clean 时子类化合成器排除错别字风格。

    必须传 knowledge_graph 做过滤：default_query_distribution 内部按图谱关系
    过滤多跳合成器（无关系边则跳过），传自定义 qd 会绕过该过滤导致报错。
    """
    if strategy == "full":
        raise ValueError("--style 仅支持 single-hop / multi-hop 策略；full 用 ragas 默认分布")
    from ragas.testset.synthesizers import default_query_distribution

    qd = default_query_distribution(llm, knowledge_graph)
    if style == "clean":
        qd = [
            (_build_clean_synthesizer(type(syn), syn.llm), prob)
            for syn, prob in qd
        ]
    return qd


def generate_testset(
    docs: list[Document], size: int, strategy: str, no_cache: bool, workers: int = 32, style: str = "all"
) -> list:
    """用 TestsetGenerator 合成测试集样本（SingleTurnSample 列表）。"""
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=workers)
    settings = get_settings()

    # 生成 LLM 复用 .env 的 LLM 配置（RAGAS_LLM_MODEL 等可覆盖，与评估脚本一致）
    gen_llm = _build_llm("RAGAS_LLM_MODEL", temperature=0.1)
    # transforms（Summary/NER）用快模型，RAGAS_FAST_LLM_MODEL 未设置时回落主模型
    fast_llm = _build_llm("RAGAS_FAST_LLM_MODEL", temperature=0.1)
    gen_emb = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL,
        check_embedding_ctx_length=False,
    )
    llm_w = LangchainLLMWrapper(gen_llm)
    fast_llm_w = LangchainLLMWrapper(fast_llm)
    emb_w = LangchainEmbeddingsWrapper(gen_emb)
    generator = TestsetGenerator(llm=llm_w, embedding_model=emb_w)

    from ragas.testset.graph import KnowledgeGraph, Node, NodeType

    # ---- 策略 1：缓存命中 → 加载图谱直接 generate()（generate 用 self.knowledge_graph）----
    kg_cache_path = Path(os.getenv("RAGAS_KG_CACHE", str(KG_CACHE)))
    if not no_cache and kg_cache_path.exists() and strategy != "full":
        cached = KnowledgeGraph.load(kg_cache_path)
        if len(cached.nodes) == len(docs):
            print(f"复用知识图谱缓存（{kg_cache_path.name}，{len(cached.nodes)} 节点）")
            generator.knowledge_graph = cached
            personas = None if no_cache else load_personas_cache()
            if personas:
                generator.persona_list = personas
                print(f"复用 personas 缓存（{len(personas)} 个，跳过 persona 生成）")
            qd = build_query_distribution(strategy, llm_w, style, cached) if style != "all" else None
            print(f"开始生成测试集（{size} 题，策略 {strategy}，LLM: {gen_llm.model_name}，并发 {workers}）…")
            testset = generator.generate(testset_size=size, run_config=run_config, query_distribution=qd)
            if personas is None:
                save_personas_cache(generator.persona_list)
            samples = list(testset.to_evaluation_dataset().samples)
            print(
                f"测试集生成完毕，共 {len(samples)} 题，"
                f"消耗 token 约 {_total_tokens_display(testset)}"
            )
            return samples
        print(f"缓存节点数 {len(cached.nodes)} 与文档块数 {len(docs)} 不符，重新抽取")

    # ---- 策略 2：缓存未命中 → 先构建图谱，再 generate()（clean 需要图谱过滤多跳合成器）----
    transforms = build_transforms(strategy, llm_w, fast_llm_w, emb_w)
    print(f"开始构建知识图谱（策略 {strategy}，{len(docs)} 块，transforms 用 {fast_llm.model_name}）…")
    t0 = time.time()
    from ragas.testset.transforms import apply_transforms

    kg = KnowledgeGraph(nodes=[Node(type=NodeType.CHUNK, properties={"page_content": d.page_content}) for d in docs])
    apply_transforms(kg, transforms, run_config=run_config)
    generator.knowledge_graph = kg
    print(f"知识图谱构建完成（{time.time() - t0:.0f}s，"
          f"{len(kg.nodes)} 节点 / {len(kg.relationships)} 关系）")
    if strategy != "full":
        kg.save(kg_cache_path)
        print(f"知识图谱已缓存: {kg_cache_path}")
    # 真正生成测试集（clean 时按图谱过滤多跳合成器）
    qd = build_query_distribution(strategy, llm_w, style, kg) if style != "all" else None
    print(f"开始生成测试集（{size} 题，策略 {strategy}，LLM: {gen_llm.model_name}，并发 {workers}）…")
    testset = generator.generate(testset_size=size, run_config=run_config, query_distribution=qd)
    save_personas_cache(generator.persona_list)
    samples = list(testset.to_evaluation_dataset().samples)
    print(
        f"测试集生成完毕，共 {len(samples)} 题，"
        f"消耗 token 约 {_total_tokens_display(testset)}"
    )
    return samples


def write_testset(samples: list, out_path: Path, strategy: str) -> None:
    """把生成的样本落盘为 Python 测试集文件（与 docs/test.py 同格式）。"""
    questions = []
    ground_truths = []
    for s in samples:
        questions.append(s.user_input)
        ground_truths.append([s.reference] if s.reference else [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 由 scripts/generate_testset.py 自动生成（ragas TestsetGenerator），",
        f"# 策略: {strategy}，基于当前 Milvus 文档库内容，可手动修改后配合 eval_ragas.py 使用。",
        "",
        "questions = [",
    ]
    for q in questions:
        lines.append(f"    {json.dumps(q, ensure_ascii=False)},")
    lines += ["]", "", "ground_truths = ["]
    for gt in ground_truths:
        lines.append(f"    {json.dumps(gt, ensure_ascii=False)},")
    lines += ["]", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"测试集已写入: {out_path}")


def _total_tokens_display(testset) -> str:
    try:
        return str(testset.total_tokens())
    except ValueError:
        return "未启用统计"


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = BACKEND_DIR.parent / out_path

    apply_language(args.lang)
    t0 = time.time()
    docs = load_documents(args.max_chunks)
    samples = generate_testset(docs, args.size, args.strategy, args.no_cache, args.workers, args.style)
    if not samples:
        sys.exit("未生成任何题目（文档块可能过短/空），请检查文档库内容。")
    write_testset(samples, out_path, args.strategy)
    print(f"耗时 {time.time() - t0:.0f}s")
    print(f"评估: ./.venv-ragas/Scripts/python.exe scripts/eval_ragas.py --testset {out_path}")


if __name__ == "__main__":
    main()

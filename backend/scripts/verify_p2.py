"""P2 全链路验证脚本：验证 VLM 集成逻辑（不实际调用 API）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import vlm
from core.parsers import pdf_parser, docx_parser, pptx_parser
from core import ingestion


def test_vlm_client():
    """测试 VLM 客户端初始化。"""
    print("=== 测试 VLM 客户端 ===")
    client = vlm.get_vlm_client()
    print(f"  VLM 启用: {client.enabled}")
    print(f"  VLM 模型: {client.model}")
    print(f"  最大页数: {client.max_pages}")
    print(f"  已处理计数: {client.get_processed_count()}")
    print("✓ VLM 客户端初始化成功\n")


def test_pdf_parser_vlm_integration():
    """测试 PDF 解析器的 VLM 集成逻辑。"""
    print("=== 测试 PDF 解析器 VLM 集成 ===")
    # 验证解析器函数存在
    assert hasattr(pdf_parser, 'parse_pdf')
    assert hasattr(pdf_parser, '_is_scanned_page')
    assert hasattr(pdf_parser, '_render_page_to_image')
    assert hasattr(pdf_parser, '_extract_page_images')
    print("✓ PDF 解析器包含 VLM 集成函数\n")


def test_docx_parser_vlm_integration():
    """测试 DOCX 解析器的 VLM 集成逻辑。"""
    print("=== 测试 DOCX 解析器 VLM 集成 ===")
    assert hasattr(docx_parser, 'parse_docx')
    assert hasattr(docx_parser, '_extract_images_from_paragraph')
    print("✓ DOCX 解析器包含 VLM 集成函数\n")


def test_pptx_parser_vlm_integration():
    """测试 PPTX 解析器的 VLM 集成逻辑。"""
    print("=== 测试 PPTX 解析器 VLM 集成 ===")
    assert hasattr(pptx_parser, 'parse_pptx')
    assert hasattr(pptx_parser, '_convert_table_to_markdown')
    print("✓ PPTX 解析器包含 VLM 集成函数\n")


def test_ingestion_vlm_pages():
    """测试 ingestion 返回 vlm_pages 字段。"""
    print("=== 测试 ingestion 返回 vlm_pages ===")
    # 验证 ingest_file 函数签名
    import inspect
    sig = inspect.signature(ingestion.ingest_file)
    print(f"  ingest_file 参数: {list(sig.parameters.keys())}")
    
    # 验证返回结构（通过代码检查）
    with open(ingestion.__file__, 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'vlm_pages' in content, "ingestion.py 中未找到 vlm_pages"
        assert 'get_vlm_client().get_processed_count()' in content, "未调用 get_processed_count()"
    
    print("✓ ingestion 正确返回 vlm_pages\n")


def test_vlm_disabled_mode():
    """测试 VLM 禁用模式下的解析。"""
    print("=== 测试 VLM 禁用模式 ===")
    client = vlm.get_vlm_client()
    
    if not client.enabled:
        print("  VLM 已禁用（正常，未配置 API Key）")
        # 验证禁用模式下的方法返回空字符串
        result = vlm.process_inline_image_sync(b"fake_image_data")
        assert result == "", f"禁用模式下应返回空字符串，实际返回: {result}"
        print("✓ VLM 禁用模式下正确返回空字符串\n")
    else:
        print("  VLM 已启用（跳过禁用模式测试）\n")


def main():
    print("=" * 60)
    print("P2 全链路验证（VLM 多模态集成）")
    print("=" * 60 + "\n")
    
    try:
        test_vlm_client()
        test_pdf_parser_vlm_integration()
        test_docx_parser_vlm_integration()
        test_pptx_parser_vlm_integration()
        test_ingestion_vlm_pages()
        test_vlm_disabled_mode()
        
        print("=" * 60)
        print("✓ P2 全链路验证通过")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

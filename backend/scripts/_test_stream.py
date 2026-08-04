"""实测 /chat/stream 是否流式返回：记录每个事件到达的时间戳。"""
import json
import time
import urllib.request

body = json.dumps({"query": "test_doc.docx 文档里写了什么内容？", "top_k": 3}).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8000/chat/stream",
    data=body,
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
n_events = 0
first_delta_t = None
with urllib.request.urlopen(req, timeout=120) as resp:
    print(f"status={resp.status}, content-type={resp.headers.get('Content-Type')}")
    while True:
        line = resp.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        if text.startswith("data: "):
            n_events += 1
            t = time.time() - t0
            payload = text[6:]
            try:
                data = json.loads(payload)
                if "delta" in data:
                    if first_delta_t is None:
                        first_delta_t = t
                        print(f"  T+{t:.2f}s  FIRST delta: {data['delta'][:20]!r}")
                    if n_events in (1, 2, 5, 10, 20, 50):
                        print(f"  T+{t:.2f}s  event#{n_events} delta: {data['delta'][:30]!r}")
                elif "sources" in data:
                    print(f"  T+{t:.2f}s  SOURCES ({len(data['sources'])} 条)")
                elif "answer" in data:
                    print(f"  T+{t:.2f}s  ANSWER (空库兜底)")
                else:
                    print(f"  T+{t:.2f}s  OTHER: {payload[:50]}")
            except json.JSONDecodeError:
                if payload == "[DONE]":
                    print(f"  T+{t:.2f}s  DONE")
                else:
                    print(f"  T+{t:.2f}s  RAW: {payload[:50]}")

print(f"总事件数: {n_events}, 总耗时: {time.time()-t0:.2f}s, 首 token: {first_delta_t:.2f}s" if first_delta_t else f"总事件数: {n_events}, 总耗时: {time.time()-t0:.2f}s")

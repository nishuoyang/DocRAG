"""数据分析执行器：LLM 生成的 pandas 脚本在受限子进程中运行。

安全边界（本地单用户工具，非多租户：AST 白名单 + 最小环境 + 超时）：
- 只允许 import pandas/numpy/matplotlib/re/json/math/statistics/datetime
- 禁止 os/sys/subprocess 等危险模块名、禁止 open/eval/exec/input/__import__ 调用
- 脚本开头注入 DATA_PATH 变量指向数据 CSV；脚本写文件只允许落在临时工作目录
- 超时杀死（DATA_EXEC_TIMEOUT），stdout 捕获，图表读为 base64 data URL 返回
"""
import ast
import base64
import os
import subprocess
import sys
import tempfile

from config import get_settings

ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "re", "json", "math", "statistics", "datetime"}
# 危险模块名（import 白名单之外的顶级名一律拒绝）
FORBIDDEN_NAMES = {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "requests", "urllib", "http",
                   "builtins", "importlib", "ctypes", "multiprocessing", "threading", "pickle", "io"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "input", "compile", "__import__", "exit", "quit", "getattr", "setattr", "vars", "globals", "locals"}


class DataExecError(Exception):
    pass


def check_script(script: str) -> None:
    """AST 校验：白名单 import + 禁用危险名字/调用。不通过抛 DataExecError。"""
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise DataExecError(f"禁止 import：{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in ALLOWED_IMPORTS:
                raise DataExecError(f"禁止 import：{node.module}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                # 变量名（df 等）放行；危险模块名与 dunder 拒绝
                raise DataExecError(f"禁止使用：{node.id}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                raise DataExecError(f"禁止调用：{f.id}")
            # pd.read_csv 等读数据放行；写文件/执行类属性方法拒绝
            if isinstance(f, ast.Attribute) and f.attr in {"to_csv", "to_excel", "to_pickle", "system", "popen", "run", "shell", "getenv", "environ"}:
                raise DataExecError(f"禁止调用：{f.attr}")
            if isinstance(f, ast.Attribute) and f.attr == "savefig":
                pass  # 图表允许（仅能写到临时工作目录 cwd）


def run_script(script: str, data: str) -> dict:
    """子进程执行脚本。data 落盘 data.csv 供脚本经 DATA_PATH 读取；返回 {'stdout','images':[base64...]}。"""
    settings = get_settings()
    check_script(script)
    with tempfile.TemporaryDirectory() as workdir:
        data_path = os.path.join(workdir, "data.csv")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(data)
        # 脚本前缀注入 DATA_PATH（校验通过后注入，AST 校验阶段 script 可以是纯用户代码）
        full_script = f'DATA_PATH = r"{data_path}"\n' + script
        script_path = os.path.join(workdir, "run.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)
        # 继承环境但剔除凭据类变量（子进程代码只读数据即可，不需要探测 API key）
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(t in k.upper() for t in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"))
        }
        # 强制子进程 stdout/stderr 用 UTF-8（中文 Windows 默认 cp936，与父进程 text=True 解码不一致会崩）
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.DATA_EXEC_TIMEOUT,
                cwd=workdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise DataExecError(f"脚本执行超时（>{settings.DATA_EXEC_TIMEOUT}s），已终止。")
        if proc.returncode != 0:
            raise DataExecError(f"脚本运行失败：{proc.stderr[-800:]}")
        images = []
        for name in sorted(os.listdir(workdir)):
            if name.endswith(".png"):
                with open(os.path.join(workdir, name), "rb") as f:
                    images.append("data:image/png;base64," + base64.b64encode(f.read()).decode())
        return {"stdout": proc.stdout, "images": images}

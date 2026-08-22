#!/usr/bin/env bash
# 一键启动 RAG 全栈服务：Milvus (docker compose) + 后端 FastAPI + 前端 Vite
# 用法: bash start.sh [--backend-only]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# ANSI 颜色
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
err()   { printf "${RED}[ERR]${NC} %s\n" "$1"; }

check() { # 检查命令是否存在
  if ! command -v "$1" >/dev/null 2>&1; then
    err "未找到命令 $1，请先安装"
    exit 1
  fi
}

# 等待健康检查通过
wait_health() { # $1=url $2=描述 $3=超时秒
  local url="$1" desc="$2" timeout="${3:-60}" i=0
  while ! curl -sf "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge "$timeout" ]; then
      err "$desc 在 ${timeout}s 内未就绪，请查看日志"
      return 1
    fi
    sleep 1
  done
  info "$desc 就绪"
}

BACKEND_ONLY=false
[ "${1:-}" = "--backend-only" ] && BACKEND_ONLY=true

# 1. Milvus 基础设施
if [ "$BACKEND_ONLY" = false ]; then
  check docker
  if ! docker compose version >/dev/null 2>&1; then
    err "未找到 docker compose（需 compose v2），请检查 Docker Desktop 是否安装"
    exit 1
  fi
  info "启动 Milvus 基础设施（etcd + MinIO + standalone）..."
  (cd "$ROOT" && docker compose up -d) || {
    err "docker compose up 失败，请确认 Docker Desktop 已启动"
    exit 1
  }
  wait_health "http://127.0.0.1:9091/healthz" "Milvus" 90 || exit 1
fi

# 2. 后端 FastAPI
check "$BACKEND/.venv/Scripts/python.exe"
info "启动后端 FastAPI (http://127.0.0.1:8001)..."
(cd "$BACKEND" && ./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001) &
BACKEND_PID=$!

# 3. 前端 Vite
if [ "$BACKEND_ONLY" = false ]; then
  check npm
  [ -d "$FRONTEND/node_modules" ] || {
    warn "frontend/node_modules 不存在，执行 npm install..."
    (cd "$FRONTEND" && npm install)
  }
  info "启动前端 Vite (http://localhost:5173)..."
  (cd "$FRONTEND" && npm run dev) &
  FRONTEND_PID=$!
fi

trap 'kill $BACKEND_PID ${FRONTEND_PID:-0} 2>/dev/null || true' INT TERM

# 等后端就绪后提示访问地址
wait_health "http://127.0.0.1:8001/health" "后端 FastAPI" 60 || true

printf "\n${GREEN}=== 服务已启动 ===${NC}\n"
printf "  后端 API:   http://127.0.0.1:8001 (Swagger: /docs)\n"
if [ "$BACKEND_ONLY" = false ]; then
  printf "  前端页面:   http://localhost:5173\n"
fi
printf "\n按 Ctrl+C 停止全部服务。\n"

wait

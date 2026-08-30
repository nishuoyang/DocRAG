# 一键启动 DocRAG 全栈服务：Milvus (docker compose) + 后端 FastAPI + 前端 Vite
# 多 Agent 研究助理工作台（LangGraph supervisor + 5 成员 agent）
# 用法: powershell -ExecutionPolicy Bypass -File start.ps1 [-BackendOnly]
param([switch]$BackendOnly)
# 兼容 bash 习惯的双横线写法: --backend-only
if ($args -contains '--backend-only') { $BackendOnly = $true }

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'

function Write-Info  { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn  { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERR]  $args" -ForegroundColor Red }

function Test-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "未找到命令 $Name，请先安装"
        exit 1
    }
}

# 等待健康检查通过
function Wait-Health([string]$Url, [string]$Desc, [int]$TimeoutSec = 60) {
    $i = 0
    $ok = $false
    while (-not $ok) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            $ok = ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300)
        } catch {
            $ok = $false
        }
        if ($ok) { break }
        $i++
        if ($i -ge $TimeoutSec) {
            Write-Error "$Desc 在 ${TimeoutSec}s 内未就绪，请查看日志"
            return $false
        }
        Start-Sleep -Seconds 1
    }
    Write-Info "$Desc 就绪"
    return $true
}

# 1. Milvus 基础设施
if (-not $BackendOnly) {
    Test-Command 'docker'
    Write-Info "启动 Milvus 基础设施（etcd + MinIO + standalone）..."
    Push-Location $Root
    try {
        docker compose up -d
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker compose up 失败，请确认 Docker Desktop 已启动"
        exit 1
    }
    if (-not (Wait-Health 'http://127.0.0.1:9091/healthz' 'Milvus' 90)) { exit 1 }
}

# 2. 后端 FastAPI
$Python = Join-Path $Backend '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    Write-Error "未找到 $Python，请先安装依赖（poetry install）"
    exit 1
}
Write-Info "启动后端 FastAPI (http://127.0.0.1:8001)..."
Push-Location $Backend
try {
    $BackendProc = Start-Process -FilePath $Python -ArgumentList '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8001' -PassThru -WindowStyle Hidden
} finally {
    Pop-Location
}

# 3. 前端 Vite
$FrontendProc = $null
if (-not $BackendOnly) {
    Test-Command 'npm'
    $NodeModules = Join-Path $Frontend 'node_modules'
    if (-not (Test-Path $NodeModules)) {
        Write-Warn "frontend/node_modules 不存在，执行 npm install..."
        Push-Location $Frontend
        try { npm install } finally { Pop-Location }
    }
    Write-Info "启动前端 Vite (http://localhost:5173)..."
    Push-Location $Frontend
    try {
        $FrontendProc = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run', 'dev' -PassThru -WindowStyle Hidden
    } finally {
        Pop-Location
    }
}

# 等后端就绪后提示访问地址
if (-not (Wait-Health 'http://127.0.0.1:8001/health' '后端 FastAPI' 60)) { exit 1 }

Write-Host ""
Write-Host "=== 服务已启动 ===" -ForegroundColor Green
Write-Host "  后端 API:   http://127.0.0.1:8001 (Swagger: /docs)"
if (-not $BackendOnly) {
    Write-Host "  前端页面:   http://localhost:5173"
}
Write-Host ""
Write-Host "提示：按任意键停止全部服务。" -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

# 停止服务（不 kill 用户其他进程）
if ($BackendProc -and -not $BackendProc.HasExited) {
    Stop-Process -Id $BackendProc.Id -Force -ErrorAction SilentlyContinue
}
if ($FrontendProc -and -not $FrontendProc.HasExited) {
    Stop-Process -Id $FrontendProc.Id -Force -ErrorAction SilentlyContinue
}
Write-Info "已停止全部服务。"

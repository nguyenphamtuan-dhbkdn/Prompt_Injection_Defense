param(
    [ValidateSet('setup', 'test', 'test-slow', 'lint', 'typecheck', 'check', 'experiments', 'quick', 'transformer', 'redteam', 'figures', 'report', 'demo')]
    [string]$Task = 'check'
)

$ErrorActionPreference = 'Stop'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvPath = if ($uvCommand) { $uvCommand.Source } else { Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
if (-not (Test-Path -LiteralPath $uvPath)) { throw 'uv executable was not found.' }

function Invoke-Uv {
    param([string[]]$Arguments)
    & $uvPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "uv exited with code $LASTEXITCODE" }
}

Push-Location $PSScriptRoot
try {
    switch ($Task) {
        'setup' { Invoke-Uv -Arguments @('sync', '--locked', '--extra', 'dev') }
        'test' { Invoke-Uv -Arguments @('run', 'pytest') }
        'test-slow' { Invoke-Uv -Arguments @('run', '--extra', 'transformer', 'pytest', '-m', 'slow') }
        'lint' { Invoke-Uv -Arguments @('run', 'ruff', 'check', '.') }
        'typecheck' { Invoke-Uv -Arguments @('run', 'mypy', '--strict', 'src') }
        'experiments' { Invoke-Uv -Arguments @('run', 'python', 'experiments/run_all.py') }
        'quick' { Invoke-Uv -Arguments @('run', 'python', 'experiments/run_all.py', '--quick') }
        'transformer' { Invoke-Uv -Arguments @('run', '--extra', 'transformer', 'python', 'experiments/e01_transformer.py') }
        'redteam' { Invoke-Uv -Arguments @('run', 'python', 'experiments/e04_indirect_injection_vs_output_filter.py') }
        'figures' {
            Invoke-Uv -Arguments @('run', 'python', 'experiments/plotting.py')
            Invoke-Uv -Arguments @('run', 'python', 'experiments/plotting.py', '--quick')
        }
        'report' { Invoke-Uv -Arguments @('run', 'python', 'experiments/report.py') }
        'check' {
            Invoke-Uv -Arguments @('run', 'ruff', 'check', '.')
            Invoke-Uv -Arguments @('run', 'mypy', '--strict', 'src')
            Invoke-Uv -Arguments @('run', 'pytest')
        }
        default { throw "Target $Task is deferred beyond M0; see SPEC.md." }
    }
} finally { Pop-Location }

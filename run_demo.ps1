param(
    [switch]$EnableUrl
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:Path = "$([Environment]::GetEnvironmentVariable('Path', 'User'));$env:Path"

foreach ($commandName in @("py", "ffmpeg", "ffprobe")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$commandName' is not available on PATH."
    }
}

$listener = $null
$port = 7860
try {
    while ($port -le 65535) {
        $candidate = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $port
        )
        try {
            $candidate.Start()
            $listener = $candidate
            break
        }
        catch [System.Net.Sockets.SocketException] {
            $candidate.Stop()
            $port++
        }
    }
}
finally {
    if ($listener) {
        $listener.Stop()
    }
}

if (-not $listener) {
    throw "No available localhost port found at or above 7860."
}

$url = "http://127.0.0.1:$port"
Write-Output "Starting the TVSum demo at $url"
Write-Output "URL downloading is $(if ($EnableUrl) { 'enabled' } else { 'disabled' })."

Push-Location $repoRoot
try {
    $arguments = @("-3.12", "demo\app.py", "--server-name", "127.0.0.1", "--server-port", "$port")
    if ($EnableUrl) {
        $arguments += "--enable-url"
    }
    & py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Demo server exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

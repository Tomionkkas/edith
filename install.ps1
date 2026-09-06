# Put `edith` on PATH so it runs from anywhere, like any other command.
#
# PowerShell does not search the current directory - that is why it has to be
# `.\edith` until this runs. Only the USER's PATH is touched, never the
# machine's, so it needs no administrator rights and affects nobody else.
#
#     powershell -ExecutionPolicy Bypass -File .\install.ps1
#     powershell -ExecutionPolicy Bypass -File .\install.ps1 -Undo

param([switch]$Undo)

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$user = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @($user -split ";" | Where-Object { $_ -ne "" })

if ($Undo) {
    $kept = @($parts | Where-Object { $_.TrimEnd("\") -ne $repo.TrimEnd("\") })
    if ($kept.Count -eq $parts.Count) {
        Write-Host "  $repo was not on your PATH; nothing to undo."
        exit 0
    }
    [Environment]::SetEnvironmentVariable("Path", ($kept -join ";"), "User")
    Write-Host "  Removed $repo from your user PATH."
    Write-Host "  Open a new terminal for it to take effect."
    exit 0
}

if ($parts | Where-Object { $_.TrimEnd("\") -eq $repo.TrimEnd("\") }) {
    Write-Host "  Already on your PATH: $repo"
} else {
    [Environment]::SetEnvironmentVariable("Path", (($parts + $repo) -join ";"), "User")
    Write-Host "  Added to your user PATH: $repo"
    Write-Host "  (undo with:  .\install.ps1 -Undo)"
}

# Run as `powershell -File install.ps1` this is a CHILD process, so setting
# $env:Path here cannot reach the shell that launched it. The first version
# claimed "in this one it works now", and it did not.
if ($env:Path -notlike "*$repo*") { $env:Path = "$env:Path;$repo" }

Write-Host ""
Write-Host "  Now type:  edith"
Write-Host ""
Write-Host "  It works in NEW terminals. To use it in this one without opening"
Write-Host "  another, run:"
Write-Host ""
Write-Host "      `$env:Path += `";$repo`""
Write-Host ""

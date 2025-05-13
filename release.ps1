<# 
release.ps1 - Automate PyPI release for uci-phonotactic-calculator
Usage: PowerShell -ExecutionPolicy Bypass -File release.ps1
#>

$ErrorActionPreference = "Stop"

# Remove previous build artifacts
Write-Host "Removing previous build artifacts..."
$dirsToRemove = @("dist", "build")
$eggInfos = Get-ChildItem -Recurse -Directory -Filter "*.egg-info"
foreach ($dir in $dirsToRemove) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
        Write-Host "Removed $dir"
    }
}
foreach ($egg in $eggInfos) {
    Remove-Item $egg.FullName -Recurse -Force
    Write-Host "Removed $($egg.FullName)"
}

# Optionally load credentials from .pypirc
function Load-PyPiCredentials {
    $pypircPath = Join-Path $env:USERPROFILE ".pypirc"
    if (Test-Path $pypircPath) {
        Write-Host "Loading PyPI credentials from $pypircPath..."
        $inPypiSection = $false
        foreach ($line in Get-Content $pypircPath) {
            $trimmed = $line.Trim()
            if ($trimmed -like ";*" -or $trimmed -like "#*") { continue }
            if ($trimmed -match "^\[pypi\]") { $inPypiSection = $true; continue }
            if ($inPypiSection -and $trimmed -match "^\[.*\]") { break }
            if ($inPypiSection) {
                if ($trimmed -match "^\s*username\s*=\s*(.+)$") {
                    $env:TWINE_USERNAME = $Matches[1].Trim()
                }
                if ($trimmed -match "^\s*password\s*=\s*(.+)$") {
                    $env:TWINE_PASSWORD = $Matches[1].Trim()
                }
            }
        }
    }
}

if (-not $env:TWINE_USERNAME -or -not $env:TWINE_PASSWORD) {
    Load-PyPiCredentials
}

if (-not $env:TWINE_USERNAME) {
    Write-Error "TWINE_USERNAME is not set. Please set it (typically to '__token__') or configure your .pypirc file."
    exit 1
}
if (-not $env:TWINE_PASSWORD) {
    Write-Error "TWINE_PASSWORD is not set. Please set it to your PyPI API token or configure your .pypirc file."
    exit 1
}

# Build the package
Write-Host "Building package..."
python -m build

# Upload to PyPI
Write-Host "Uploading package to PyPI..."
python -m twine upload dist/*

Write-Host "Release process completed successfully."

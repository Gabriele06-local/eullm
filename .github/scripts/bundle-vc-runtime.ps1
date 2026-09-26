<#
.SYNOPSIS
    Copy the Visual C++ runtime DLLs an exe imports next to it, and prove
    none is missing.

.DESCRIPTION
    eullm.exe is linked against the dynamic MSVC runtime (MSVCP140,
    VCRUNTIME140, VCRUNTIME140_1) and OpenMP (VCOMP140). None of them ships
    with Windows: on a machine without the "Visual C++ Redistributable"
    installed the exe does not start at all ("MSVCP140.dll was not found").
    Most machines have it from some other program, which is how the
    omission went unnoticed; a clean install, Windows Sandbox or a Store
    certification machine does not.

    Microsoft allows these files to be deployed app-local, next to the exe
    (the "Distributable Code" of Visual Studio, listed in its redist.txt).
    This copies the CRT and OpenMP folders of the newest redist in the
    runner's Visual Studio install into -Destination, then reads the exe's
    import table and fails if it imports an MSVC runtime DLL that is still
    not there — the same "prove it, don't assume it" check the CUDA bundle
    runs for NVIDIA DLLs.

.PARAMETER Exe
    The eullm.exe whose imports are checked.

.PARAMETER Destination
    Folder the DLLs are copied into (the folder eullm.exe ships in).
#>
param(
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$Destination
)

$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -property installationPath
if (-not $vs) { throw 'vswhere did not find a Visual Studio install' }

# Newest redist version folder that actually has an x64 CRT. The folder
# also holds non-version entries (e.g. "onecore", "v143"), which the
# numeric sort puts aside.
$redist = Get-ChildItem "$vs\VC\Redist\MSVC" -Directory |
    Where-Object { $_.Name -match '^\d+(\.\d+)+$' -and (Test-Path "$($_.FullName)\x64") } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if (-not $redist) { throw "no MSVC redist with an x64 folder under $vs\VC\Redist\MSVC" }

$copied = @()
foreach ($pattern in 'Microsoft.VC*.CRT', 'Microsoft.VC*.OpenMP') {
    $dir = Get-ChildItem "$($redist.FullName)\x64" -Directory -Filter $pattern | Select-Object -First 1
    if (-not $dir) { throw "no $pattern folder in $($redist.FullName)\x64" }
    foreach ($dll in Get-ChildItem $dir.FullName -Filter *.dll -File) {
        Copy-Item $dll.FullName -Destination $Destination -Force
        $copied += $dll.Name
    }
}
Write-Host "Bundled MSVC runtime $($redist.Name): $($copied -join ', ')"

$dumpbin = Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $dumpbin) { throw "dumpbin.exe not found under $vs\VC\Tools\MSVC" }

# Only the start-up imports; delay-loaded ones are listed after this marker
# and are not needed for the process to start.
$out = & $dumpbin.FullName /dependents $Exe | Out-String
$head = ($out -split 'Image has the following delay load dependencies')[0]
$needed = [regex]::Matches($head, '(?im)^\s+((?:msvcp|vcruntime|vcomp|concrt)\w*\.dll)\s*$') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$have = (Get-ChildItem $Destination -Filter *.dll -File).Name
$missing = $needed | Where-Object { $name = $_; -not ($have | Where-Object { $_ -ieq $name }) }
if ($missing) {
    throw "MSVC runtime DLLs imported by $Exe but not bundled: $($missing -join ', ')"
}
Write-Host "MSVC runtime imports all bundled: $($needed -join ', ')"

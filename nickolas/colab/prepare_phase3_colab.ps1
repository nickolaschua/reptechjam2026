param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$colabRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$stagingRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".phase3_staging"))

if (-not $stagingRoot.StartsWith($colabRoot, [System.StringComparison]::OrdinalIgnoreCase) -or $stagingRoot -eq $colabRoot) {
    throw "Refusing unsafe staging path: $stagingRoot"
}

if (-not $Destination) {
    $Destination = Join-Path $PSScriptRoot "phase3_colab_bundle.zip"
}
$destinationPath = [System.IO.Path]::GetFullPath($Destination)

$bundleFiles = @(
    "nickolas/shopping_agent/agent.py",
    "nickolas/shopping_agent/agent_bge.py",
    "nickolas/shopping_agent/agent_openai.py",
    "nickolas/shopping_agent/embedding_backends.py",
    "nickolas/shopping_agent/compare_embeddings.py",
    "nickolas/shopping_agent/PHASE3_EMBEDDING_BAKEOFF.md",
    "nickolas/shopping_agent/tests/test_state_routing.py",
    "nickolas/shopping_agent/tests/test_embedding_bakeoff.py",
    "nickolas/colab/requirements-phase3-colab.txt",
    "experiment_1/run_eval_v2.py",
    "experiment_1/shopper_agent.py",
    "experiment_1/shop_agent.py",
    "techjam-conversational-search/evaluator/__init__.py",
    "techjam-conversational-search/evaluator/local_evaluator.py",
    "techjam-conversational-search/starter/__init__.py",
    "techjam-conversational-search/starter/agent.py",
    "techjam-conversational-search/data/catalog.jsonl",
    "techjam-conversational-search/data/public_set.jsonl"
)

if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}

try {
    foreach ($relativePath in $bundleFiles) {
        $source = Join-Path $repoRoot $relativePath
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Required bundle file is missing: $relativePath"
        }
        $target = Join-Path (Join-Path $stagingRoot "techjam26") $relativePath
        $targetDirectory = Split-Path -Parent $target
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $target
    }

    $manifestPath = Join-Path $stagingRoot "techjam26\COLAB_BUNDLE_MANIFEST.txt"
    $bundleFiles | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    if (Test-Path -LiteralPath $destinationPath) {
        Remove-Item -LiteralPath $destinationPath -Force
    }
    Compress-Archive -LiteralPath (Join-Path $stagingRoot "techjam26") -DestinationPath $destinationPath -CompressionLevel Optimal
    $sizeMB = [math]::Round((Get-Item -LiteralPath $destinationPath).Length / 1MB, 2)
    Write-Output "Created $destinationPath ($sizeMB MB)"
    Write-Output "The archive contains no .env file or API key."
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}

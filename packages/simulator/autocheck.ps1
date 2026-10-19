Write-Host "Running checks on Simulator package" -Foregroundcolor Cyan
Write-Host "Autoformatting..." -Foregroundcolor Yellow
ruff format src/simulator
ruff format tests
Write-Host "Linting package code..." -Foregroundcolor Yellow
ruff check --fix src/simulator
Write-Host "Linting test code..."
ruff check --fix tests
Write-Host "Type checking package code..." -Foregroundcolor Yellow
pyright src/simulator
Write-Host "Type checking test code..." -Foregroundcolor Yellow
pyright tests
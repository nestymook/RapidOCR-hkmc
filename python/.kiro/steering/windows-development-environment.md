# Windows Development Environment Guidelines

## Overview

This development environment operates on **Microsoft Windows**. All commands, scripts, and development workflows must be compatible with Windows systems. This guide ensures proper command usage and avoids Linux/Unix-specific patterns that will fail on Windows.

## Critical Windows Compatibility Rules

### 1. Command Chaining and Operators

**❌ NEVER Use Linux Command Chaining:**
```bash
# ❌ WRONG - Linux/Unix syntax that fails on Windows
command1 && command2
command1 || command2
command1; command2
```

**✅ ALWAYS Use Windows-Compatible Alternatives:**
```powershell
# ✅ CORRECT - Windows PowerShell syntax
command1; command2                    # Sequential execution
if ($?) { command2 }                  # Conditional execution (success)
if (-not $?) { command2 }            # Conditional execution (failure)
```

```cmd
# ✅ CORRECT - Windows CMD syntax
command1 & command2                   # Sequential execution
command1 && command2                  # Conditional execution (CMD only)
```

### 2. Shell Script Generation

**❌ NEVER Generate Shell Scripts (.sh files):**
```bash
# ❌ WRONG - Will not execute on Windows
#!/bin/bash
echo "This won't work on Windows"
```

**✅ ALWAYS Generate Windows Scripts:**
```powershell
# ✅ CORRECT - PowerShell script (.ps1)
Write-Host "This works on Windows"
```

```batch
# ✅ CORRECT - Batch script (.bat/.cmd)
@echo off
echo This works on Windows
```

### 3. File Path Conventions

**❌ NEVER Use Unix Path Separators:**
```bash
# ❌ WRONG - Unix paths
/home/user/project
./scripts/deploy.sh
```

**✅ ALWAYS Use Windows Path Conventions:**
```powershell
# ✅ CORRECT - Windows paths
C:\Users\User\project
.\scripts\deploy.ps1
.\scripts\deploy.bat
```

### 4. Environment Variables

**❌ NEVER Use Unix Environment Variable Syntax:**
```bash
# ❌ WRONG - Unix syntax
export VAR_NAME=value
$VAR_NAME
```

**✅ ALWAYS Use Windows Environment Variable Syntax:**
```powershell
# ✅ CORRECT - PowerShell syntax
$env:VAR_NAME = "value"
$env:VAR_NAME
```

```cmd
# ✅ CORRECT - CMD syntax
set VAR_NAME=value
%VAR_NAME%
```

## Windows-Specific Command Patterns

### 1. AgentCore Development Commands

**Local Testing:**
```powershell
# ✅ CORRECT - Windows PowerShell
python agent.py
Start-Sleep 3
Invoke-RestMethod -Uri "http://localhost:8080/ping" -Method Get
```

**Agent Deployment:**
```powershell
# ✅ CORRECT - Sequential PowerShell commands
agentcore configure -e agent.py
agentcore launch --auto-update-on-conflict
agentcore status
```

**Testing Workflow:**
```powershell
# ✅ CORRECT - PowerShell testing script
Write-Host "Testing AgentCore Agent..."
$payload = @{input = @{prompt = "Hello, are you working?"}} | ConvertTo-Json
agentcore invoke $payload
```

### 2. Virtual Environment Management

**❌ WRONG - Unix activation:**
```bash
source venv/bin/activate
```

**✅ CORRECT - Windows activation:**
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```

```cmd
# CMD
venv\Scripts\activate.bat
```

### 3. Package Installation and Management

**✅ CORRECT - Windows package management:**
```powershell
# Install packages
pip install bedrock-agentcore strands-agents

# Create requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### 4. File Operations

**✅ CORRECT - Windows file operations:**
```powershell
# Create directories
New-Item -ItemType Directory -Path "deployment" -Force

# Copy files
Copy-Item "agent.py" "backup\agent.py"

# Remove files
Remove-Item "temp_file.txt" -Force

# List files
Get-ChildItem -Path "." -Recurse
```

## Development Workflow Patterns

### 1. Local Development Setup

**✅ Windows-Compatible Setup Script:**
```powershell
# setup_development.ps1
Write-Host "Setting up AgentCore development environment..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import bedrock_agentcore; print('✅ AgentCore installed successfully')"

Write-Host "✅ Development environment ready!"
```

### 2. Testing and Validation

**✅ Windows Testing Script:**
```powershell
# test_agent.ps1
param(
    [string]$AgentFile = "agent.py",
    [int]$Port = 8080
)

Write-Host "🧪 Testing Agent: $AgentFile"

# Start agent in background
$agentProcess = Start-Process python -ArgumentList $AgentFile -PassThru -WindowStyle Hidden

# Wait for startup
Start-Sleep 5

try {
    # Test health endpoint
    $healthResponse = Invoke-RestMethod -Uri "http://localhost:$Port/ping" -Method Get -TimeoutSec 10
    
    if ($healthResponse.status -eq "healthy") {
        Write-Host "✅ Agent health check passed"
        
        # Test invocation
        $payload = @{prompt = "Hello, test message"} | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "http://localhost:$Port/invocations" -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 30
        
        Write-Host "✅ Agent invocation test passed"
        Write-Host "Response: $($response | ConvertTo-Json -Depth 3)"
    }
    else {
        Write-Host "❌ Agent health check failed"
    }
}
catch {
    Write-Host "❌ Agent test failed: $($_.Exception.Message)"
}
finally {
    # Cleanup
    if ($agentProcess -and !$agentProcess.HasExited) {
        Stop-Process -Id $agentProcess.Id -Force
        Write-Host "🧹 Agent process terminated"
    }
}
```

### 3. Deployment Automation

**✅ Windows Deployment Script:**
```powershell
# deploy_agent.ps1
param(
    [string]$AgentName = "my_agent",
    [string]$Region = "us-east-1"
)

Write-Host "🚀 Deploying Agent: $AgentName to $Region"

try {
    # Configure agent
    Write-Host "📋 Configuring agent..."
    agentcore configure -e "agent.py" --agent-name $AgentName --region $Region
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Configuration successful"
        
        # Launch agent
        Write-Host "🚀 Launching agent..."
        agentcore launch --auto-update-on-conflict
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Deployment successful"
            
            # Check status
            Write-Host "📊 Checking agent status..."
            agentcore status
        }
        else {
            Write-Host "❌ Deployment failed"
            exit 1
        }
    }
    else {
        Write-Host "❌ Configuration failed"
        exit 1
    }
}
catch {
    Write-Host "❌ Deployment error: $($_.Exception.Message)"
    exit 1
}
```

## Common Windows Command Equivalents

### File and Directory Operations

| Linux/Unix Command | Windows PowerShell Equivalent | Windows CMD Equivalent |
|-------------------|-------------------------------|----------------------|
| `ls -la` | `Get-ChildItem -Force` | `dir /a` |
| `mkdir -p dir` | `New-Item -ItemType Directory -Path dir -Force` | `mkdir dir` |
| `rm -rf dir` | `Remove-Item -Recurse -Force dir` | `rmdir /s /q dir` |
| `cp file1 file2` | `Copy-Item file1 file2` | `copy file1 file2` |
| `mv file1 file2` | `Move-Item file1 file2` | `move file1 file2` |
| `cat file.txt` | `Get-Content file.txt` | `type file.txt` |
| `grep pattern file` | `Select-String -Pattern pattern -Path file` | `findstr pattern file` |

### Process and System Operations

| Linux/Unix Command | Windows PowerShell Equivalent | Windows CMD Equivalent |
|-------------------|-------------------------------|----------------------|
| `ps aux` | `Get-Process` | `tasklist` |
| `kill -9 pid` | `Stop-Process -Id pid -Force` | `taskkill /f /pid pid` |
| `which command` | `Get-Command command` | `where command` |
| `env` | `Get-ChildItem Env:` | `set` |
| `export VAR=value` | `$env:VAR = "value"` | `set VAR=value` |

### Network and Service Operations

| Linux/Unix Command | Windows PowerShell Equivalent | Windows CMD Equivalent |
|-------------------|-------------------------------|----------------------|
| `curl -X POST url` | `Invoke-RestMethod -Uri url -Method Post` | `curl -X POST url` |
| `wget url` | `Invoke-WebRequest -Uri url` | `curl url` |
| `netstat -an` | `Get-NetTCPConnection` | `netstat -an` |

## AgentCore-Specific Windows Patterns

### 1. Local Agent Testing

**✅ Windows PowerShell Pattern:**
```powershell
# Start agent locally
python agent.py

# In another PowerShell window, test the agent
$testPayload = @{
    input = @{
        prompt = "Hello, can you help me with S3 operations?"
    }
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8080/invocations" -Method Post -Body $testPayload -ContentType "application/json"
```

### 2. Environment Setup

**✅ Windows Environment Setup:**
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install AgentCore dependencies
pip install bedrock-agentcore strands-agents bedrock-agentcore-starter-toolkit

# Verify installation
python -c "from bedrock_agentcore.runtime import BedrockAgentCoreApp; print('✅ AgentCore ready')"
```

### 3. Configuration Management

**✅ Windows Configuration Pattern:**
```powershell
# Set environment variables for development
$env:AWS_REGION = "us-east-1"
$env:LOG_LEVEL = "INFO"
$env:ENABLE_DEBUG = "true"

# Run agent with configuration
python agent.py
```

### 4. Deployment Verification

**✅ Windows Deployment Check:**
```powershell
# Check agent status
agentcore status

# Test deployed agent
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('{"input": {"prompt": "Hello, test deployment"}}'))
agentcore invoke $payload

# Monitor logs
aws logs tail "/aws/bedrock-agentcore/runtimes/AGENT_ID-DEFAULT" --since 10m --region us-east-1
```

## Error Prevention Checklist

### Before Writing Commands:

1. ✅ **Check OS Compatibility**: Ensure all commands work on Windows
2. ✅ **Use Proper Path Separators**: Use backslashes `\` for Windows paths
3. ✅ **Avoid Unix Operators**: Never use `&&`, `||`, or `;` for command chaining in shell contexts
4. ✅ **Use Windows Scripts**: Generate `.ps1` or `.bat` files, never `.sh` files
5. ✅ **Test Locally**: Always test commands in Windows PowerShell or CMD
6. ✅ **Environment Variables**: Use Windows syntax for environment variables
7. ✅ **File Operations**: Use Windows-compatible file operation commands

### Common Mistakes to Avoid:

1. ❌ **Linux Command Chaining**: `command1 && command2`
2. ❌ **Unix Paths**: `/home/user/file`
3. ❌ **Shell Scripts**: `#!/bin/bash`
4. ❌ **Unix Environment Variables**: `export VAR=value`
5. ❌ **Linux File Operations**: `rm -rf`, `ls -la`
6. ❌ **Unix Process Management**: `ps aux`, `kill -9`

## Development Best Practices

### 1. Always Test Locally First

```powershell
# ✅ CORRECT - Local testing workflow
Write-Host "Testing agent locally..."
python agent.py
# Wait for startup, then test in another window
```

### 2. Use PowerShell for Complex Operations

```powershell
# ✅ CORRECT - PowerShell for complex workflows
function Test-AgentDeployment {
    param([string]$AgentName)
    
    Write-Host "Deploying $AgentName..."
    
    try {
        agentcore configure -e "agent.py" --agent-name $AgentName
        agentcore launch --auto-update-on-conflict
        agentcore status
        Write-Host "✅ Deployment successful"
    }
    catch {
        Write-Host "❌ Deployment failed: $($_.Exception.Message)"
        return $false
    }
    
    return $true
}
```

### 3. Environment-Specific Configuration

```powershell
# ✅ CORRECT - Windows environment configuration
if ($env:OS -eq "Windows_NT") {
    Write-Host "Running on Windows - using Windows-specific configurations"
    $env:PYTHONPATH = ".\src;$env:PYTHONPATH"
    $scriptExtension = ".ps1"
}
```

## Summary

This Windows development environment requires:

1. **No Linux/Unix command syntax** (`&&`, `||`, shell scripts)
2. **Windows-compatible paths** (backslashes, drive letters)
3. **PowerShell or CMD commands** instead of bash
4. **Windows script files** (`.ps1`, `.bat`) instead of shell scripts (`.sh`)
5. **Windows environment variable syntax** (`$env:VAR` or `%VAR%`)
6. **Windows file operations** (PowerShell cmdlets or CMD commands)

Always verify commands work in Windows PowerShell or CMD before using them in development workflows.
# Python Virtual Environment Guidelines

## Virtual Environment Management

All Python projects in this workspace must use virtual environments to ensure proper dependency isolation and reproducible builds.

### Requirements

1. **Virtual Environment Creation**: Every Python project MUST create a dedicated virtual environment before installing any packages
2. **Activation Requirement**: All Python program execution MUST occur within the activated virtual environment
3. **Dependency Management**: Package installations MUST be performed within the virtual environment context

### Implementation Guidelines

#### Creating Virtual Environments

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate
```

#### Package Management

```bash
# Install packages (within activated environment)
pip install package-name

# Generate requirements file
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

#### Project Structure

```
project-root/
├── venv/                 # Virtual environment directory
├── requirements.txt      # Package dependencies
├── src/                  # Source code
└── README.md            # Project documentation
```

### Best Practices

- Always activate the virtual environment before running Python scripts
- Use `requirements.txt` to track dependencies
- Include virtual environment activation in deployment scripts
- Document virtual environment setup in project README files
- Never commit the `venv/` directory to version control

### Compliance

All Python development and execution in this workspace must follow these virtual environment guidelines to ensure consistent and isolated development environments.
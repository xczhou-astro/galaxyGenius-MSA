# How to Build the Documentation Website

## Prerequisites

1. **Activate the conda environment:**
   ```bash
   conda activate msa
   ```

2. **Install documentation dependencies:**
   ```bash
   cd docs
   pip install -r requirements.txt
   ```

## Building the Website

### Method 1: Using Make (Recommended)

```bash
# Navigate to the docs directory
cd docs

# Build HTML documentation
make html
```

The built website will be in `_build/html/` directory.

### Method 2: Using sphinx-build Directly

```bash
# Navigate to the docs directory
cd docs

# Build HTML documentation
sphinx-build -b html . _build/html
```

### Method 3: Clean Build (if you want to rebuild from scratch)

```bash
cd docs

# Clean previous builds
make clean

# Build fresh
make html
```

## Viewing the Website Locally

After building, open the documentation in your browser:

**macOS:**
```bash
open _build/html/index.html
```

**Linux:**
```bash
xdg-open _build/html/index.html
```

**Windows:**
```bash
start _build/html/index.html
```

Or simply navigate to `docs/_build/html/index.html` in your file browser and open it.

## Live Reload Development (Recommended for Development)

For automatic rebuilding when files change:

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Run with auto-reload
cd docs
sphinx-autobuild . _build/html --open-browser
```

This will:
- Automatically rebuild when you save changes
- Open the website in your browser
- Refresh automatically when changes are detected

## Other Build Formats

You can also build other formats:

```bash
# PDF (requires LaTeX)
make latexpdf

# EPUB
make epub

# Single HTML page
make singlehtml

# See all available formats
make help
```

## Troubleshooting

### Import Errors

If you get import errors when building:

1. Make sure the `msa` environment is activated
2. Ensure `galaxyGeniusMSA` package is in the Python path (it should be in `docs/galaxyGeniusMSA/`)
3. Check that all dependencies are installed

### Build Errors

If the build fails:

1. **Clean and rebuild:**
   ```bash
   make clean
   make html
   ```

2. **Check for syntax errors in .rst files:**
   ```bash
   sphinx-build -b html . _build/html -W
   ```
   The `-W` flag turns warnings into errors for debugging.

3. **Check Python path:**
   Make sure `conf.py` correctly points to the source code location.

### Module Not Found

If you see "Module not found" errors:

- Verify that `docs/galaxyGeniusMSA/` directory exists
- Check that `conf.py` has: `sys.path.insert(0, os.path.abspath('.'))`
- Ensure the package structure is correct

## Quick Reference

```bash
# Full build process
conda activate msa
cd docs
pip install -r requirements.txt
make html
open _build/html/index.html
```

## Build Output Location

All build outputs are in:
- `docs/_build/html/` - HTML documentation (this is what you deploy)
- `docs/_build/doctrees/` - Internal build cache (can be deleted)

The `_build/` directory is ignored by git (see `.gitignore`).

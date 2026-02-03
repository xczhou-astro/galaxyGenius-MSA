# How to Initialize Documentation Website on GitHub Pages

This guide will help you set up your Sphinx documentation website on GitHub Pages.

## Method 1: Using GitHub Actions (Recommended - Automatic)

This method automatically builds and deploys your documentation whenever you push to the main branch.

### Step 1: Enable GitHub Pages

1. Go to your GitHub repository
2. Click on **Settings** (top menu)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select **"GitHub Actions"** (not "Deploy from a branch")
5. Click **Save**

### Step 2: Push the Workflow File

The GitHub Actions workflow file (`.github/workflows/docs.yml`) is already created. You just need to commit and push it:

```bash
# Make sure you're in the repository root
cd galaxyGenius-MSA

# Add the workflow file
git add .github/workflows/docs.yml

# Commit
git commit -m "Add GitHub Actions workflow for documentation"

# Push to GitHub
git push origin main  # or 'master' depending on your branch name
```

### Step 3: Verify Deployment

1. Go to your repository on GitHub
2. Click on the **Actions** tab
3. You should see the workflow running
4. Once it completes, go to **Settings → Pages**
5. Your site will be available at: `https://<username>.github.io/<repository-name>/`

**Note:** The first deployment may take a few minutes. Subsequent deployments happen automatically on every push to main/master.

---

## Method 2: Manual Deployment to gh-pages Branch

If you prefer manual control over when to deploy:

### Step 1: Build the Documentation Locally

```bash
# Activate environment
conda activate msa

# Navigate to repository root
cd galaxyGenius-MSA

# Install dependencies
pip install -r requirements.txt

# Build documentation
make html
```

### Step 2: Deploy Using ghp-import (Easiest)

```bash
# Install ghp-import
pip install ghp-import

# Deploy to gh-pages branch
ghp-import -n -p -f _build/html
```

The flags mean:
- `-n`: Create `.nojekyll` file (important for Sphinx)
- `-p`: Push to GitHub
- `-f`: Force push (overwrites existing gh-pages)

### Step 3: Enable GitHub Pages

1. Go to **Settings → Pages**
2. Under **Source**, select **"Deploy from a branch"**
3. Select **"gh-pages"** branch
4. Select **"/ (root)"** folder
5. Click **Save**

Your site will be available at: `https://<username>.github.io/<repository-name>/`

---

## Method 3: Manual Deployment (Full Control)

If you want complete control:

### Step 1: Build Documentation

```bash
conda activate msa
cd galaxyGenius-MSA
pip install -r requirements.txt
make html
```

### Step 2: Create/Checkout gh-pages Branch

```bash
# Create orphan branch (if it doesn't exist)
git checkout --orphan gh-pages

# Remove all files from staging
git rm -rf .

# Copy built documentation
cp -r _build/html/* .

# Create .nojekyll file (important!)
touch .nojekyll

# Add all files
git add .

# Commit
git commit -m "Deploy documentation"

# Push to GitHub
git push origin gh-pages

# Return to main branch
git checkout main
```

### Step 3: Enable GitHub Pages

Same as Method 2, Step 3.

---

## Troubleshooting

### Site Shows 404

- Make sure `.nojekyll` file exists in the root of gh-pages branch (or in the deployed directory)
- Wait a few minutes for GitHub to process the deployment
- Check the Actions tab for any build errors

### Documentation Not Updating

- Clear your browser cache
- Check that the workflow/build completed successfully
- Verify the correct branch is selected in Pages settings

### Build Errors in GitHub Actions

- Check the Actions tab for error messages
- Ensure `requirements.txt` includes all necessary dependencies
- Verify Python version matches your local environment (currently set to 3.11)

### Custom Domain

If you want to use a custom domain:

1. Add a `CNAME` file in `_build/html/` with your domain name
2. Configure DNS settings as instructed by GitHub
3. Update the CNAME file in your repository

---

## Quick Reference

**Automatic (Recommended):**
```bash
# Just push to main branch - GitHub Actions handles the rest!
git push origin main
```

**Manual (ghp-import):**
```bash
make html
ghp-import -n -p -f _build/html
```

**Manual (Full):**
```bash
make html
git checkout --orphan gh-pages
git rm -rf .
cp -r _build/html/* .
touch .nojekyll
git add .
git commit -m "Deploy docs"
git push origin gh-pages
git checkout main
```

---

## Recommended Approach

**Use Method 1 (GitHub Actions)** - it's automatic, reliable, and keeps your documentation always up-to-date with your code!

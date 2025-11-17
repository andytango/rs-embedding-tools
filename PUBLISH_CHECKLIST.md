# Quick Publishing Checklist

## Before Publishing

- [ ] **Update package.json** with your details:
  - [ ] `name`: Choose unique package name (check: `npm search embedding-tools`)
  - [ ] `author`: Update "Andrew Hall <andrewhall@example.com>" with your actual info
  - [ ] `repository.url`: Update with your actual GitHub URL

- [ ] **Check name availability:**
  ```bash
  npm search embedding-tools  # or your chosen name
  ```

- [ ] **Create npm account** (if you don't have one):
  - Go to https://www.npmjs.com/signup

- [ ] **Get npm token:**
  - Go to https://www.npmjs.com/settings/tokens
  - Click "Generate New Token" → "Automation"
  - Copy the token

- [ ] **Add token to GitHub:**
  - Your GitHub repo → Settings → Secrets and variables → Actions
  - New repository secret
  - Name: `NPM_TOKEN`
  - Value: Paste your npm token

## Publishing Steps

### Option A: Automated (Recommended)

```bash
# 1. Commit everything
git add .
git commit -m "chore: prepare for release"
git push origin main

# 2. Create version tag
git tag 0.1.0
git push origin 0.1.0

# 3. Watch GitHub Actions build and publish automatically
# Go to: https://github.com/YOUR_USERNAME/rs-embedding-tools/actions
```

### Option B: Manual (single platform only)

```bash
npm login
npm run build
npm publish
```

## After Publishing

```bash
# Verify it worked
npm view embedding-tools

# Test installation
cd /tmp
mkdir test && cd test
npm install embedding-tools
node -e "console.log(require('embedding-tools'))"
```

## Quick Commands

```bash
# Check current version
npm view embedding-tools version

# Update version
npm version patch   # 0.1.0 → 0.1.1
npm version minor   # 0.1.1 → 0.2.0
npm version major   # 0.2.0 → 1.0.0

# Publish new version
git push origin main --follow-tags
```

## Need Help?

See full guide: [PUBLISHING.md](./PUBLISHING.md)

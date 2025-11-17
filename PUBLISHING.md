# Publishing Guide for embedding-tools

This guide explains how to publish the `embedding-tools` Node.js package to npm with pre-built binaries for all supported platforms.

## Prerequisites

1. **npm account**: Create one at [npmjs.com](https://www.npmjs.com/signup) if you don't have one
2. **GitHub repository**: Push your code to GitHub
3. **npm authentication token**: Generate from [npmjs.com/settings/tokens](https://www.npmjs.com/settings/tokens)

## Initial Setup

### 1. Check Package Name Availability

```bash
npm search embedding-tools
```

If the name is taken, you have two options:
- Choose a different name (update `package.json` → `name`)
- Use a scoped package: `@your-username/embedding-tools`

### 2. Update package.json

Make sure these fields are correct:
- `name`: Your chosen package name
- `version`: Follow [semver](https://semver.org/) (e.g., `0.1.0`)
- `author`: Your name and email
- `repository.url`: Your GitHub repository URL

### 3. Set up GitHub Secrets

Add your npm token to GitHub repository secrets:

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `NPM_TOKEN`
4. Value: Your npm authentication token (from npmjs.com → Access Tokens → Generate New Token → **Automation**)

## Publishing Methods

### Method 1: Automated Multi-Platform Publishing (Recommended)

This method uses GitHub Actions to build binaries for all platforms and publish automatically.

**Platforms built:**
- macOS x64 & Apple Silicon (arm64)
- Linux x64 & arm64 (glibc only, **no musl**)
- Windows x64 & arm64 (msvc)

**Steps:**

1. **Commit all changes:**
```bash
git add .
git commit -m "chore: prepare for v0.1.0 release"
git push origin main
```

2. **Create and push a version tag:**
```bash
# For a stable release
git tag 0.1.0
git push origin 0.1.0

# Or for a prerelease
git tag 0.1.0-beta.1
git push origin 0.1.0-beta.1
```

3. **Monitor GitHub Actions:**
   - Go to your repository → **Actions** tab
   - Watch the CI workflow build and test on all platforms
   - If all tests pass, it will automatically publish to npm

4. **Verify publication:**
```bash
npm view embedding-tools  # or your package name
```

**How it works:**
- The GitHub Actions workflow (`.github/workflows/ci.yml`) detects version tags
- It builds native binaries for all platforms in parallel
- Tests run on each platform
- Creates platform-specific packages (e.g., `embedding-tools-darwin-arm64`)
- Publishes the main package and all platform packages to npm

### Method 2: Manual Local Publishing (Simple, single platform only)

This method only builds for your current platform. **Not recommended** for production as users on other platforms won't be able to install.

```bash
# Log in to npm
npm login

# Build for your current platform
npm run build

# Publish
npm publish
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major version** (1.0.0): Breaking changes
- **Minor version** (0.1.0): New features, backwards compatible
- **Patch version** (0.1.1): Bug fixes

Examples:
```bash
# Stable releases
git tag 1.0.0    # Major release
git tag 1.1.0    # Minor release  
git tag 1.1.1    # Patch release

# Pre-releases
git tag 1.0.0-alpha.1
git tag 1.0.0-beta.1
git tag 1.0.0-rc.1
```

## Publishing Workflow

### For Regular Releases (e.g., 1.0.0)

```bash
# 1. Update version in package.json
npm version 1.0.0

# 2. Commit and push
git push origin main
git push origin 1.0.0

# 3. GitHub Actions will automatically publish to the 'latest' tag on npm
```

### For Pre-releases (e.g., 1.0.0-beta.1)

```bash
# 1. Update version in package.json
npm version 1.0.0-beta.1

# 2. Commit and push
git push origin main
git push origin 1.0.0-beta.1

# 3. GitHub Actions will automatically publish to the 'next' tag on npm
```

Users can install pre-releases with:
```bash
npm install embedding-tools@next
```

## Testing Before Publishing

Always test locally before pushing a version tag:

```bash
# Build
npm run build

# Run tests
npm test

# Check package contents
npm pack
tar -xzf embedding-tools-0.1.0.tgz
ls package/

# Clean up
rm -rf package embedding-tools-0.1.0.tgz
```

## Troubleshooting

### Build fails on a platform

Check the GitHub Actions logs for specific errors. Common issues:
- Missing dependencies in Cargo.toml
- Platform-specific compilation errors
- Test failures

### npm publish fails

- **Authentication error**: Check your `NPM_TOKEN` secret in GitHub
- **Package name taken**: Choose a different name or use a scoped package
- **Version conflict**: You can't republish the same version. Increment the version number.

### Users report installation failures

- Check they're not on musl-based Linux (Alpine) - this is not supported
- Verify all platform packages were published: `npm view embedding-tools`
- Check the `optionalDependencies` in package.json are correct

## Post-Publishing

1. **Verify installation:**
```bash
# In a clean directory
mkdir test-install && cd test-install
npm install embedding-tools
node -e "console.log(require('embedding-tools'))"
```

2. **Update GitHub Release:**
   - Go to your repo → **Releases** → Create release from tag
   - Add release notes describing changes

3. **Announce:**
   - Update README with installation instructions
   - Share on relevant platforms (Twitter, Reddit, etc.)

## Updating the Package

To publish a new version:

1. Make your changes
2. Update version: `npm version patch` (or `minor`, `major`)
3. Push: `git push origin main --follow-tags`
4. GitHub Actions handles the rest

## Platform Support Matrix

| Platform | Architecture | Supported | Note |
|----------|-------------|-----------|------|
| macOS | x64 | ✅ | |
| macOS | arm64 (Apple Silicon) | ✅ | |
| Linux | x64 (glibc) | ✅ | Ubuntu, Debian, CentOS, etc. |
| Linux | arm64 (glibc) | ✅ | |
| Linux | x64 (musl) | ❌ | Alpine Linux NOT supported |
| Windows | x64 (msvc) | ✅ | |
| Windows | arm64 (msvc) | ✅ | |

## Important Notes

- **No musl support**: The package explicitly does not support musl-based Linux distributions (like Alpine Linux) due to Rust compatibility issues
- **Pre-built binaries**: Users don't need Rust or build tools installed
- **Automatic platform detection**: The `index.js` wrapper automatically loads the correct binary for the user's platform
- **Fallback**: If a platform binary is missing, installation will fail with a clear error message

## Need Help?

- Check GitHub Actions logs for build/publish errors
- Review npm documentation: [docs.npmjs.com](https://docs.npmjs.com/)
- Check napi-rs documentation: [napi.rs](https://napi.rs/)

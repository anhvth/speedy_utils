---
name: 'skill-creation'
description: 'Guide for creating new Agent Skills with proper structure, frontmatter, bundled assets, and validation. Includes templates, best practices, and examples for building reusable skill resources.'
---

# Skill Creation Guide

This skill provides comprehensive guidance for creating new Agent Skills in the awesome-copilot repository. Use this when you need to create a self-contained, reusable skill with instructions and optional bundled assets.

## When to Create a Skill

Create a skill when you need:
- **Reusable workflows** that combine instructions with bundled resources (scripts, templates, data files)
- **Complex, multi-step processes** that benefit from structured guidance
- **Domain-specific toolkits** with reference materials and code samples
- **Specialized capabilities** that extend beyond simple prompts or instructions

**Don't create a skill if:**
- A simple prompt file (`.prompt.md`) would suffice for a one-off task
- An instruction file (`.instructions.md`) is more appropriate for coding standards
- The task doesn't require bundled assets or complex guidance

## Skill Structure

Every skill is a folder containing:
```
skills/
  your-skill-name/
    SKILL.md          # Required: Main skill definition with frontmatter
    script.py         # Optional: Bundled scripts
    template.txt      # Optional: Code templates
    reference.json    # Optional: Reference data
    assets/           # Optional: Additional resources
```

## Creating a New Skill

### Method 1: Using the Creation Script (Recommended)

```bash
npm run skill:create -- --name your-skill-name --description 'Your skill description here'
```

This will:
1. Create the skill folder with proper naming
2. Generate a `SKILL.md` template with valid frontmatter
3. Set up the basic structure

### Method 2: Manual Creation

1. **Create the folder**: `skills/your-skill-name/`
   - Use lowercase letters only
   - Separate words with hyphens
   - Keep names concise and descriptive

2. **Create SKILL.md** with proper frontmatter:
   ```markdown
   ---
   name: 'your-skill-name'
   description: 'A clear, concise description of what this skill does and when to use it.'
   ---
   
   # Your Skill Title
   
   [Skill content here]
   ```

## Frontmatter Requirements

The SKILL.md file **must** include markdown frontmatter with these fields:

### Required Fields

#### `name`
- **Type**: String (wrapped in single quotes)
- **Format**: Lowercase with hyphens (e.g., `'web-testing'`, `'skill-creation'`)
- **Rules**: 
  - Must match the folder name exactly
  - Maximum 64 characters
  - Only lowercase letters, numbers, and hyphens
  - Cannot start or end with a hyphen

**Example:**
```yaml
name: 'api-testing-toolkit'
```

#### `description`
- **Type**: String (wrapped in single quotes)
- **Length**: 10-1024 characters
- **Purpose**: Brief summary of the skill's purpose and capabilities
- **Style**: Should be clear, concise, and informative

**Example:**
```yaml
description: 'Comprehensive toolkit for testing REST APIs with sample requests, response validation, and debugging utilities.'
```

### Example Complete Frontmatter

```yaml
---
name: 'database-migration'
description: 'Guide for creating and managing database migrations with schema versioning, rollback procedures, and best practices for multiple database systems.'
---
```

## Skill Content Structure

After the frontmatter, structure your SKILL.md with these sections:

### 1. Introduction
Brief overview of what the skill does.

```markdown
# Database Migration Toolkit

This skill provides comprehensive guidance for creating, managing, and executing database migrations.
```

### 2. When to Use This Skill
Clear criteria for when this skill should be invoked.

```markdown
## When to Use This Skill

Use this skill when you need to:
- Create new database schema migrations
- Version control database changes
- Rollback problematic migrations
- Migrate between different database systems
```

### 3. Prerequisites (if applicable)
List any required tools, dependencies, or setup.

```markdown
## Prerequisites

- Database access credentials
- Migration tool installed (e.g., Alembic, Flyway, Liquibase)
- Backup of production data before running migrations
```

### 4. Core Capabilities
Detail what the skill can help accomplish.

```markdown
## Core Capabilities

### Schema Management
- Create tables, indexes, and constraints
- Alter existing schema structures
- Drop deprecated objects

### Data Migration
- Transform data between schema versions
- Bulk data imports/exports
- Data validation and cleanup
```

### 5. Usage Examples
Provide concrete examples with code snippets.

```markdown
## Usage Examples

### Example 1: Create a Migration File
\`\`\`bash
alembic revision -m "add users table"
\`\`\`

### Example 2: Apply Migration
\`\`\`bash
alembic upgrade head
\`\`\`
```

### 6. Guidelines
Best practices and recommendations.

```markdown
## Guidelines

1. **Always backup before migrating** - Create backups of production databases
2. **Test migrations locally** - Verify migrations work on development data first
3. **Use transactions** - Wrap migrations in transactions when possible
4. **Document changes** - Include clear comments in migration files
```

### 7. Common Patterns
Reusable code patterns and solutions.

```markdown
## Common Patterns

### Pattern: Reversible Migration
\`\`\`python
def upgrade():
    op.add_column('users', sa.Column('email', sa.String(255)))

def downgrade():
    op.drop_column('users', 'email')
\`\`\`
```

### 8. Limitations (if applicable)
Known constraints or edge cases.

```markdown
## Limitations

- Cannot handle cross-database migrations automatically
- Large data migrations may require manual chunking
- Some database-specific features may not be portable
```

## Bundled Assets

Skills can include bundled files to support the instructions:

### Asset Types

1. **Scripts** (`.py`, `.js`, `.sh`, etc.)
   - Automation scripts
   - Helper utilities
   - Example implementations

2. **Templates** (`.txt`, `.md`, `.json`, etc.)
   - Code templates
   - Configuration templates
   - Documentation templates

3. **Reference Data** (`.json`, `.yaml`, `.csv`, etc.)
   - Sample data
   - Configuration examples
   - Lookup tables

4. **Documentation** (`.md`, `.pdf`, etc.)
   - Extended guides
   - API references
   - Cheatsheets

### Asset Guidelines

- **Reference in SKILL.md**: Always mention bundled assets in the instructions
- **Keep files small**: Each file should be under 5MB
- **Use descriptive names**: Make filenames clear and self-documenting
- **Organize with folders**: Use subdirectories for complex skills

### Example Asset Structure

```
skills/api-testing/
  SKILL.md
  scripts/
    test-runner.py
    validate-response.js
  templates/
    request-template.json
    test-suite-template.yaml
  examples/
    sample-api-test.md
  reference/
    http-status-codes.json
```

### Referencing Assets in SKILL.md

```markdown
## Using the Test Runner Script

This skill includes a test runner script located at `scripts/test-runner.py`.

To use it:
\`\`\`bash
python scripts/test-runner.py --config config.json
\`\`\`

See `examples/sample-api-test.md` for a complete example.
```

## Validation

### Before Committing

Run the validation command:
```bash
npm run skill:validate
```

This checks:
- ✅ SKILL.md exists in each skill folder
- ✅ Frontmatter is present and valid
- ✅ `name` field matches folder name
- ✅ `name` is lowercase with hyphens (max 64 chars)
- ✅ `description` is 10-1024 characters
- ✅ Description is wrapped in single quotes

### Manual Validation Checklist

- [ ] Folder name is lowercase-with-hyphens
- [ ] SKILL.md has frontmatter with `name` and `description`
- [ ] `name` matches folder name exactly
- [ ] `description` is clear and informative (10-1024 chars)
- [ ] All sections are present and well-documented
- [ ] Bundled assets are referenced in the instructions
- [ ] Asset files are under 5MB each
- [ ] Examples are practical and runnable
- [ ] Guidelines are actionable
- [ ] README.md has been updated (`npm run build`)

## Complete Example: Creating a "code-review" Skill

### Step 1: Create Folder
```bash
mkdir skills/code-review
```

### Step 2: Create SKILL.md

```markdown
---
name: 'code-review'
description: 'Automated code review toolkit with checklists, linting rules, and best practice guidelines for multiple programming languages.'
---

# Code Review Toolkit

This skill provides comprehensive code review guidance and automation tools.

## When to Use This Skill

Use this skill when you need to:
- Perform thorough code reviews
- Apply language-specific best practices
- Check for common code smells
- Ensure coding standards compliance

## Prerequisites

- Access to the codebase being reviewed
- Linting tools installed (optional but recommended)

## Core Capabilities

### Review Checklists
- Security vulnerability checks
- Performance optimization opportunities
- Code maintainability assessment
- Documentation completeness

### Automated Analysis
- Static code analysis
- Complexity metrics
- Test coverage evaluation

## Usage Examples

### Example 1: Basic Review
Review a pull request for common issues and suggest improvements.

### Example 2: Security Audit
Focus on security vulnerabilities and potential exploits.

## Guidelines

1. **Be constructive** - Provide actionable feedback
2. **Check context** - Understand the purpose before critiquing
3. **Prioritize issues** - Focus on critical problems first
4. **Suggest solutions** - Don't just point out problems

## Common Patterns

### Pattern: Checklist-Based Review
Use the bundled `checklists/python-review.md` for Python code reviews.

## Limitations

- Automated tools may miss context-specific issues
- Human judgment still required for architecture decisions
```

### Step 3: Add Bundled Assets (Optional)

Create `skills/code-review/checklists/python-review.md`:

```markdown
# Python Code Review Checklist

## Style & Formatting
- [ ] Follows PEP 8 style guide
- [ ] Docstrings present for all public functions
- [ ] Type hints used appropriately

## Functionality
- [ ] Error handling implemented
- [ ] Edge cases covered
- [ ] No code duplication
```

### Step 4: Validate

```bash
npm run skill:validate
```

### Step 5: Update README

```bash
npm run build
```

### Step 6: Fix Line Endings

```bash
bash scripts/fix-line-endings.sh
```

## Best Practices

### 1. Be Specific and Actionable
❌ "This skill helps with testing"
✅ "This skill provides Playwright-based browser automation for testing web applications with screenshot capture and console log inspection"

### 2. Include Concrete Examples
Always provide runnable code examples that demonstrate the skill's usage.

### 3. Document Prerequisites Clearly
List all required tools, dependencies, and setup steps upfront.

### 4. Keep Assets Organized
Use subdirectories for multiple asset types:
```
skills/your-skill/
  scripts/
  templates/
  examples/
  reference/
```

### 5. Reference Assets Explicitly
Don't just bundle files—explain when and how to use them in SKILL.md.

### 6. Test Your Skill
Before committing:
- Validate frontmatter and structure
- Test bundled scripts
- Verify examples are runnable
- Update README with `npm run build`

### 7. Follow the Agent Skills Specification
This repository follows the [Agent Skills specification](https://agentskills.io/specification) for maximum compatibility.

## Common Mistakes to Avoid

❌ **Forgetting quotes in frontmatter**
```yaml
name: skill-name  # Wrong
```
```yaml
name: 'skill-name'  # Correct
```

❌ **Mismatched folder and name**
```
skills/web-testing/
  SKILL.md with name: 'webapp-testing'  # Wrong
```

❌ **Description too short**
```yaml
description: 'Testing tool'  # Only 12 chars, needs 10+ but should be descriptive
```

❌ **Uppercase in folder name**
```
skills/WebTesting/  # Wrong
skills/web-testing/  # Correct
```

❌ **Not referencing bundled assets**
Including `script.py` but never mentioning it in SKILL.md.

❌ **Skipping validation**
Not running `npm run skill:validate` before committing.

## Additional Resources

- [Agent Skills Specification](https://agentskills.io/specification)
- [Project Documentation](../../docs/README.skills.md)
- [AGENTS.md](../../AGENTS.md) - Full project overview
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines

## Workflow Summary

1. **Plan** - Determine if a skill is the right resource type
2. **Create** - Use `npm run skill:create` or create manually
3. **Write** - Add comprehensive instructions and examples
4. **Bundle** - Include relevant scripts, templates, or data
5. **Validate** - Run `npm run skill:validate`
6. **Build** - Run `npm run build` to update README
7. **Normalize** - Run `bash scripts/fix-line-endings.sh`
8. **Commit** - Submit your pull request

## Meta: About This Skill

This skill itself follows all the guidelines it recommends. It demonstrates:
- ✅ Proper frontmatter with `name` and `description`
- ✅ Clear section structure
- ✅ Concrete examples and code snippets
- ✅ Actionable guidelines
- ✅ Common patterns and anti-patterns
- ✅ Comprehensive documentation

Use this as a reference template when creating your own skills.

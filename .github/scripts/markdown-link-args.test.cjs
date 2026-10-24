const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const test = require('node:test');
const markdownLinkArgs = require('./markdown-link-args.cjs');

test('PRs check added, modified and renamed Markdown, excluding deleted files', () => {
  assert.equal(
    markdownLinkArgs('pull_request', [
      { filename: 'README.md', status: 'modified' },
      { filename: 'docs/new.md', status: 'added' },
      { filename: 'docs/renamed.md', status: 'renamed' },
      { filename: 'docs/deleted.md', status: 'removed' },
      { filename: 'src/index.ts', status: 'modified' },
    ]),
    "'./README.md' './docs/new.md' './docs/renamed.md'"
  );
});

test('code-only PRs have no Markdown inputs', () => {
  assert.equal(
    markdownLinkArgs('pull_request', [
      { filename: 'index.ts', status: 'modified' },
    ]),
    ''
  );
});

test('scheduled, manual and push runs retain the full audit', () => {
  for (const event of ['schedule', 'workflow_dispatch', 'push']) {
    assert.equal(markdownLinkArgs(event), "'./**/*.md'");
  }
});

test('PR file limit falls back to a full audit', () => {
  assert.equal(
    markdownLinkArgs(
      'pull_request',
      Array(3000).fill({ filename: 'index.ts' })
    ),
    "'./**/*.md'"
  );
});

test('shell metacharacters and spaces stay literal file names', () => {
  const filename = 'docs/a b\'$(printf INJECTED)"`printf INJECTED`.md';
  const args = markdownLinkArgs('pull_request', [
    { filename, status: 'added' },
  ]);
  assert.equal(
    execFileSync('bash', ['-c', `printf '%s' ${args}`], { encoding: 'utf8' }),
    `./${filename}`
  );
});

module.exports = (eventName, files = []) => {
  // GitHub returns at most 3,000 PR files; audit everything if truncated.
  if (eventName !== 'pull_request' || files.length >= 3000) {
    return "'./**/*.md'";
  }

  return (
    files
      .filter(
        ({ filename, status }) =>
          filename.endsWith('.md') && status !== 'removed'
      )
      // The pinned lychee action evaluates its arguments in a shell.
      .map(({ filename }) => `'./${filename.replaceAll("'", "'\\''")}'`)
      .join(' ')
  );
};

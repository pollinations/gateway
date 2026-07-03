/** @type {import('ts-jest').JestConfigWithTsJest} **/
export default {
  testEnvironment: 'node',
  transform: {
    '^.+.tsx?$': [
      'ts-jest',
      {
        // These files carry pre-existing type errors (unrelated to any test)
        // that would otherwise block ts-jest from transpiling modules that
        // import them. Diagnostics are scoped off for just these paths; every
        // other file keeps full type-checking, and `tsc --noEmit` remains the
        // authoritative type gate for the whole tree.
        diagnostics: {
          exclude: [
            '**/providers/google/chatComplete.ts',
            '**/providers/google-vertex-ai/chatComplete.ts',
            '**/providers/google-vertex-ai/transformGenerationConfig.ts',
          ],
        },
      },
    ],
  },
  testTimeout: 30000, // Set default timeout to 30 seconds
};

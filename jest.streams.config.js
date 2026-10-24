import config from './jest.config.js';

const [transformer, options] = config.transform['^.+.tsx?$'];

export default {
  ...config,
  extensionsToTreatAsEsm: ['.ts'],
  testMatch: ['**/src/handlers/streamHandler.test.ts'],
  transform: { '^.+.tsx?$': [transformer, { ...options, useESM: true }] },
};

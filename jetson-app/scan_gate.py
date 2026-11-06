"""Pure bounded consensus for repeated scan observations."""


class ScanConsensus:
    """Require consecutive matching (shape, color) observations within a budget."""

    def __init__(self, max_attempts=8, required_matches=2):
        if (isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
                or isinstance(required_matches, bool) or not isinstance(required_matches, int)
                or not 1 <= required_matches <= max_attempts <= 20):
            raise ValueError('require integer limits: 1 <= required_matches <= max_attempts <= 20')
        self.max_attempts = max_attempts
        self.required_matches = required_matches
        self.attempts = 0
        self.matches = 0
        self.candidate = None
        self._terminal = False

    def observe(self, key):
        """Return accept, retry or exhausted; invalid inputs consume no attempts."""
        if self._terminal:
            raise ValueError('scan consensus is already terminal')
        if key is not None and (
                not isinstance(key, tuple) or len(key) != 2
                or any(not isinstance(value, str) or not value.strip() for value in key)):
            raise ValueError('key must be None or a pair of non-empty strings')
        self.attempts += 1
        if key is None:
            self.candidate = None
            self.matches = 0
        elif key == self.candidate:
            self.matches += 1
        else:
            self.candidate = key
            self.matches = 1
        if self.matches >= self.required_matches:
            self._terminal = True
            return 'accept'
        if self.attempts >= self.max_attempts:
            self._terminal = True
            return 'exhausted'
        return 'retry'

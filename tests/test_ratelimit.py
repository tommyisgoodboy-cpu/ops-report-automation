import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
    from ops_report.api import create_app
    from ops_report.ratelimit import RateLimiter
except ImportError:  # API deps are optional
    TestClient = None


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class RateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.limiter = RateLimiter(limit=3, window=60, clock=self.clock)

    def test_allows_up_to_the_limit_then_blocks(self):
        for _ in range(3):
            self.assertEqual(self.limiter.check("a"), (True, 0))
        allowed, retry = self.limiter.check("a")
        self.assertFalse(allowed)
        self.assertEqual(retry, 60)

    def test_retry_after_counts_down(self):
        for _ in range(3):
            self.limiter.check("a")
        self.clock.now = 25
        self.assertEqual(self.limiter.check("a"), (False, 35))

    def test_window_slides_and_allows_again(self):
        for _ in range(3):
            self.limiter.check("a")
        self.clock.now = 60
        self.assertTrue(self.limiter.check("a")[0])

    def test_clients_are_independent(self):
        for _ in range(3):
            self.limiter.check("a")
        self.assertFalse(self.limiter.check("a")[0])
        self.assertTrue(self.limiter.check("b")[0])

    def test_memory_is_bounded(self):
        small = RateLimiter(limit=1, max_keys=3, clock=self.clock)
        for i in range(20):
            small.check(f"ip{i}")
        self.assertLessEqual(len(small._hits), 3)


@unittest.skipIf(TestClient is None, "fastapi / httpx not installed")
class RateLimitApiTests(unittest.TestCase):
    ASK = {"question": "VPN keeps dropping"}

    def test_returns_429_with_retry_after_when_over_limit(self):
        client = TestClient(create_app(n_tickets=20, rate_limit=2))
        self.assertEqual(client.post("/ask", json=self.ASK).status_code, 200)
        self.assertEqual(client.post("/ask", json=self.ASK).status_code, 200)
        r = client.post("/ask", json=self.ASK)
        self.assertEqual(r.status_code, 429)
        self.assertIn("Too many requests", r.json()["detail"])
        self.assertGreaterEqual(int(r.headers["retry-after"]), 1)

    def test_limit_is_shared_across_the_protected_endpoints(self):
        client = TestClient(create_app(n_tickets=20, rate_limit=2))
        self.assertEqual(client.get("/tickets").status_code, 200)
        self.assertEqual(client.post("/classify", json={"summary": "vpn down"}).status_code, 200)
        self.assertEqual(client.get("/report").status_code, 429)

    def test_health_and_home_are_never_limited(self):
        client = TestClient(create_app(n_tickets=20, rate_limit=1))
        for _ in range(5):
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)

    def test_zero_disables_the_limit(self):
        client = TestClient(create_app(n_tickets=20, rate_limit=0))
        for _ in range(40):
            self.assertEqual(client.get("/tickets").status_code, 200)

    def test_env_var_sets_the_limit(self):
        with mock.patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "1"}):
            client = TestClient(create_app(n_tickets=20))
        self.assertEqual(client.get("/tickets").status_code, 200)
        self.assertEqual(client.get("/tickets").status_code, 429)

    def test_forged_forwarded_for_is_ignored_by_default(self):
        client = TestClient(create_app(n_tickets=20, rate_limit=2))
        statuses = [client.get("/tickets", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
                    for i in range(3)]
        self.assertEqual(statuses, [200, 200, 429])  # rotating the header did not help

    def test_behind_a_trusted_proxy_the_last_address_is_used(self):
        with mock.patch.dict(os.environ, {"RATE_LIMIT_TRUST_PROXY": "1"}):
            client = TestClient(create_app(n_tickets=20, rate_limit=1))
        # Caller rotates the first (forgeable) entry; the proxy-written last entry is the same.
        a1 = client.get("/tickets", headers={"X-Forwarded-For": "1.1.1.1, 9.9.9.9"})
        a2 = client.get("/tickets", headers={"X-Forwarded-For": "2.2.2.2, 9.9.9.9"})
        # A genuinely different client (different last entry) has its own budget.
        b = client.get("/tickets", headers={"X-Forwarded-For": "1.1.1.1, 8.8.8.8"})
        self.assertEqual((a1.status_code, a2.status_code, b.status_code), (200, 429, 200))


if __name__ == "__main__":
    unittest.main()

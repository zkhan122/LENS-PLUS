from __future__ import annotations

import unittest

try:
    from fastapi.testclient import TestClient
    from app import main as app_main
    _HAS_FASTAPI = True
except ModuleNotFoundError:
    TestClient = None  # type: ignore[assignment]
    app_main = None  # type: ignore[assignment]
    _HAS_FASTAPI = False


class _FakePeerConnection:
    connectionState = "connected"
    iceConnectionState = "completed"

    async def close(self) -> None:
        return None


@unittest.skipUnless(_HAS_FASTAPI, "fastapi/testclient not installed")
class ApiIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        app_main.sessions.clear()
        self.client = TestClient(app_main.app)

    def tearDown(self) -> None:
        app_main.sessions.clear()
        self.client.close()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_debug_sessions_endpoint(self) -> None:
        response = self.client.get("/debug/sessions")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("sessions", payload)
        self.assertIsInstance(payload["sessions"], list)

    def test_debug_sessions_history_endpoint(self) -> None:
        response = self.client.get("/debug/sessions/history")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("sessions", payload)
        self.assertIsInstance(payload["sessions"], list)

    def test_latest_snapshot_unknown_session_returns_404(self) -> None:
        response = self.client.get("/debug/sessions/does-not-exist/latest.jpg")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Unknown session")

    def test_latest_snapshot_without_frame_returns_404(self) -> None:
        app_main.sessions["test-session"] = app_main.Session(
            peer_connection=_FakePeerConnection()
        )
        response = self.client.get("/debug/sessions/test-session/latest.jpg")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "No snapshot available yet")

    def test_ice_unknown_session_returns_404(self) -> None:
        response = self.client.post(
            "/webrtc/ice",
            json={
                "session_id": "unknown",
                "candidate": "candidate:abc",
                "sdpMid": "0",
                "sdpMLineIndex": 0,
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Unknown session")

    def test_offer_validation_error_returns_422(self) -> None:
        response = self.client.post("/webrtc/offer", json={"type": "offer"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

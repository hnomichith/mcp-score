"""Tests for the shared Remote Control WebSocket protocol.

Tests the handshake, commands, disconnect, and protocol messages using
DoricoBridge as the concrete subclass. These tests cover protocol logic
shared by all RemoteControlBridge subclasses — they don't need to be
duplicated for each application.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from mcp_score.bridge.dorico import DoricoBridge
from mcp_score.bridge.remote_control import DEFAULT_CLIENT_NAME, HANDSHAKE_VERSION

# ── Handshake protocol ───────────────────────────────────────────────


class TestRemoteControlHandshake:
    @pytest.mark.anyio()
    async def test_fresh_handshake_sends_connect_and_accept_messages(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        mock_connection = AsyncMock()

        session_token_response = json.dumps(
            {"message": "sessiontoken", "sessionToken": "abc-123"}
        )
        accept_response = json.dumps({"message": "response", "code": "kConnected"})
        mock_connection.recv = AsyncMock(
            side_effect=[session_token_response, accept_response]
        )
        mock_connection.send = AsyncMock()

        with patch(
            "mcp_score.bridge.remote_control.websockets.connect",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            # Act
            connected = await bridge.connect()

        # Assert
        assert connected is True
        assert bridge._session_token == "abc-123"
        assert mock_connection.send.call_count == 2

        # Verify connect message format
        connect_msg = json.loads(mock_connection.send.call_args_list[0].args[0])
        assert connect_msg["message"] == "connect"
        assert connect_msg["clientName"] == DEFAULT_CLIENT_NAME
        assert connect_msg["handshakeVersion"] == HANDSHAKE_VERSION

        # Verify accept message format
        accept_msg = json.loads(mock_connection.send.call_args_list[1].args[0])
        assert accept_msg["message"] == "acceptsessiontoken"
        assert accept_msg["sessionToken"] == "abc-123"

    @pytest.mark.anyio()
    async def test_connect_with_cached_token_skips_accept(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge._session_token = "cached-token"
        mock_connection = AsyncMock()

        connected_response = json.dumps({"message": "response", "code": "kConnected"})
        mock_connection.recv = AsyncMock(return_value=connected_response)
        mock_connection.send = AsyncMock()

        with patch(
            "mcp_score.bridge.remote_control.websockets.connect",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            # Act
            connected = await bridge.connect()

        # Assert
        assert connected is True
        assert mock_connection.send.call_count == 1
        connect_msg = json.loads(mock_connection.send.call_args_list[0].args[0])
        assert connect_msg["sessionToken"] == "cached-token"

    @pytest.mark.anyio()
    async def test_connect_with_expired_token_accepts_new_token(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge._session_token = "expired-token"
        mock_connection = AsyncMock()

        new_token_response = json.dumps(
            {"message": "sessiontoken", "sessionToken": "new-token-456"}
        )
        accept_response = json.dumps({"message": "response", "code": "kConnected"})
        mock_connection.recv = AsyncMock(
            side_effect=[new_token_response, accept_response]
        )
        mock_connection.send = AsyncMock()

        with patch(
            "mcp_score.bridge.remote_control.websockets.connect",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            # Act
            connected = await bridge.connect()

        # Assert
        assert connected is True
        assert bridge._session_token == "new-token-456"

    @pytest.mark.anyio()
    async def test_connect_with_expired_token_unexpected_code_fails(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge._session_token = "expired-token"
        mock_connection = AsyncMock()

        new_token_response = json.dumps(
            {"message": "sessiontoken", "sessionToken": "new-token"}
        )
        unexpected_response = json.dumps({"message": "response", "code": "kPending"})
        mock_connection.recv = AsyncMock(
            side_effect=[new_token_response, unexpected_response]
        )
        mock_connection.send = AsyncMock()
        mock_connection.close = AsyncMock()

        with patch(
            "mcp_score.bridge.remote_control.websockets.connect",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            # Act
            connected = await bridge.connect()

        # Assert
        assert connected is False

    @pytest.mark.anyio()
    async def test_connect_with_rejected_handshake_returns_false(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        mock_connection = AsyncMock()

        session_token_response = json.dumps(
            {"message": "sessiontoken", "sessionToken": "abc"}
        )
        error_response = json.dumps(
            {"message": "response", "code": "kError", "detail": "Connection refused"}
        )
        mock_connection.recv = AsyncMock(
            side_effect=[session_token_response, error_response]
        )
        mock_connection.send = AsyncMock()
        mock_connection.close = AsyncMock()

        with patch(
            "mcp_score.bridge.remote_control.websockets.connect",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            # Act
            connected = await bridge.connect()

        # Assert
        assert connected is False

    @pytest.mark.anyio()
    async def test_connect_without_session_token_returns_false(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        mock_connection = AsyncMock()

        bad_response = json.dumps({"message": "sessiontoken"})
        mock_connection.recv = AsyncMock(return_value=bad_response)
        mock_connection.send = AsyncMock()
        mock_connection.close = AsyncMock()

        with patch(
            "mcp_score.bridge.remote_control.websockets.connect",
            new_callable=AsyncMock,
            return_value=mock_connection,
        ):
            # Act
            connected = await bridge.connect()

        # Assert
        assert connected is False


# ── Command execution ────────────────────────────────────────────────


class TestRemoteControlCommands:
    @pytest.mark.anyio()
    async def test_send_command_formats_correct_message(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge._connection = AsyncMock()
        bridge._connection.recv = AsyncMock(
            return_value=json.dumps({"message": "response", "code": "kOK"})
        )

        # Act
        await bridge.send_command("Edit.Undo")

        # Assert
        sent_json = json.loads(bridge._connection.send.call_args.args[0])
        assert sent_json["message"] == "command"
        assert sent_json["commandName"] == "Edit.Undo"
        assert "parameters" not in sent_json

    @pytest.mark.anyio()
    async def test_send_command_with_params_includes_parameters(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge._connection = AsyncMock()
        bridge._connection.recv = AsyncMock(
            return_value=json.dumps({"message": "response", "code": "kOK"})
        )

        # Act
        await bridge.send_command("Edit.GoToBar", {"barNumber": "5"})

        # Assert
        sent_json = json.loads(bridge._connection.send.call_args.args[0])
        assert sent_json["parameters"] == {"barNumber": "5"}

    @pytest.mark.anyio()
    async def test_send_command_without_connection_returns_error(self) -> None:
        # Arrange
        bridge = DoricoBridge(host="localhost", port=19999)

        # Act
        result = await bridge.send_command("Edit.Undo")

        # Assert
        assert "error" in result
        assert "Cannot connect" in result["error"]

    @pytest.mark.anyio()
    async def test_undo_sends_edit_undo_command(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge.send_command = AsyncMock(
            return_value={"message": "response", "code": "kOK"}
        )

        # Act
        await bridge.undo()

        # Assert
        bridge.send_command.assert_called_once_with("Edit.Undo")

    @pytest.mark.anyio()
    async def test_go_to_measure_sends_edit_gotobar_command(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge.send_command = AsyncMock(
            return_value={"message": "response", "code": "kOK"}
        )

        # Act
        await bridge.go_to_measure(5)

        # Assert
        bridge.send_command.assert_called_once_with("Edit.GoToBar", {"barNumber": "5"})

    @pytest.mark.anyio()
    async def test_go_to_staff_returns_unsupported_warning(self) -> None:
        # Arrange
        bridge = DoricoBridge()

        # Act
        result = await bridge.go_to_staff(2)

        # Assert
        assert "warning" in result
        assert "does not support" in result["warning"]


# ── Barline mapping ──────────────────────────────────────────────────


class TestRemoteControlBarlines:
    @pytest.mark.anyio()
    @pytest.mark.parametrize(
        ("barline_type", "expected_command"),
        [
            ("double", "AddBarlineDouble"),
            ("final", "AddBarlineFinal"),
            ("startRepeat", "AddBarlineStartRepeat"),
            ("endRepeat", "AddBarlineEndRepeat"),
        ],
    )
    async def test_set_barline_sends_correct_command(
        self, barline_type: str, expected_command: str
    ) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge.send_command = AsyncMock(
            return_value={"message": "response", "code": "kOK"}
        )

        # Act
        await bridge.set_barline(barline_type)

        # Assert
        bridge.send_command.assert_called_once_with(expected_command)

    @pytest.mark.anyio()
    async def test_set_barline_with_unknown_type_returns_error(self) -> None:
        # Arrange
        bridge = DoricoBridge()

        # Act
        result = await bridge.set_barline("nonexistent")

        # Assert
        assert "error" in result
        assert "Unknown barline type" in result["error"]


# ── Limitations ──────────────────────────────────────────────────────


class TestRemoteControlLimitations:
    @pytest.mark.anyio()
    async def test_set_key_signature_returns_unsupported_error(self) -> None:
        # Arrange
        bridge = DoricoBridge()

        # Act
        result = await bridge.set_key_signature(2)

        # Assert
        assert "error" in result
        assert "does not support" in result["error"]

    @pytest.mark.anyio()
    async def test_set_tempo_returns_unsupported_error(self) -> None:
        # Arrange
        bridge = DoricoBridge()

        # Act
        result = await bridge.set_tempo(120)

        # Assert
        assert "error" in result
        assert "does not support" in result["error"]

    @pytest.mark.anyio()
    async def test_add_rehearsal_mark_returns_auto_numbering_warning(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge.send_command = AsyncMock(
            return_value={"message": "response", "code": "kOK"}
        )

        # Act
        result = await bridge.add_rehearsal_mark("B")

        # Assert
        assert "warning" in result
        assert "B" in result["warning"]

    @pytest.mark.anyio()
    async def test_add_chord_symbol_returns_unsupported_error(self) -> None:
        # Arrange
        bridge = DoricoBridge()

        # Act
        result = await bridge.add_chord_symbol("Cmaj7")

        # Assert
        assert "error" in result
        assert "Cmaj7" in result["error"]


# ── Ping ─────────────────────────────────────────────────────────────


class TestRemoteControlPing:
    @pytest.mark.anyio()
    async def test_ping_with_successful_app_info_returns_true(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge.get_app_info = AsyncMock(
            return_value={"variant": "Dorico Pro", "number": "5.1"}
        )

        # Act
        alive = await bridge.ping()

        # Assert
        assert alive is True

    @pytest.mark.anyio()
    async def test_ping_with_failed_app_info_returns_false(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        bridge.get_app_info = AsyncMock(return_value={"error": "Connection failed"})

        # Act
        alive = await bridge.ping()

        # Assert
        assert alive is False


# ── Disconnect ───────────────────────────────────────────────────────


class TestRemoteControlDisconnect:
    @pytest.mark.anyio()
    async def test_disconnect_sends_message_and_closes_connection(self) -> None:
        # Arrange
        bridge = DoricoBridge()
        mock_connection = AsyncMock()
        bridge._connection = mock_connection

        # Act
        await bridge.disconnect()

        # Assert
        mock_connection.send.assert_called_once()
        sent_msg = json.loads(mock_connection.send.call_args.args[0])
        assert sent_msg["message"] == "disconnect"
        mock_connection.close.assert_called_once()
        assert bridge._connection is None

    @pytest.mark.anyio()
    async def test_disconnect_without_connection_succeeds(self) -> None:
        # Arrange
        bridge = DoricoBridge()

        # Act — should not raise
        await bridge.disconnect()

        # Assert
        assert bridge._connection is None


# ── Error handling ───────────────────────────────────────────────────


class TestRemoteControlErrorHandling:
    @pytest.mark.anyio()
    async def test_malformed_json_response_returns_error(self) -> None:
        """Non-JSON responses should produce a clear error, not crash."""
        # Arrange
        bridge = DoricoBridge()
        mock_connection = AsyncMock()
        bridge._connection = mock_connection

        mock_connection.send = AsyncMock()
        mock_connection.recv = AsyncMock(return_value="not valid json {{{")

        # Act
        result = await bridge._send_and_receive({"message": "getstatus"})

        # Assert
        assert "error" in result
        assert "Invalid JSON" in result["error"]

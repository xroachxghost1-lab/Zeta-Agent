"""
Security Manager — Handles encrypted secrets, API keys, sandbox execution,
permission checks, and workspace isolation.
"""

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from rich.console import Console

from zeta_cli.config.manager import ConfigManager

console = Console()

# Sentinel value for redacted output
REDACTED = "********"

class SecurityManager:
    """
    Manages all security-related functionality.

    Features:
    - API key encryption using Fernet (AES-128-CBC)
    - Master key derivation via PBKDF2
    - Secure secret storage
    - Permission checking
    - Command sandboxing
    - Never prints secrets
    """

    def __init__(self, config: ConfigManager):
        self._config = config
        self._fernet: Optional[Fernet] = None
        self._secrets_path: Path = Path(config.get("system.data_dir")) / "secrets.enc"
        self._master_salt: Optional[bytes] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize security — derive or create master key."""
        self._secrets_path.parent.mkdir(parents=True, exist_ok=True)

        # Get or create master salt
        salt_path = self._secrets_path.parent / ".salt"
        if salt_path.exists():
            self._master_salt = salt_path.read_bytes()
        else:
            self._master_salt = os.urandom(32)
            salt_path.write_bytes(self._master_salt)

        # Derive key from machine-specific data + stored salt
        machine_id = self._get_machine_identifier()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._master_salt,
            iterations=600_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
        self._fernet = Fernet(key)
        self._initialized = True

    async def store_secret(self, key: str, value: str) -> None:
        """
        Encrypt and store a secret.

        Args:
            key: Identifier for the secret
            value: Secret value to encrypt
        """
        if not self._fernet:
            raise RuntimeError("SecurityManager not initialized")

        secrets_data = await self._load_secrets()
        secrets_data[key] = self._fernet.encrypt(value.encode()).decode()
        await self._save_secrets(secrets_data)

    async def get_secret(self, key: str) -> Optional[str]:
        """
        Retrieve and decrypt a secret.

        Args:
            key: Identifier for the secret

        Returns:
            Decrypted secret value or None
        """
        if not self._fernet:
            raise RuntimeError("SecurityManager not initialized")

        secrets_data = await self._load_secrets()
        encrypted = secrets_data.get(key)
        if encrypted:
            try:
                return self._fernet.decrypt(encrypted.encode()).decode()
            except InvalidToken:
                console.print(f"[red]Error: Failed to decrypt secret '{key}'. Data may be corrupted.[/red]")
                return None
        return None

    async def delete_secret(self, key: str) -> bool:
        """
        Delete a stored secret.

        Args:
            key: Identifier for the secret

        Returns:
            True if deleted, False if not found
        """
        secrets_data = await self._load_secrets()
        if key in secrets_data:
            del secrets_data[key]
            await self._save_secrets(secrets_data)
            return True
        return False

    async def list_secrets(self) -> list[str]:
        """List all stored secret keys (never reveals values)."""
        secrets_data = await self._load_secrets()
        return list(secrets_data.keys())

    def get_redacted(self, value: str) -> str:
        """Return a redacted version of a sensitive value."""
        if len(value) <= 8:
            return REDACTED
        return value[:4] + "..." + value[-4:]

    def hash_content(self, content: str) -> str:
        """Create a SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def check_path_permission(self, target_path: Path, workspace: Path) -> bool:
        """
        Check if a path is within the allowed workspace.

        Args:
            target_path: Path to check
            workspace: Allowed workspace root

        Returns:
            True if path is within workspace
        """
        try:
            resolved = target_path.resolve()
            workspace_resolved = workspace.resolve()
            return resolved.is_relative_to(workspace_resolved)
        except (ValueError, OSError, AttributeError):
            try:
                resolved = target_path.resolve()
                workspace_resolved = workspace.resolve()
                return str(resolved) == str(workspace_resolved) or str(resolved).startswith(str(workspace_resolved) + os.sep)
            except Exception:
                return False

    def is_command_safe(self, command: str, blocked: list[str]) -> bool:
        """
        Basic command safety check.

        Args:
            command: Command string to check
            blocked: List of blocked command prefixes

        Returns:
            True if command appears safe
        """
        command_lower = command.lower().strip()
        for blocked_cmd in blocked:
            if command_lower.startswith(blocked_cmd.lower()):
                return False
        return True

    async def _load_secrets(self) -> dict:
        """Load encrypted secrets from file."""
        if self._secrets_path.exists():
            try:
                with open(self._secrets_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    async def _save_secrets(self, data: dict) -> None:
        """Save encrypted secrets to file."""
        self._secrets_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._secrets_path, "w") as f:
            json.dump(data, f)

    def _get_machine_identifier(self) -> str:
        """
        Generate a machine-specific identifier for key derivation.
        Uses multiple system characteristics for entropy.
        """
        import platform
        import socket

        components = [
            platform.node(),
            platform.machine(),
            platform.processor(),
            socket.gethostname(),
            str(Path.home()),
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Clean up security resources."""
        self._fernet = None
        self._initialized = False

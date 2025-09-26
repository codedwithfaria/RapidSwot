"""
Secure file operations with configurable access controls.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

class FileSystem:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or os.getcwd())
        self.access_rules = {}
        
    def set_access_rules(self, rules: Dict[str, List[str]]):
        """
        Set access control rules for paths.
        
        Args:
            rules: Dict mapping paths to allowed operations
        """
        self.access_rules = rules
        
    def read_file(self, path: str) -> str:
        """
        Read file contents securely.
        
        Args:
            path: Path to file relative to root
            
        Returns:
            File contents as string
        """
        full_path = self._resolve_path(path)
        self._check_access(full_path, 'read')
        
        with open(full_path, 'r') as f:
            return f.read()
            
    def write_file(self, path: str, content: str):
        """
        Write content to file securely.
        
        Args:
            path: Path to file relative to root
            content: Content to write
        """
        full_path = self._resolve_path(path)
        self._check_access(full_path, 'write')
        
        # Ensure parent directories exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
            
    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path within root."""
        resolved = (self.root_dir / path).resolve()
        if not str(resolved).startswith(str(self.root_dir)):
            raise ValueError("Path outside root directory")
        return resolved
        
    def _check_access(self, path: Path, operation: str):
        """Check if operation is allowed for path."""
        rel_path = str(path.relative_to(self.root_dir))
        if rel_path in self.access_rules:
            if operation not in self.access_rules[rel_path]:
                raise PermissionError(
                    f"Operation {operation} not allowed for {rel_path}"
                )
"""
Git integration tools for repository operations.
"""
import git
from typing import Dict, List, Optional
from pathlib import Path

class GitTools:
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path
        self._repo = None
        
    @property
    def repo(self) -> git.Repo:
        """Get or initialize git repository."""
        if not self._repo:
            if self.repo_path:
                self._repo = git.Repo(self.repo_path)
            else:
                self._repo = git.Repo()
        return self._repo
        
    def search_content(self, query: str) -> List[Dict[str, str]]:
        """
        Search repository content.
        
        Args:
            query: Search query
            
        Returns:
            List of matches with file paths and content
        """
        results = []
        for blob in self.repo.head.commit.tree.traverse():
            if blob.type != 'blob':
                continue
            content = blob.data_stream.read().decode('utf-8')
            if query.lower() in content.lower():
                results.append({
                    'path': blob.path,
                    'content': content
                })
        return results
        
    def get_file_history(self, file_path: str) -> List[Dict[str, str]]:
        """Get commit history for a specific file."""
        history = []
        for commit in self.repo.iter_commits(paths=file_path):
            history.append({
                'hash': commit.hexsha,
                'message': commit.message,
                'author': commit.author.name,
                'date': commit.authored_datetime.isoformat()
            })
        return history
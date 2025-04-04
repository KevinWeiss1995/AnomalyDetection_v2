import subprocess
import os

def get_git_repo_root():
    """
    Gets the root directory of the git repo, falls back to current dir if not in a repo.
    
    Returns:
        String path to repository root or current directory if not in a git repo
    """
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return os.getcwd()
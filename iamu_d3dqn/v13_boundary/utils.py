"""Supplement existing V13 provenance with SHA256 of this additive package."""
from pathlib import Path
from v13_diagnostics.common import provenance,digest

def boundary_provenance(args,paths=None,inputs=None):
    result=provenance(args,paths,inputs)
    root=Path(__file__).resolve().parent
    result['boundary_code_sha256']={str(p.relative_to(root)):digest(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.sh')}
    return result

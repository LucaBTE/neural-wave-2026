#!/usr/bin/env python3
"""Verify or restore a saved pipeline version, changing files only inside dev."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile

DEV=Path(__file__).resolve().parent
ROOT=DEV.parent


def load_version(version):
    if not re.fullmatch(r'\d+\.\d+(?:\.\d+)?',version):
        raise ValueError('Expected a version such as 0.1')
    folder=DEV/'versions'/version
    record=json.loads((folder/'manifest.json').read_text())
    archive=folder/'source.tar.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['archive_sha256']:
        raise ValueError('Version archive checksum mismatch')
    payloads={}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len({m.name for m in members})!=len(members):
            raise ValueError('Duplicate archive paths')
        if {m.name for m in members}!=set(record['files']):
            raise ValueError('Archive and manifest file lists differ')
        for member in members:
            path=Path(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or not path.parts or path.parts[0]!='dev'):
                raise ValueError('Unsafe version archive entry')
            if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
                raise ValueError('Version archives must not replace local secrets')
            destination=(ROOT/path).resolve()
            if not destination.is_relative_to(DEV):
                raise ValueError('Version file resolves outside dev')
            data=tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=record['files'][member.name]:
                raise ValueError('Version file checksum mismatch: '+member.name)
            payloads[member.name]=data
    return record,payloads


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','restore'])
    parser.add_argument('version')
    args=parser.parse_args(argv)
    record,payloads=load_version(args.version)
    if args.action=='verify':
        print(f"Version {record['version']}: {len(payloads)} files verified; no files changed.")
        return 0
    backup=DEV/'.cache/version-restores'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True,exist_ok=False)
    # Preserve all existing files before changing any of them; never touch .env.
    for name in payloads:
        dest=ROOT/name
        if dest.exists():
            prior=backup/name;prior.parent.mkdir(parents=True,exist_ok=True)
            prior.write_bytes(dest.read_bytes())
    for name,data in payloads.items():
        dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print(f"Restored pipeline {record['version']}. Restart the backend. Previous files: {backup}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())

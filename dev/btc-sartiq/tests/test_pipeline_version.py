"""A rollback must verify the complete archive before replacing local code."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

spec=importlib.util.spec_from_file_location('pipeline_version',Path(__file__).resolve().parents[2]/'pipeline_version.py')
version=importlib.util.module_from_spec(spec);spec.loader.exec_module(version)


@pytest.fixture
def archive(tmp_path,monkeypatch):
    dev=tmp_path/'dev';folder=dev/'versions/0.1';folder.mkdir(parents=True)
    monkeypatch.setattr(version,'DEV',dev);monkeypatch.setattr(version,'ROOT',tmp_path)
    def create(files):
        path=folder/'source.tar.gz'
        with tarfile.open(path,'w:gz') as tar:
            for name,data in files.items():
                entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
        (folder/'manifest.json').write_text(json.dumps({'version':'0.1','archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}))
        return path
    return dev,create


def test_restore_recovers_saved_code_and_keeps_secret_and_backup(archive):
    dev,create=archive;create({'dev/backend/pipeline.py':b'saved code\n'})
    backend=dev/'backend';backend.mkdir();(backend/'pipeline.py').write_bytes(b'new code\n');(backend/'.env').write_bytes(b'local secret\n')
    assert version.main(['restore','0.1'])==0
    assert (backend/'pipeline.py').read_bytes()==b'saved code\n'
    assert (backend/'.env').read_bytes()==b'local secret\n'
    backups=list((dev/'.cache/version-restores').glob('*/dev/backend/pipeline.py'))
    assert len(backups)==1 and backups[0].read_bytes()==b'new code\n'


def test_corrupt_snapshot_cannot_overwrite_code(archive):
    dev,create=archive;path=create({'dev/pipeline.py':b'saved'})
    current=dev/'pipeline.py';current.write_bytes(b'keep me')
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        version.main(['restore','0.1'])
    assert current.read_bytes()==b'keep me'
    assert not (dev/'.cache/version-restores').exists()


def test_even_a_validly_hashed_archive_cannot_replace_local_env(archive):
    dev,create=archive;create({'dev/backend/.env':b'replacement'})
    with pytest.raises(ValueError,match='local secrets'):
        version.main(['restore','0.1'])
    assert not (dev/'.cache/version-restores').exists()

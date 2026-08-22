from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import PurePosixPath
from typing import Callable

from boto3.s3.transfer import TransferConfig


class RcloneNotAvailable(RuntimeError):
    pass


class BoxRemoteNotConfigured(RuntimeError):
    pass


def find_rclone() -> str:
    path = shutil.which('rclone')
    if not path:
        raise RcloneNotAvailable(
            'rclone is not installed on this compute worker. Install it with `sudo apt install -y rclone`.'
        )
    return path


def list_rclone_remotes() -> list[str]:
    rclone = find_rclone()
    proc = subprocess.run([rclone, 'listremotes'], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or 'Could not read rclone remotes.')
    return [line.strip().rstrip(':') for line in proc.stdout.splitlines() if line.strip()]


def _remote_spec(remote: str, path: str = '') -> str:
    remote = str(remote or '').strip().rstrip(':')
    if not remote:
        raise ValueError('Box rclone remote name is required.')
    path = str(path or '').strip().strip('/')
    return f'{remote}:{path}' if path else f'{remote}:'


def list_box_nd2(remote: str = 'box', folder: str = '') -> list[dict]:
    rclone = find_rclone()
    remotes = list_rclone_remotes()
    if remote.rstrip(':') not in remotes:
        raise BoxRemoteNotConfigured(
            f'Rclone remote `{remote.rstrip(":")}:` is not configured on this worker. Run `rclone config` once to connect Box.'
        )

    cmd = [
        rclone,
        'lsjson',
        _remote_spec(remote, folder),
        '--recursive',
        '--files-only',
        '--no-mimetype',
        '--no-modtime',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or 'Could not list Box files through rclone.')

    rows = []
    for item in json.loads(proc.stdout or '[]'):
        path = str(item.get('Path', '') or '')
        if not path.lower().endswith('.nd2'):
            continue
        rows.append({
            'path': path,
            'name': PurePosixPath(path).name,
            'size': int(item.get('Size', 0) or 0),
        })
    rows.sort(key=lambda r: r['path'].lower())
    return rows


def transfer_box_file_to_s3(
    s3_client,
    *,
    remote: str,
    folder: str,
    box_path: str,
    expected_size: int,
    bucket: str,
    destination_key: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Stream one Box file through rclone directly into S3.

    The file is not first downloaded to local disk. S3 upload is sequential so the
    rclone stdout stream does not need to be seekable. The final object size is
    verified and any incomplete destination object is deleted on failure.
    """
    rclone = find_rclone()
    folder = str(folder or '').strip().strip('/')
    box_path = str(box_path or '').strip().strip('/')
    source_path = '/'.join(part for part in (folder, box_path) if part)
    source = _remote_spec(remote, source_path)
    destination_key = str(destination_key or '').strip().lstrip('/')
    if not bucket or not destination_key:
        raise ValueError('S3 bucket and destination key are required.')

    transferred = 0

    def _progress(delta: int):
        nonlocal transferred
        transferred += int(delta)
        if progress_callback:
            progress_callback(min(transferred, expected_size), expected_size)

    config = TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
    )

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            [rclone, 'cat', source],
            stdout=subprocess.PIPE,
            stderr=err,
        )
        try:
            if proc.stdout is None:
                raise RuntimeError('rclone did not provide a readable Box stream.')
            s3_client.upload_fileobj(
                proc.stdout,
                bucket,
                destination_key,
                Callback=_progress,
                Config=config,
            )
            proc.stdout.close()
            return_code = proc.wait()
            if return_code != 0:
                err.seek(0)
                message = err.read().decode('utf-8', errors='replace').strip()
                raise RuntimeError(message or f'rclone exited with code {return_code}.')

            head = s3_client.head_object(Bucket=bucket, Key=destination_key)
            actual_size = int(head.get('ContentLength', 0) or 0)
            if expected_size >= 0 and actual_size != int(expected_size):
                raise IOError(
                    f'Transfer size mismatch: Box reported {expected_size} bytes but S3 contains {actual_size} bytes.'
                )
            if progress_callback:
                progress_callback(actual_size, actual_size)
            return {
                'box_source': source,
                'bucket': bucket,
                'key': destination_key,
                'size': actual_size,
                's3_uri': f's3://{bucket}/{destination_key}',
            }
        except Exception:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            try:
                s3_client.delete_object(Bucket=bucket, Key=destination_key)
            except Exception:
                pass
            raise

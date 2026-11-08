"""Atomic parquet writes for the caches the gate, dn_tactical and the Schwab/byz readers open from disk.

Its own module (no load_dotenv, yfinance or warnings filter at import) so data_fetcher and the VX builder share it."""
import os
import secrets

TEMP_PREFIX, TEMP_SUFFIX = ".", ".tmp"  # .gitignore: .*.parquet.*.tmp


def write_parquet_atomic(df, path, **to_parquet_kwargs):
    """Write df to path so that no reader ever sees a half-written file: a temp file in path's own directory, then
    os.replace. The temp name is unique per call (O_EXCL, random token), because the launchd end-session and a manual
    gate-refresh can write the same parquet at once. The new file keeps the old file's mode, else 0o666 minus the umask
    (the kernel applies it at create). The temp is removed on any exception, KeyboardInterrupt included."""
    path = os.fspath(path)
    directory, name = os.path.split(path)
    for _ in range(100):
        tmp = os.path.join(directory, f"{TEMP_PREFIX}{name}.{secrets.token_hex(6)}{TEMP_SUFFIX}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"no free temp name for {path} after 100 tries")
    try:
        os.close(fd)
        df.to_parquet(tmp, **to_parquet_kwargs)
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except FileNotFoundError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

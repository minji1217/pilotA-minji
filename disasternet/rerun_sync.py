"""PC에서 다시 돌린 결과를 Google Drive와 GitHub(pc-results 브랜치)로 보낸다. run_rerun_all.bat이 부른다.

bat 파일 안에 한글 경로(G:\\내 드라이브)를 두면 cmd가 줄을 잘못 읽어 멈추는 일이 있어서, 복사 · 업로드는 여기서 한다.

Google Drive : <드라이브>\\pilotA_rerun\\muni\\<실험>\\ (시정촌 결과 전부), ...\\cells\\<실험>\\ (칸 파라미터 rowparams_*.csv.gz 와
               칸 예측 cells_*.csv.gz 는 커서 빼고), rerun_log.txt
GitHub       : <git 폴더>\\pc_rerun\\ 에 Drive와 같은 파일을 복사한 뒤 commit, pc-results 브랜치로 push
"""
import argparse
import fnmatch
import shutil
import subprocess
from pathlib import Path

SKIP_ALWAYS = ["ckpt_*.pt", "*.tmp"]
SKIP_CELLS = ["rowparams_*.csv.gz", "cells_*.csv.gz"]


def _copy_tree(src: Path, dst: Path, skip):
    n = 0
    if not src.exists():
        return 0
    for f in src.rglob("*"):
        if not f.is_file() or any(fnmatch.fnmatch(f.name, p) for p in skip):
            continue
        t = dst / f.relative_to(src)
        if t.exists() and t.stat().st_size == f.stat().st_size and t.stat().st_mtime >= f.stat().st_mtime:
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
        n += 1
    return n


def _sync_to(dest: Path, root: Path, log: Path):
    a = _copy_tree(root / "rerun_muni", dest / "muni", SKIP_ALWAYS)
    b = _copy_tree(root / "rerun_cells", dest / "cells", SKIP_ALWAYS + SKIP_CELLS)
    if log.exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, dest / log.name)
    return a + b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/disasternet")
    ap.add_argument("--git", default=r"C:\work\pilotA-minji-git")
    a = ap.parse_args(argv)
    root = Path(a.root)
    log = root / "rerun_log.txt"

    drive = next((p for p in (Path("G:/내 드라이브"), Path("G:/My Drive")) if p.exists()), None)
    if drive is None:
        print("  Google Drive(G:) 없음: Drive 복사 건너뜀", flush=True)
    else:
        try:
            n = _sync_to(drive / "pilotA_rerun", root, log)
            print(f"  Google Drive 복사 {n}개: {drive / 'pilotA_rerun'}", flush=True)
        except Exception as e:   # Drive가 잠깐 안 돼도 실험은 계속
            print(f"  Google Drive 복사 실패: {e}", flush=True)

    g = Path(a.git)
    if not (g / ".git").exists():
        print(f"  git 폴더 없음({g}): GitHub 업로드 건너뜀", flush=True)
        return 0
    try:
        n = _sync_to(g / "pc_rerun", root, log)
        run = lambda *c: subprocess.run(["git", *c], cwd=g, capture_output=True, text=True, encoding="utf-8", errors="replace")
        run("add", "pc_rerun")
        if run("diff", "--cached", "--quiet").returncode == 0:
            print("  GitHub: 새로 올릴 것 없음", flush=True)
            return 0
        run("commit", "-q", "-m", "[Results] PC 다시 돌리기: 파라미터 · loss")
        r = run("push", "-q", "-f", "origin", "HEAD:refs/heads/pc-results")
        print(f"  GitHub 업로드 {'완료' if r.returncode == 0 else '실패: ' + r.stderr.strip()[:200]} (파일 {n}개 갱신)", flush=True)
    except Exception as e:
        print(f"  GitHub 업로드 실패: {e}", flush=True)
    return 0


if __name__ == "__main__":
    main()

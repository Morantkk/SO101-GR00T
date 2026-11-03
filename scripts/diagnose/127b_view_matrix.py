"""127b 视角矩阵：把若干 (标签, 文件, 帧号) 两两做相位相关，用来判断相机是否被挪动过。"""
import itertools, sys
import cv2
import numpy as np

DS = "E:/lerobot_datasets/arm101_pick_all_objects_trim/videos/chunk-000/observation.images.front/file-000.mp4"
ITEMS = [
    ("ds#0", DS, 0),
    ("ds#5", DS, 5),
    ("ds#60", DS, 60),
    ("r15#0", "logs/v3_off10_193533.mp4", 0),
    ("r20#0", "logs/v3_off10_run_131721.mp4", 0),
    ("r20b#0", "logs/v3_off10_run_131650.mp4", 0),
]


def read(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"读不到 {path}#{idx}")
    return fr


def prep(img):
    g = cv2.cvtColor(cv2.resize(img, (640, 360)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g -= g.mean()
    h, w = g.shape
    return g * np.hanning(h)[:, None] * np.hanning(w)[None, :]


def main():
    imgs = {}
    for lab, p, i in ITEMS:
        try:
            imgs[lab] = prep(read(p, i))
        except SystemExit as e:
            print("skip", e)
    print(f"{'pair':<22}{'shift(dx,dy)':>22}{'resp':>8}{'ncc':>8}")
    print("-" * 60)
    for a, b in itertools.combinations(imgs, 2):
        (dx, dy), resp = cv2.phaseCorrelate(imgs[a], imgs[b])
        ncc = float(np.corrcoef(imgs[a].ravel(), imgs[b].ravel())[0, 1])
        print(f"{a+' vs '+b:<22}{f'({dx:+7.2f},{dy:+7.2f})':>22}{resp:>8.3f}{ncc:>+8.3f}")


if __name__ == "__main__":
    sys.exit(main())

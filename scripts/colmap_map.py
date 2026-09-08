"""pycolmap glue (vendored from slam-exp colmap.py, pycolmap 4.1.1): build a
SIMPLE_RADIAL database from our features + matches, two-view-verify, bootstrap
+ incremental SfM, return each 3D point's (image_id, feature_idx) observations.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pycolmap


def build_database(
    db_path: str,
    images: List[dict],
    features: Dict[int, np.ndarray],
    matches: List[Tuple[int, int, np.ndarray]],
    camera_model: str = "SIMPLE_RADIAL",
    camera_params: Optional[np.ndarray] = None,
    single_camera: bool = True,
) -> Dict[int, int]:
    """Fill a fresh COLMAP database: SIMPLE_RADIAL camera (rough focal guess,
    refined during BA) + images, keypoints, raw matches. Returns a dict mapping
    your image_id -> the id COLMAP assigned."""
    if os.path.exists(db_path):
        os.remove(db_path)
    # Database.open(path) creates the schema if the path doesn't exist yet --
    # no separate create_tables() call needed. (pycolmap >= 4.x style; older
    # releases used `pycolmap.Database(path)` directly.)
    db = pycolmap.Database.open(db_path)

    id_map: Dict[int, int] = {}
    shared_camera_id = None

    for img in images:
        w, h = img["width"], img["height"]

        if camera_params is not None:
            params = np.asarray(camera_params, dtype=np.float64)
        else:
            f = 1.2 * max(w, h)  # rough guess; refined during bundle adjustment
            params = np.array([f, w / 2.0, h / 2.0, 0.0], dtype=np.float64)

        if single_camera and shared_camera_id is not None:
            camera_id = shared_camera_id
        else:
            camera = pycolmap.Camera(
                model=camera_model, width=w, height=h, params=params,
            )
            camera_id = db.write_camera(camera)
            if single_camera:
                shared_camera_id = camera_id

        image = pycolmap.Image(name=img["name"], camera_id=camera_id)
        colmap_image_id = db.write_image(image)
        id_map[img["image_id"]] = colmap_image_id

        kps = np.asarray(features[img["image_id"]], dtype=np.float64)
        assert kps.ndim == 2 and kps.shape[1] in (2, 4, 6), \
            "keypoints must be Nx2 (x,y) or Nx4/6 (x,y,scale,orientation,...)"
        db.write_keypoints(colmap_image_id, kps)

    for (a, b, pair_matches) in matches:
        ca, cb = id_map[a], id_map[b]
        pm = np.asarray(pair_matches, dtype=np.uint32)
        if pm.size == 0:
            continue
        db.write_matches(ca, cb, pm)

    db.close()
    return id_map


def run_reconstruction(
    db_path: str,
    image_dir: str,
    output_dir: str,
    id_map: Dict[int, int],
    features: Dict[int, np.ndarray],
    matches: List[Tuple[int, int, np.ndarray]],
    max_reproj_error: float = 4.0,
    min_num_inliers: int = 3,
    stats: Optional[dict] = None,
) -> Dict[int, "pycolmap.Reconstruction"]:
    """Two-view-verify our raw matches (COLMAP's own verifier assumes SIFT), then
    bootstrap the initial pair and run incremental SfM from it. Returns
    {reconstruction_index: Reconstruction}; fills `stats` if given."""
    os.makedirs(output_dir, exist_ok=True)

    db = pycolmap.Database.open(db_path)
    tvg_opts = pycolmap.TwoViewGeometryOptions()
    tvg_opts.min_num_inliers = min_num_inliers
    tvg_opts.detect_watermark = False
    tvg_opts.ransac.random_seed = 0  # deterministic verification

    verified_pairs = 0
    for (a, b, pair_matches) in matches:
        ca, cb = id_map[a], id_map[b]
        if len(pair_matches) < min_num_inliers:
            db.delete_matches(ca, cb)
            continue
        cam_a = db.read_camera(db.read_image(ca).camera_id)
        cam_b = db.read_camera(db.read_image(cb).camera_id)
        # NOTE: points arrays must be the FULL keypoint lists of each image --
        # the estimator indexes them with the match's (idx_a, idx_b) pairs.
        pts_a = np.asarray(features[a], dtype=np.float64)
        pts_b = np.asarray(features[b], dtype=np.float64)
        tvg = pycolmap.estimate_two_view_geometry(
            cam_a, pts_a, cam_b, pts_b,
            matches=np.asarray(pair_matches, dtype=np.uint32),
            options=tvg_opts,
        )
        if (tvg.config == pycolmap.TwoViewGeometryConfiguration.DEGENERATE
                or len(tvg.inlier_matches) < min_num_inliers):
            db.delete_matches(ca, cb)
            continue
        inliers = np.asarray(tvg.inlier_matches, dtype=np.uint32)
        db.delete_matches(ca, cb)
        db.write_matches(ca, cb, inliers)
        db.write_two_view_geometry(ca, cb, tvg)
        verified_pairs += 1

    if verified_pairs == 0:
        db.close()
        raise RuntimeError(
            "geometric verification rejected every image pair -- no verified "
            "matches to reconstruct from. Check feature/match quality."
        )
    if stats is not None:
        stats["pairs_verified"] = verified_pairs
    print(f"geometrically verified {verified_pairs}/{len(matches)} image pairs")

    # --- bootstrap the initial pair ourselves: pick the verified pair with the
    # most inliers that passes the mapper's sanity checks (forward motion, tri
    # angle). COLMAP pair ids encode as min * (2^31 - 1) + max. ---
    k_max_images = 2147483647
    cands = []
    est_opts = pycolmap.TwoViewGeometryOptions()
    est_opts.min_num_inliers = 2
    est_opts.detect_watermark = False
    est_opts.ransac.random_seed = 0
    est_opts.ransac.min_num_trials = 30
    est_opts.ransac.max_error = 4.0
    ids, tvgs = db.read_two_view_geometries()
    for pid, tvg in zip(ids, tvgs):
        i1, i2 = pid // k_max_images, pid % k_max_images
        if len(tvg.inlier_matches) < 4:
            continue
        cam1 = db.read_camera(db.read_image(i1).camera_id)
        cam2 = db.read_camera(db.read_image(i2).camera_id)
        p1 = db.read_keypoints(i1).astype(np.float64)
        p2 = db.read_keypoints(i2).astype(np.float64)
        t = pycolmap.estimate_calibrated_two_view_geometry(
            cam1, p1, cam2, p2,
            matches=np.asarray(tvg.inlier_matches, dtype=np.uint32),
            options=est_opts,
        )
        ok_pose = pycolmap.estimate_two_view_geometry_pose(cam1, p1, cam2, p2, t)
        if (ok_pose and len(t.inlier_matches) >= 4
                and abs(t.cam2_from_cam1.translation[2]) < 0.95
                and np.degrees(t.tri_angle) > 2.0):
            cands.append((len(t.inlier_matches), i1, i2, t))
    db.close()
    if not cands:
        raise RuntimeError(
            "no verified pair passed the two-view pose sanity checks -- "
            "cannot bootstrap the reconstruction."
        )
    _, i1, i2, two_view = max(cands, key=lambda c: c[0])

    recon = pycolmap.Reconstruction()
    db = pycolmap.Database.open(db_path)
    for iid, pose in ((i1, pycolmap.Rigid3d()), (i2, two_view.cam2_from_cam1)):
        cam = db.read_camera(db.read_image(iid).camera_id)
        if not recon.exists_camera(cam.camera_id):
            recon.add_camera_with_trivial_rig(cam)
        im = pycolmap.Image(name=db.read_image(iid).name, camera_id=cam.camera_id)
        im.image_id = iid
        recon.add_image_with_trivial_frame(im, pose)
    db.close()

    bootstrap_dir = str(Path(output_dir) / "bootstrap")
    os.makedirs(bootstrap_dir, exist_ok=True)
    tri_options = pycolmap.IncrementalPipelineOptions()
    tri_options.min_num_matches = min_num_inliers
    tri_options.triangulation.ignore_two_view_tracks = False
    recon = pycolmap.triangulate_points(
        recon, db_path, image_dir, bootstrap_dir,
        clear_points=True, options=tri_options, refine_intrinsics=False,
    )
    recon.write(bootstrap_dir)
    if stats is not None:
        stats["bootstrap_pair"] = (i1, i2)
        stats["bootstrap_images"] = recon.num_images()
        stats["bootstrap_points"] = recon.num_points3D()
    print(f"bootstrapped from pair {i1}-{i2}: {recon.num_images()} images, "
          f"{recon.num_points3D()} points")

    # --- continue incremental SfM from the bootstrap ------------------------
    options = pycolmap.IncrementalPipelineOptions()
    options.mapper.filter_max_reproj_error = max_reproj_error
    # Relax SIFT-scale defaults for our low match counts; init_min_num_inliers
    # must stay >= 4 (pycolmap 4.1.1 maps 1-3 to 0 and then aborts).
    options.min_num_matches = min_num_inliers
    options.min_model_size = 2
    options.mapper.init_min_num_inliers = max(4, min_num_inliers)
    options.mapper.abs_pose_min_num_inliers = min_num_inliers
    options.mapper.init_min_tri_angle = 4.0  # small baselines: relax 16 deg
    options.mapper.random_seed = 0
    # Two-view tracks are the norm at this scale; don't skip triangulating them.
    options.triangulation.ignore_two_view_tracks = False

    # Returns {reconstruction_index: pycolmap.Reconstruction}; more than one
    # entry means the images didn't all merge into a single connected model.
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=output_dir,
        options=options,
        input_path=bootstrap_dir,
    )
    if not reconstructions:
        raise RuntimeError(
            "incremental_mapping produced no reconstruction -- check that "
            "verified matches actually exist (inspect two_view_geometries) "
            "and that images register."
        )
    if stats is not None:
        stats["reconstructions"] = {
            idx: (r.num_images(), r.num_points3D())
            for idx, r in reconstructions.items()
        }
    return reconstructions


def reconstruct_feature_positions(
    images: List[dict],
    features: Dict[int, np.ndarray],
    matches: List[Tuple[int, int, np.ndarray]],
    image_dir: str,
    workdir: str = "colmap_work",
    camera_model: str = "SIMPLE_RADIAL",
    camera_params: Optional[np.ndarray] = None,
    single_camera: bool = True,
    min_num_inliers: int = 3,
    stats: Optional[dict] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Top level: build the DB, reconstruct, and map each triangulated 3D point
    back to its (image_id, feature_idx) observations (features that COLMAP
    couldn't triangulate are simply absent). Fills `stats` if given."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = str(workdir / "database.db")
    sparse_dir = str(workdir / "sparse")

    id_map = build_database(
        db_path, images, features, matches,
        camera_model=camera_model, camera_params=camera_params,
        single_camera=single_camera,
    )
    if stats is not None:
        stats["camera_initial"] = np.asarray(
            camera_params if camera_params is not None else [1.2 * max(images[0]["width"], images[0]["height"]), images[0]["width"] / 2.0, images[0]["height"] / 2.0, 0.0],
            dtype=np.float64,
        )

    reconstructions = run_reconstruction(
        db_path, image_dir, sparse_dir, id_map, features, matches,
        max_reproj_error=4.0, min_num_inliers=min_num_inliers,
        stats=stats,
    )

    inv_id_map = {v: k for k, v in id_map.items()}
    result: Dict[Tuple[int, int], np.ndarray] = {}

    for recon in reconstructions.values():
        for point3D in recon.points3D.values():
            xyz = np.array(point3D.xyz)
            for el in point3D.track.elements:
                orig_image_id = inv_id_map[el.image_id]
                result[(orig_image_id, el.point2D_idx)] = xyz

    return result


def load_images_features_matches(
    image_path: str,
    features_path: str,
    matches_csv: str,
    expected_size: Optional[Tuple[int, int]] = (640, 480),
):
    """Load images (dims read from disk), per-image feature CSVs and the matches
    CSV into the (images, features, matches) shapes reconstruct_feature_positions
    expects. expected_size=(w, h) cross-checks every image's dims."""
    from PIL import Image
    import csv as csv_mod

    image_path = Path(image_path)
    features_path = Path(features_path)

    image_files = sorted([
        p for p in image_path.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    ])
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_path}")

    # stable id assignment: filename stem -> image_id, needed to translate
    # the matches CSV (which references images by name) into the
    # (image_id_a, image_id_b, idx pairs) format build_database expects.
    stem_to_id = {p.stem: i for i, p in enumerate(image_files)}

    images = []
    mismatches = []
    for p in image_files:
        with Image.open(p) as im:
            w, h = im.size  # PIL gives (width, height)
        if expected_size is not None and (w, h) != tuple(expected_size):
            mismatches.append((p.name, w, h))
        images.append({
            "image_id": stem_to_id[p.stem], "name": p.name,
            "width": w, "height": h,
        })

    if mismatches:
        ew, eh = expected_size
        details = ", ".join(f"{name} is {w}x{h}" for name, w, h in mismatches)
        raise ValueError(
            f"Expected all images to be {ew}x{eh} but found "
            f"{len(mismatches)} that aren't: {details}. Pass expected_size=None to "
            f"reconstruct_feature_positions/load_images_features_matches to skip this "
            f"check if that's intentional -- per-image width/height are read from disk "
            f"either way and used correctly regardless."
        )

    features: Dict[int, np.ndarray] = {}
    for p in image_files:
        csv_path = features_path / f"{p.stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing feature CSV for {p.name}: expected {csv_path}")
        # parse only the x,y columns: np.loadtxt would choke on extra non-numeric
        # columns (e.g. the hex descriptor), so use the csv module instead
        with open(csv_path) as fcsv:
            csv_rows = [[c.strip() for c in r] for r in csv_mod.reader(fcsv) if r]
        if not csv_rows:
            raise ValueError(f"Empty feature CSV: {csv_path}")
        xy = np.array([[float(r[0]), float(r[1])] for r in csv_rows], dtype=np.float64)
        features[stem_to_id[p.stem]] = xy

    # --- matches CSV: columns image_a, image_b, idx_a, idx_b ---
    # matches.csv has no header row, but tolerate one if present: it's a header
    # iff the idx columns aren't integers.
    with open(matches_csv, newline="") as f:
        raw = [r for r in csv_mod.reader(f) if r]
    if not raw:
        raise ValueError(f"matches CSV is empty: {matches_csv}")
    try:
        int(raw[0][2]); int(raw[0][3])
        headerless = True
    except (ValueError, IndexError):
        headerless = False
    rows = raw[1:] if not headerless else raw

    pair_rows: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r in rows:
        if len(r) < 4:
            continue
        name_a, name_b = Path(r[0]).stem, Path(r[1]).stem
        id_a, id_b = stem_to_id[name_a], stem_to_id[name_b]
        pair_rows.setdefault((id_a, id_b), []).append((int(r[2]), int(r[3])))

    matches = [
        (a, b, np.array(pairs, dtype=np.uint32))
        for (a, b), pairs in pair_rows.items()
    ]

    return images, features, matches


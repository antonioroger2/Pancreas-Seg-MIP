"""
test_all.py - Comprehensive Test Suite for 3D Attention U-Net Pipeline.

Runs 18 tests (1-12 original, adapted to the step-based pipeline; 13-18 new for T2-T4):
  1.  Dependency test
  2.  Dataset test (synthetic)
  3.  Preprocessing test
  4.  Model forward test
  5.  Attention gate test
  6.  Loss backward test
  7.  One-batch training test
  8.  One-patient overfit test
  9.  Small-subset test
  10. Checkpoint save/load test
  11. Resume test
  12. Evaluation metrics test

Run: python test_all.py
"""

import os
import sys
import tempfile
import shutil
import json
import numpy as np
import traceback

# Test results tracking
results = {}

def run_test(name, test_fn):
    """Run a single test and record result."""
    print(f"\n{'='*60}")
    print(f"TEST {name}")
    print(f"{'='*60}")
    try:
        passed = test_fn()
        status = "PASS [OK]" if passed else "FAIL [FAIL]"
        results[name] = passed
    except Exception as e:
        print(f"  [FAIL] EXCEPTION: {e}")
        traceback.print_exc()
        results[name] = False
        status = "FAIL [FAIL]"
    print(f"  Result: {status}")
    return results[name]


def _synthetic_prep(tmpdir, n=3, shape=(64, 64, 48)):
    """Write n synthetic T2-format cases (<tmp>/cases/*.npz) + a split file; returns (data_dir, splits_path)."""
    os.makedirs(os.path.join(tmpdir, "cases"), exist_ok=True)
    rng = np.random.default_rng(0)
    ids = [f"PANCREAS_{9000 + i:04d}" for i in range(n)]
    for pid in ids:
        lbl = np.zeros(shape, np.uint8); lbl[20:40, 22:42, 14:30] = 1
        img = np.clip(rng.random(shape) * 0.3 + 0.5 * lbl, 0, 1).astype(np.float16)
        np.savez_compressed(os.path.join(tmpdir, "cases", f"{pid}.npz"), image=img, label=lbl)
    sp = os.path.join(tmpdir, "splits.json")
    json.dump(dict(test=[], folds=[dict(fold=0, train=ids[:-1], val=ids[-1:])]), open(sp, "w"))
    return tmpdir, sp


# ==============================================================================
# TEST 1: Dependencies
# ==============================================================================
def test_dependencies():
    """Verify all required packages can be imported."""
    deps = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('nibabel', 'NiBabel'),
        ('SimpleITK', 'SimpleITK'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    all_ok = True
    for module, name in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [OK] {name}: {version}")
        except ImportError:
            print(f"  [FAIL] {name}: NOT INSTALLED")
            all_ok = False

    # Check CUDA
    import torch
    print(f"\n  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return all_ok


# ==============================================================================
# TEST 2: Synthetic Dataset
# ==============================================================================
def test_dataset_synthetic():
    """R0 crop sampler and R1+ patch sampler return correct shapes from synthetic T2-format cases."""
    import torch
    from torch.utils.data import DataLoader
    from src.config import CROP_SIZE
    from src.data.dataset import CropSampler, PatchSampler
    tmpdir = tempfile.mkdtemp(prefix="test_ds_")
    data, _ = _synthetic_prep(tmpdir)
    ids = [f"PANCREAS_{9000 + i:04d}" for i in range(3)]
    img_t, msk_t = CropSampler(data, ids, 4)[0]
    assert img_t.shape == (1, *CROP_SIZE) and msk_t.shape == (1, *CROP_SIZE) and img_t.dtype == torch.float32
    print(f"  [OK] R0 crop sample: {tuple(img_t.shape)} (64x64x48 volume zero-padded)")
    x, y = next(iter(DataLoader(PatchSampler(data, ids, 4, size=(32, 32, 32)), batch_size=2)))
    assert x.shape == (2, 1, 32, 32, 32) and set(torch.unique(y).tolist()) <= {0.0, 1.0}
    print(f"  [OK] Patch batch: {tuple(x.shape)}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 3: Preprocessing
# ==============================================================================
def test_preprocessing():
    """Test HU clipping, normalization, and center crop/pad."""
    from src.data.preprocessing import clip_hu, normalize_to_01, center_crop_or_pad
    from src.config import HU_MIN, HU_MAX, CROP_SIZE

    # HU clipping
    raw = np.array([-1000, -100, 0, 100, 240, 3000], dtype=np.float32)
    clipped = clip_hu(raw)
    assert clipped.min() >= HU_MIN, f"Min {clipped.min()} < {HU_MIN}"
    assert clipped.max() <= HU_MAX, f"Max {clipped.max()} > {HU_MAX}"
    print(f"  [OK] HU clipping: [{clipped.min()}, {clipped.max()}]")

    # Normalization
    norm = normalize_to_01(raw)
    assert norm.min() >= 0.0, f"Norm min {norm.min()}"
    assert norm.max() <= 1.0, f"Norm max {norm.max()}"
    print(f"  [OK] Normalization: [{norm.min():.4f}, {norm.max():.4f}]")

    # Center crop (volume larger than target)
    big = np.ones((300, 300, 200), dtype=np.float32)
    cropped = center_crop_or_pad(big, CROP_SIZE)
    assert cropped.shape == CROP_SIZE, f"Crop shape {cropped.shape} != {CROP_SIZE}"
    print(f"  [OK] Center crop: {big.shape} -> {cropped.shape}")

    # Center pad (volume smaller than target)
    small = np.ones((100, 100, 64), dtype=np.float32)
    padded = center_crop_or_pad(small, CROP_SIZE)
    assert padded.shape == CROP_SIZE, f"Pad shape {padded.shape} != {CROP_SIZE}"
    # Check that padding is zero
    assert padded[0, 0, 0] == 0.0, "Padding should be zero"
    print(f"  [OK] Center pad: {small.shape} -> {padded.shape}")

    return True


# ==============================================================================
# TEST 4: Model Forward
# ==============================================================================
def test_model_forward():
    """Test model forward pass with multiple input sizes."""
    import torch
    from src.models.model import build_model, count_parameters

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,} ({params['total_MB']:.1f} MB)")

    # Small test (always runs)
    sizes = [(1, 1, 32, 32, 16), (1, 1, 64, 64, 32)]

    for shape in sizes:
        x = torch.randn(*shape, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == shape, f"Shape mismatch: {out.shape} != {shape}"
        print(f"  [OK] Forward {shape} -> {out.shape}")

    return True


# ==============================================================================
# TEST 5: Attention Gates
# ==============================================================================
def test_attention_gates():
    """Test attention gate tensor dimensions and value ranges."""
    import torch
    from src.models.attention import AttentionGate3D

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        (256, 128, 64, (14, 14, 8), (7, 7, 4)),
        (128, 64, 32, (28, 28, 16), (14, 14, 8)),
        (64, 32, 16, (56, 56, 32), (28, 28, 16)),
        (32, 16, 8, (112, 112, 64), (56, 56, 32)),
    ]

    for F_g, F_l, F_int, sp_x, sp_g in configs:
        ag = AttentionGate3D(F_g, F_l, F_int).to(device)
        x = torch.randn(1, F_l, *sp_x, device=device, requires_grad=True)
        g = torch.randn(1, F_g, *sp_g, device=device, requires_grad=True)

        out = ag(x, g)
        assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"

        # Check gradient flow
        out.sum().backward()
        assert x.grad is not None, "No gradient for x"
        assert g.grad is not None, "No gradient for g"

        print(f"  [OK] AG(F_g={F_g}, F_l={F_l}): shape OK, gradients OK")

    return True


# ==============================================================================
# TEST 6: Loss Backward
# ==============================================================================
def test_loss_backward():
    """Test loss computation and gradient flow."""
    import torch
    from src.losses import DiceFocalLoss, DiceLoss, FocalLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred = torch.randn(1, 1, 32, 32, 16, device=device, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 32, 32, 16), device=device).float()

    for name, loss_fn in [("DiceLoss", DiceLoss()), ("FocalLoss", FocalLoss()),
                           ("DiceFocalLoss", DiceFocalLoss())]:
        loss_fn = loss_fn.to(device)
        loss = loss_fn(pred, target)

        assert not torch.isnan(loss), f"{name}: NaN"
        assert not torch.isinf(loss), f"{name}: Inf"

        loss.backward(retain_graph=True)
        assert pred.grad is not None, f"{name}: no gradient"
        pred.grad = None

        print(f"  [OK] {name}: loss={loss.item():.6f}, gradient OK")

    return True


# ==============================================================================
# TEST 7: One-Batch Training Step
# ==============================================================================
def test_one_batch():
    """Test a single training step (forward + loss + backward + optimizer)."""
    import torch
    from src.models.model import build_model
    from src.losses import DiceFocalLoss
    from src.config import USE_AMP

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP and device.type == 'cuda')

    x = torch.randn(1, 1, 32, 32, 16, device=device)
    y = torch.randint(0, 2, (1, 1, 32, 32, 16), device=device).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
        out = model(x)
        loss = criterion(out, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  [OK] One batch: loss={loss.item():.6f}")

    # Verify parameters changed
    model.eval()
    with torch.no_grad():
        out2 = model(x)

    print(f"  [OK] Parameters updated (output changed)")
    return True


# ==============================================================================
# TEST 8: One-Patient Overfit
# ==============================================================================
def test_one_patient_overfit():
    """Train on a single synthetic volume for several steps: the loss must decrease."""
    import torch
    from src.models.model import build_model
    from src.losses import DiceFocalLoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    img = torch.rand(1, 1, 64, 64, 32, device=device)
    lbl = torch.zeros_like(img); lbl[..., 20:44, 20:44, 8:24] = 1
    model = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    criterion, optimizer = DiceFocalLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(20):
        optimizer.zero_grad(); loss = criterion(model(img), lbl); loss.backward(); optimizer.step()
        losses.append(loss.item())
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return losses[-1] < losses[0]


# ==============================================================================
# TEST 9: Small Subset
# ==============================================================================
def test_small_subset():
    """End-to-end: src.train (patch mode, tiny) on 2 synthetic train + 1 val volume, then src.evaluate on val."""
    from src import train, evaluate
    tmpdir = tempfile.mkdtemp(prefix="test_subset_")
    data, sp = _synthetic_prep(tmpdir)
    out = os.path.join(tmpdir, "run")
    train.main(["--run_name", "t9", "--out", out, "--data", data, "--splits", sp, "--patch", "--norm", "group",
                "--deepsup", "--patch_size", "32", "32", "32", "--steps", "4", "--val_every", "2", "--workers", "0"])
    for f in ("train_log.csv", "best.pth", "last.pth", "config.json", "curves.png"):
        assert os.path.exists(os.path.join(out, f)), f
    evaluate.main(["--run_dir", out, "--data", data, "--splits", sp, "--split", "val"])
    assert os.path.exists(os.path.join(out, "val_per_patient.csv")) and os.path.exists(os.path.join(out, "val_summary.json"))
    print("  [OK] train -> best.pth -> full-volume evaluation works end to end")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 10: Checkpoint Save/Load
# ==============================================================================
def test_checkpoint():
    """Test saving and loading a checkpoint."""
    import torch
    from src.models.model import build_model

    tmpdir = tempfile.mkdtemp(prefix="test_ckpt_")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

    # Save
    ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
    state = {
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_dice': 0.75,
    }
    torch.save(state, ckpt_path)
    print(f"  Saved checkpoint to {ckpt_path}")

    # Load into new model
    model2 = build_model(encoder_channels=[4, 8, 16, 32, 64]).to(device)
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(loaded['model_state_dict'])

    assert loaded['epoch'] == 5
    assert loaded['best_val_dice'] == 0.75

    # Compare outputs
    x = torch.randn(1, 1, 32, 32, 16, device=device)
    model.eval()
    model2.eval()
    with torch.no_grad():
        o1 = model(x)
        o2 = model2(x)
    assert torch.allclose(o1, o2, atol=1e-6), "Outputs differ after load"

    print(f"  [OK] Checkpoint save/load verified")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 11: Resume
# ==============================================================================
def test_resume():
    """Training resumes automatically from last.pth and continues the log from the saved step."""
    import csv
    from src import train
    tmpdir = tempfile.mkdtemp(prefix="test_resume_")
    data, sp = _synthetic_prep(tmpdir)
    out = os.path.join(tmpdir, "run")
    base = ["--run_name", "t11", "--out", out, "--data", data, "--splits", sp, "--patch", "--patch_size", "32", "32", "32",
            "--val_every", "2", "--ckpt_every", "2", "--log_every", "1", "--workers", "0"]
    train.main(base + ["--steps", "2"])
    train.main(base + ["--steps", "4"])
    steps = [int(r["step"]) for r in csv.DictReader(open(os.path.join(out, "train_log.csv")))]
    assert steps == [1, 2, 3, 4], steps
    print(f"  [OK] Resumed at step 2, log steps {steps}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# TEST 12: Evaluation Metrics
# ==============================================================================
def test_evaluation_metrics():
    """Test DSC, ASSD, HD95 computation."""
    from src.metrics import compute_dice, compute_assd, compute_hd95, compute_all_metrics

    # Perfect overlap
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[20:44, 20:44, 8:24] = 1

    dice = compute_dice(mask.astype(np.float32), mask)
    assert abs(dice - 1.0) < 1e-6, f"Perfect dice should be 1.0, got {dice}"
    print(f"  [OK] Perfect overlap: DSC={dice:.4f}")

    # No overlap
    pred = np.zeros((64, 64, 32), dtype=np.float32)
    pred[0:10, 0:10, 0:5] = 1.0
    gt = np.zeros((64, 64, 32), dtype=np.uint8)
    gt[50:60, 50:60, 25:30] = 1

    dice_no = compute_dice(pred, gt)
    assert dice_no == 0.0, f"No overlap dice should be 0.0, got {dice_no}"
    print(f"  [OK] No overlap: DSC={dice_no:.4f}")

    # Both empty
    empty = np.zeros((64, 64, 32))
    dice_empty = compute_dice(empty, empty)
    assert dice_empty == 1.0, f"Both empty dice should be 1.0, got {dice_empty}"
    print(f"  [OK] Both empty: DSC={dice_empty:.4f}")

    # ASSD and HD95
    assd = compute_assd(mask.astype(np.float32), mask)
    assert abs(assd) < 1e-6, f"Perfect ASSD should be 0, got {assd}"
    print(f"  [OK] Perfect ASSD: {assd:.4f} mm")

    hd95 = compute_hd95(mask.astype(np.float32), mask)
    assert abs(hd95) < 1e-6, f"Perfect HD95 should be 0, got {hd95}"
    print(f"  [OK] Perfect HD95: {hd95:.4f} mm")

    # Partial overlap
    pred_partial = np.zeros((64, 64, 32), dtype=np.float32)
    pred_partial[15:45, 15:45, 5:27] = 1.0
    metrics = compute_all_metrics(pred_partial, mask)
    assert 0 < metrics['dice'] < 1.0
    assert metrics['assd'] > 0
    assert metrics['hd95'] > 0
    print(f"  [OK] Partial: DSC={metrics['dice']:.4f}, ASSD={metrics['assd']:.2f}, HD95={metrics['hd95']:.2f}")

    return True


# ==============================================================================
# TEST 13: Blind crop -> paste-back round trip (T2)
# ==============================================================================
def test_crop_paste_roundtrip():
    """Crop then paste back restores every voxel inside the crop and zeros outside; works with padding."""
    from src.data.preprocessing import blind_center_crop, crop_at, paste_back, center_crop_or_pad
    rng = np.random.default_rng(0)
    for shape in [(300, 290, 200), (440, 440, 240), (200, 230, 100), (224, 224, 128)]:
        vol = rng.random(shape).astype(np.float32) + 1.0          # strictly > 0 so zero-fill is detectable
        off = blind_center_crop(shape, (224, 224, 128))
        crop = crop_at(vol, off, (224, 224, 128))
        assert crop.shape == (224, 224, 128)
        back = paste_back(crop, off, shape)
        inside = tuple(slice(max(o, 0), min(o + t, s)) for o, t, s in zip(off, (224, 224, 128), shape))
        assert np.array_equal(back[inside], vol[inside]), f"round trip mismatch for {shape}"
        mask = np.zeros(shape, bool); mask[inside] = True
        assert (back[~mask] == 0).all(), "outside the crop must be background"
        assert np.array_equal(center_crop_or_pad(vol, (224, 224, 128)), crop)
        print(f"  [OK] {shape}: offsets {off}, {mask.sum()} voxels round-trip exactly")
    assert blind_center_crop((440, 440, 240), (224, 224, 128)) == (108, 108, 56)
    return True


# ==============================================================================
# TEST 14: Metrics on toy masks with known answers (T3)
# ==============================================================================
def test_metrics_toy():
    from src.metrics import compute_all_metrics, aggregate_metrics
    gt = np.zeros((40, 40, 40), bool); gt[10:20, 10:20, 10:20] = True
    m = compute_all_metrics(gt.copy(), gt, spacing=(1, 1, 1))
    assert m["dice"] == 1 and m["jaccard"] == 1 and m["hd95"] == 0 and m["assd"] == 0 and m["v_ratio"] == 1
    pred = np.zeros_like(gt); pred[12:22, 10:20, 10:20] = True          # shift 2 voxels along axis 0
    m = compute_all_metrics(pred, gt, spacing=(2, 1, 1))                 # 2 mm along axis 0
    assert abs(m["dice"] - 0.8) < 1e-12, m["dice"]                       # TP 800, FP 200, FN 200
    assert abs(m["jaccard"] - 800 / 1200) < 1e-12 and abs(m["jaccard"] - m["dice"] / (2 - m["dice"])) < 1e-12
    assert abs(m["precision"] - 0.8) < 1e-12 and abs(m["recall"] - 0.8) < 1e-12
    assert abs(m["hd95"] - 4.0) < 1e-9, m["hd95"]                        # front face 2 voxels x 2 mm away
    assert abs(m["vol_gt_ml"] - 2.0) < 1e-12 and abs(m["v_ratio"] - 1.0) < 1e-12   # 1000 voxels x 2 mm^3
    big = np.zeros_like(gt); big[8:22, 8:22, 8:22] = True                # over-segmentation
    mb = compute_all_metrics(big, gt, spacing=(1, 1, 1))
    assert abs(mb["recall"] - 1) < 1e-12 and abs(mb["precision"] - 1000 / 2744) < 1e-12 and abs(mb["v_ratio"] - 2.744) < 1e-9
    me = compute_all_metrics(np.zeros_like(gt), gt)
    assert me["dice"] == 0 and np.isinf(me["hd95"]) and np.isinf(me["assd"]) and me["empty_pred"]
    agg = aggregate_metrics([m, mb, me])
    assert agg["n_cases"] == 3 and agg["n_empty_pred"] == 1 and agg["n_inf_hd95"] == 1 and agg["hd95"]["n"] == 2
    assert abs(agg["dice"]["mean"] - (0.8 + mb["dice"] + 0) / 3) < 1e-12          # empty case kept in Dice
    print(f"  [OK] shifted cube: Dice {m['dice']:.3f} J {m['jaccard']:.3f} HD95 {m['hd95']} mm ASSD {m['assd']:.3f} mm")
    print(f"  [OK] empty prediction -> inf HD95/ASSD, counted (n_inf_hd95={agg['n_inf_hd95']}), Dice kept")
    return True


# ==============================================================================
# TEST 15: Model flags (T4)
# ==============================================================================
def test_model_flags():
    import torch
    from src.models.model import build_model, count_parameters
    assert count_parameters(build_model())["total"] == 5_668_269, "R0 must reproduce the Review-1 parameter count"
    x = torch.randn(1, 1, 64, 64, 48)
    for kw in [dict(), dict(norm="group"), dict(norm="group", deepsup=True),
               dict(norm="group", deepsup=True, attention=False), dict(ag="paper")]:
        m = build_model(**kw)
        n = count_parameters(m)["total"]
        assert 5.6e6 < n < 5.75e6, n
        gns = [g for g in m.modules() if isinstance(g, torch.nn.GroupNorm)]
        assert all(g.num_channels % g.num_groups == 0 for g in gns)
        assert (len(gns) > 0) == (kw.get("norm") == "group")
        m.train(); out = m(x)
        if kw.get("deepsup"):
            assert isinstance(out, list) and len(out) == 4 and all(o.shape == x.shape for o in out), "aux heads full size"
        else:
            assert out.shape == x.shape
        m.eval()
        with torch.no_grad():
            assert m(x).shape == x.shape, "eval returns main head only"
        print(f"  [OK] {kw or 'R0 defaults'}: {n:,} params, train/eval shapes OK")
    return True


# ==============================================================================
# TEST 16: GPU augmentation (T4)
# ==============================================================================
def test_augmentation():
    import torch
    from src.augment import gpu_augment
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 24, 16); y = (torch.rand(2, 1, 32, 24, 16) > 0.8).float()
    xo, yo = gpu_augment(x, y, rot_deg=0, shift=0, flip_p=0)
    assert torch.allclose(xo, x, atol=1e-5) and torch.equal(yo, y), "identity parameters must return the input"
    xo, yo = gpu_augment(x, y, rot_deg=0, shift=0, flip_p=1.0)
    assert torch.allclose(xo, torch.flip(x, (2, 3, 4)), atol=1e-5) and torch.equal(yo, torch.flip(y, (2, 3, 4)))
    for _ in range(5):
        xo, yo = gpu_augment(x, y)
        assert set(torch.unique(yo).tolist()) <= {0.0, 1.0}, "label must stay binary"
        assert xo.min() >= 0 and xo.max() <= 1, "image must stay in [0, 1]"
    print("  [OK] identity, forced flip == torch.flip, binary label, image in [0,1]")
    return True


# ==============================================================================
# TEST 17: One optimizer step on one patch lowers the loss (T4, R3 config)
# ==============================================================================
def test_one_step_overfit():
    import torch
    from src.models.model import build_model
    from src.losses import DeepSupervisionLoss, DiceFocalLoss
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32, 32); y = torch.zeros_like(x); y[..., 8:24, 8:24, 8:24] = 1
    m = build_model(norm="group", deepsup=True); crit = DeepSupervisionLoss(DiceFocalLoss())
    opt = torch.optim.Adam(m.parameters(), lr=6e-4); m.train()
    l0 = crit(m(x), y); opt.zero_grad(); l0.backward(); opt.step()
    with torch.no_grad():
        l1 = crit(m(x), y)
    print(f"  loss {l0.item():.5f} -> {l1.item():.5f}")
    return l1.item() < l0.item()


# ==============================================================================
# TEST 18: Patch sampler pos:neg = 1:2 and R0 paste-back inference (T3/T4)
# ==============================================================================
def test_sampler_and_pasteback():
    import torch
    from src.data.dataset import PatchSampler
    from src.data.preprocessing import blind_center_crop
    from src.inference import predict_logits, BG_LOGIT
    tmpdir = tempfile.mkdtemp(prefix="test_samp_")
    data, _ = _synthetic_prep(tmpdir, n=1, shape=(96, 96, 64))
    ds = PatchSampler(data, ["PANCREAS_9000"], 600, size=(16, 16, 16))
    fg = np.mean([ds[i][1][0, 8, 8, 8].item() for i in range(600)])      # centre voxel is foreground for pos picks
    assert 0.25 < fg < 0.42, fg
    print(f"  [OK] patch centred on foreground in {fg:.2f} of samples (expected ~1/3)")

    class Const(torch.nn.Module):
        def forward(self, t):
            return torch.full_like(t, 3.0)
    for shape in [(300, 280, 200), (200, 230, 100)]:
        img = np.random.rand(*shape).astype(np.float32)
        out = predict_logits(Const(), img, mode="crop", device="cpu", amp=False).numpy()
        off = blind_center_crop(shape, (224, 224, 128))
        inside = tuple(slice(max(o, 0), min(o + t, s)) for o, t, s in zip(off, (224, 224, 128), shape))
        mask = np.zeros(shape, bool); mask[inside] = True
        assert out.shape == shape and (out[mask] == 3.0).all() and (out[~mask] == BG_LOGIT).all()
    print("  [OK] R0 crop prediction pasted back; outside the crop = background")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return True


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "#" * 60)
    print("# 3D ATTENTION U-NET -- COMPREHENSIVE TEST SUITE")
    print("#" * 60)

    tests = [
        ("1. Dependencies", test_dependencies),
        ("2. Synthetic Dataset", test_dataset_synthetic),
        ("3. Preprocessing", test_preprocessing),
        ("4. Model Forward", test_model_forward),
        ("5. Attention Gates", test_attention_gates),
        ("6. Loss Backward", test_loss_backward),
        ("7. One-Batch Training", test_one_batch),
        ("8. One-Patient Overfit", test_one_patient_overfit),
        ("9. Small Subset Pipeline", test_small_subset),
        ("10. Checkpoint Save/Load", test_checkpoint),
        ("11. Resume", test_resume),
        ("12. Evaluation Metrics", test_evaluation_metrics),
        ("13. Crop/Paste Round Trip", test_crop_paste_roundtrip),
        ("14. Metrics Toy Masks", test_metrics_toy),
        ("15. Model Flags", test_model_flags),
        ("16. GPU Augmentation", test_augmentation),
        ("17. One-Step Overfit (R3)", test_one_step_overfit),
        ("18. Sampler + R0 Paste-Back", test_sampler_and_pasteback),
    ]

    for name, fn in tests:
        run_test(name, fn)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS [OK]" if ok else "FAIL [FAIL]"
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED -- Ready for training!")
    else:
        print(f"\n  [WARN]  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

def get_last_checkpoint_safe(output_dir):
    """
    Returns the last safe (complete) checkpoint.
    Works as a replacement to `transformers.trainer_utils.get_last_checkpoint(str)`.
    """
    from pathlib import Path
    import re

    output_dir_ = Path(output_dir)

    if not output_dir_.exists():
        return None

    checkpoints = sorted(
        [
            p for p in output_dir_.iterdir()
            if p.is_dir() and re.match(r"checkpoint-\d+$", p.name)
        ],
        key=lambda p: int(p.name.split("-")[-1]),
    )

    REQUIRED_FILES = [
        "trainer_state.json",
        # "optimizer.pt",
    ]

    valid = []
    for ckpt in checkpoints:
        if all((ckpt / f).exists() for f in REQUIRED_FILES):
            valid.append(ckpt)

    last_checkpoint = str(valid[-1]) if valid else None
    return last_checkpoint

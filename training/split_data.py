from collections import defaultdict

def split_no_leakage(records, test_frac=0.15):
    """Group by field/capture-burst ID before splitting so near-duplicate
    images from the same photo session never land on both sides."""
    groups = defaultdict(list)
    for r in records:
        groups[r["burst_id"]].append(r)

    group_ids = list(groups.keys())
    n_test = int(len(group_ids) * test_frac)
    test_ids = set(group_ids[:n_test])

    train = [r for gid, rs in groups.items() if gid not in test_ids for r in rs]
    test = [r for gid, rs in groups.items() if gid in test_ids for r in rs]
    return train, test
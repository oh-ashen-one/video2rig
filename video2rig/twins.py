"""Twin limbs given in creature.json, for rigs whose bone names carry no .L/.R (a Tripo rig's "0_Left_Limb" and
"1_Left_Limb" are its two FRONT legs, and "_L" in "_Limb" read as "left" for every limb):

    "twins": [["tripo::bone_11", "tripo::bone_6"], ["tripo::0_Right_Limb_0", "tripo::1_Right_Limb_0"]]

Each pair names the same bone of a limb on both sides (any order: which one is the creature's left is measured on
the start pose). From each pair the two chains are walked down together: a bone with one child pairs with the twin's
one child; where a chain branches, the children are paired by where they lie mirrored across the body's middle (head
and tail), and what has no partner (a toe bone only one side has) is kept on its side, unpaired, and logged. Pairs
given further down a chain already walked are checked against it.

When "twins" is set it replaces the names: reader.side_of / twin_label answer from it (bones it does not name are
on the centre line), and the left / right check of the reader fit, the far-limb and behind-the-twin tests and the
training poses that bring twin limb ends together use these chains (Creature.twin_chains). Without it nothing
changes.

    python -m video2rig.twins <project>      the pairs, sides and unpaired bones the tool will use (no clip needed)
"""
import sys

import numpy as np

_SAID = set()        # log lines already printed by this process (a creature is built several times)


def resolve(cr, log=None):
    """Read cr.cfg['twins'] into cr.twins (None without it) and register it with the reader's name rules.
    cr.twins: dict(pairs [(left bone, right bone)], side {bone: 'L'|'R'}, chains [(left ids, right ids)],
    unpaired [bone names], notes [str])."""
    from .reader import set_twins
    spec = cr.cfg.get('twins')
    if not spec:
        cr.twins = None
        set_twins(None)
        return None
    from scipy.optimize import linear_sum_assignment
    rig = cr.rig
    H, T = rig.bone_points(cr.p0(None))
    lateral = np.cross(cr.up, cr.facing)
    lateral /= np.linalg.norm(lateral)
    mid = float(H[rig.root] @ lateral)
    mirror = lambda x: x - 2 * ((x @ lateral) - mid)[..., None] * lateral
    kids = {j: [k for k in range(rig.nb) if rig.parents[k] == j] for j in range(rig.nb)}

    def below(j):
        out, todo = [], [j]
        while todo:
            b = todo.pop(0)
            out.append(b)
            todo += kids[b]
        return out
    twin, side, chains, unpaired, notes = {}, {}, [], [], []
    for pair in spec:
        if len(pair) != 2 or any(n not in rig.names for n in pair):
            raise SystemExit(f'creature.json "twins": {pair} is not a pair of bone names of the rig')
        a, b = (rig.names.index(n) for n in pair)
        if a in twin or b in twin:
            if twin.get(a) != b:
                notes.append(f'{pair} disagrees with the chains walked before (kept: '
                             f'{rig.names[a]} <-> {rig.names[twin[a]] if a in twin else None})')
            continue
        # which is the left one: the bone's middle, not its head (two shoulders can start at one point on the spine)
        la, lb = (float((H[j] + T[j]) @ lateral) / 2 - mid for j in (a, b))
        left, right = (a, b) if la >= lb else (b, a)
        if abs(la - lb) < 1e-6:
            notes.append(f'{pair}: both on the middle line, the first taken as the left')
        lc, rc, queue = [], [], [(left, right)]
        while queue:
            l, r = queue.pop(0)
            twin[l], twin[r] = r, l
            lc.append(l)
            rc.append(r)
            kl = [k for k in kids[l] if k not in twin]
            kr = [k for k in kids[r] if k not in twin]
            if not kl or not kr:
                rest = [(k, 'L') for k in kl] + [(k, 'R') for k in kr]
            else:
                cost = np.array([[np.linalg.norm(H[i] - mirror(H[j])) + np.linalg.norm(T[i] - mirror(T[j]))
                                  for j in kr] for i in kl])
                rows, cols = linear_sum_assignment(cost)
                queue += [(kl[i], kr[j]) for i, j in zip(rows, cols)]
                rest = [(k, 'L') for i, k in enumerate(kl) if i not in rows] + \
                       [(k, 'R') for j, k in enumerate(kr) if j not in cols]
            for k, s in rest:                          # no partner: on its side with everything below it
                for d in below(k):
                    (lc if s == 'L' else rc).append(d)
                    unpaired.append(rig.names[d])
        for j in lc:
            side[rig.names[j]] = 'L'
        for j in rc:
            side[rig.names[j]] = 'R'
        chains.append((lc, rc))
    pairs = [(rig.names[l], rig.names[r]) for l, r in twin.items() if side.get(rig.names[l]) == 'L']
    set_twins(side, {rig.names[j]: rig.names[k] for j, k in twin.items()})
    cr.twins = dict(pairs=pairs, side=side, chains=chains, unpaired=unpaired, notes=notes)
    msg = (f'twins: {len(pairs)} pairs from creature.json' + (f'; no twin for {unpaired}' if unpaired else '')
           + ''.join(f'; {n}' for n in notes))
    if log is not None and (unpaired or notes) and msg not in _SAID:
        _SAID.add(msg)
        log(msg)
    return cr.twins


def main():
    from .common import Project
    from .creature import Creature
    if len(sys.argv) != 2:
        raise SystemExit('python -m video2rig.twins <project>')
    pr = Project(sys.argv[1])
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    from .reader import side_of, twin_label
    if cr.twins is None:
        print('no "twins" in creature.json: the pairs come from the bone names (.L/.R, _L/_R, Left/Right)')
        pairs = [(n, twin_label(n)) for n in cr.rig.names if side_of(n) == 'L' and twin_label(n) in cr.rig.names]
    else:
        pairs = cr.twins['pairs']
    print(f'TWINS {pr.name}: {len(pairs)} pairs (left <-> right; left = the creature\'s left, facing '
          f'{np.round(cr.facing, 2).tolist()})')
    for l, r in pairs:
        print(f'  {l:28s} <-> {r}')
    if cr.twins is not None:
        print(f'  unpaired (kept on their side): {cr.twins["unpaired"] or "none"}')
        for n in cr.twins['notes']:
            print(f'  note: {n}')
        print('  twin limb chains (root first):')
        for lc, rc in cr.twins['chains']:
            print(f'    L {[cr.rig.names[j] for j in lc]}\n    R {[cr.rig.names[j] for j in rc]}')
    chains = cr.twin_chains()
    print(f'  limb chains the left/right check sees: {[cr.rig.names[c[0]] + " (" + str(side_of(cr.rig.names[c[0]])) + ")" for c in chains]}')


if __name__ == '__main__':
    main()

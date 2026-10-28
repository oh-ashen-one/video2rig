"""Limb chains and feet given in creature.json, for rigs whose limbs the skeleton's shape does not find: the tool
takes a limb as a chain that leaves a hub and runs on without branching, so a leg that ends in two bones (a Tripo
paw: a toe bone and a paw bone side by side) or a wing with two fingers is no limb at all, and a creature whose four
legs all branch stands on none (no foot lock, no steps, no stance pinning).

    "limbs": {
     "front_l": {"chain": ["bone_6", "bone_7", "tripo::1_Left_Limb_0", "bone_9"], "foot": "bone_9", "standing": true},
     "wing_l":  {"chain": ["WingRoot", "WingWrist", "WingTip"], "standing": false}
    }

chain: the bones from the shoulder or hip down to the end, each the child of the one before. foot: the bone that
stands on the floor; it must be the chain's last bone (the tool reads a limb's end as its foot everywhere: standing
legs, foot lock, steps, flat feet, the floor). standing: true (a leg it stands on), false (an arm, a wing: never a
standing leg), or left out (the tool's own test: the last two bones' skin reaches the floor in the start pose).

A named limb takes the place of every auto-detected chain it shares a bone with; the others (a neck, a tail) are
still found by the shape. With "twins" (video2rig/twins.py) too, each limb's twin bones should lie in its twin
limb: where they do not, it is logged. Without "limbs" nothing changes.

    python -m video2rig.limbs <project>      the chains, which stand, the feet, the twins check (no clip needed)
"""
import sys

_SAID = set()


def explicit(cr):
    """[(name, chain ids, standing True|False|None)] from creature.json "limbs" ([] without it), checked."""
    spec = cr.cfg.get('limbs')
    if not spec:
        return []
    if getattr(cr, '_limbs', None) is not None:
        return cr._limbs
    rig = cr.rig
    out, seen = [], {}
    for name, d in spec.items():
        chain = d.get('chain') if isinstance(d, dict) else None
        if not chain:
            raise SystemExit(f'creature.json "limbs".{name}: needs a "chain" of bone names')
        bad = [b for b in chain if b not in rig.names]
        if bad:
            raise SystemExit(f'creature.json "limbs".{name}: no such bones {bad}')
        ids = [rig.names.index(b) for b in chain]
        for a, b in zip(ids, ids[1:]):
            if rig.parents[b] != a:
                raise SystemExit(f'creature.json "limbs".{name}: {rig.names[b]} is not a child of {rig.names[a]}')
        if d.get('foot') not in (None, chain[-1]):
            raise SystemExit(f'creature.json "limbs".{name}: the foot ({d["foot"]}) must be the chain\'s last bone '
                             f'({chain[-1]}); end the chain at the foot')
        for j in ids:
            if j in seen:
                raise SystemExit(f'creature.json "limbs": {rig.names[j]} is in both {seen[j]} and {name}')
            seen[j] = name
        st = d.get('standing')
        out.append((name, ids, None if st is None else bool(st)))
    cr._limbs = out
    return out


def merge(cr, chains):
    """The auto-detected chains with every one that shares a bone with a named limb replaced by the named ones."""
    mine = explicit(cr)
    if not mine:
        return chains
    named = {j for _, ids, _ in mine for j in ids}
    return [c for c in chains if not set(c) & named] + [list(ids) for _, ids, _ in mine]


def standing(cr, chain):
    """True / False when the chain is a named limb with "standing" given, else None (the tool's own test)."""
    for _, ids, st in explicit(cr):
        if list(ids) == list(chain):
            return st
    return None


def check_twins(cr, log=None):
    """Notes where a named limb's twin bones (creature.json "twins") do not all lie in one other named limb."""
    tw = getattr(cr, 'twins', None)
    mine = explicit(cr)
    if not tw or not mine:
        return []
    rig = cr.rig
    from .reader import twin_label
    limb_of = {j: name for name, ids, _ in mine for j in ids}
    notes = []
    for name, ids, st in mine:
        partners = {limb_of.get(rig.names.index(twin_label(rig.names[j])), '(no limb)')
                    for j in ids if twin_label(rig.names[j]) in rig.names}
        lone = [rig.names[j] for j in ids if twin_label(rig.names[j]) not in rig.names]
        if len(partners) > 1:
            notes.append(f'limb {name}: its twin bones lie in {sorted(partners)}, not in one limb')
        other = next(iter(partners)) if len(partners) == 1 else None
        if other and other != '(no limb)':
            st2 = next(s for n, _, s in mine if n == other)
            if st != st2:
                notes.append(f'limb {name} standing {st}, its twin {other} standing {st2}')
        if lone:
            notes.append(f'limb {name}: no twin for {lone}')
    for n in notes:
        if log is not None and n not in _SAID:
            _SAID.add(n)
            log(f'limbs: {n}')
    return notes


def main():
    import numpy as np
    from .common import Project
    from .creature import Creature
    if len(sys.argv) != 2:
        raise SystemExit('python -m video2rig.limbs <project>')
    pr = Project(sys.argv[1])
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    rig = cr.rig
    named = {tuple(ids): (name, st) for name, ids, st in explicit(cr)}
    p0 = cr.p0(None)
    H0, T0 = rig.bone_points(p0)
    print(f'LIMBS {pr.name}: {len(cr.limb_chains(as_list=True))} limb chains'
          + ('' if named else ' (no "limbs" in creature.json: all by the skeleton\'s shape)'))
    for ch in cr.limb_chains(as_list=True):
        name, st = named.get(tuple(ch), ('(auto)', None))
        j = ch[-1]
        d0 = T0[j] - H0[j]
        on_floor = T0[j] @ cr.up - cr.floor <= .05 * cr.height
        flat = abs(d0 @ cr.up) / max(np.linalg.norm(d0), 1e-9) <= .8
        leg = set(ch) <= set(cr.legs)
        print(f'  {name:9s} {"STANDS" if leg else "free  "} '
              f'(standing {"given " + str(st).lower() if st is not None else "by the floor test"}) '
              f'{"foot" if leg else "end"} {rig.names[j]}{" -> foot lock, steps" if leg else ""}'
              f'{"; flat-foot term" if on_floor and flat else "; no flat-foot term (" + ("points down" if not flat else "off the floor") + ")" if leg else ""}'
              f'\n            {[rig.names[b] for b in ch]}')
    print(f'  standing-leg bones: {len(cr.legs)}; feet the tool locks: '
          f'{[rig.names[ch[-1]] for ch in cr.limb_chains(as_list=True) if set(ch) <= set(cr.legs)]}')
    notes = check_twins(cr)
    if cr.twins is not None:
        print('  twins check: ' + ('every named limb\'s twin bones lie in one twin limb' if not notes else ''))
        for n in notes:
            print(f'    {n}')


if __name__ == '__main__':
    main()

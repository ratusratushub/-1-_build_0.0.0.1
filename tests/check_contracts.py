"""Source contracts and independent numerical reference checks; no C++ build or GL context.
These checks supplement, and never replace, the CTest CPU and GPU executables.
"""
from pathlib import Path
import math
import re
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def random_unit(x):
    x &= 0xffffffff
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xffffffff
    x ^= x >> 15
    x = (x * 0x846ca68b) & 0xffffffff
    x ^= x >> 16
    return (x >> 8) / 16777216


def locate(p):
    if any(not math.isfinite(v) or v < -500 or v > 500 for v in p):
        return 1000
    def axis(value):
        value = f32(value)
        cell = max(0, min(9, math.floor(f32(f32(value + 500) / 100))))
        while cell > 0 and value < cell * 100 - 500:
            cell -= 1
        while cell < 9 and value >= (cell + 1) * 100 - 500:
            cell += 1
        return cell
    x, y, z = [axis(v) for v in p]
    return (z * 10 + y) * 10 + x


class Contracts(unittest.TestCase):
    def test_case_sensitive_paths(self):
        for p in list((ROOT / 'include').rglob('*.hpp')) + list((ROOT / 'src').glob('*.cpp')) + list((ROOT / 'tests').glob('*.cpp')):
            for include in re.findall(r'#include <(space/[^>]+)>', p.read_text()):
                self.assertTrue((ROOT / 'include' / include).is_file(), f'{p}: {include}')
        for p in re.findall(r'(?:src|tests)/[A-Za-z_]+\.cpp', (ROOT / 'CMakeLists.txt').read_text()):
            self.assertTrue((ROOT / p).is_file(), p)

    def test_shader_includes(self):
        for shader in (ROOT / 'shaders').iterdir():
            for include in re.findall(r'^#include "([^"]+)"', shader.read_text(), re.MULTILINE):
                self.assertTrue((shader.parent / include).is_file(), include)
        for name in ('aggregate.comp', 'neighbors.comp'):
            self.assertIn('#include "grid.glsl"', (ROOT / 'shaders' / name).read_text())
        self.assertIn('std::fma', (ROOT / 'include/space/World.hpp').read_text())
        self.assertIn('fma(', (ROOT / 'shaders/grid.glsl').read_text())

    def test_shader_interfaces(self):
        required = {'force': {0, 2, 3}, 'integrate': {0, 1, 2},
                    'aggregate': {0, 4, 5, 6, 7}, 'neighbors': {0, 2, 6, 7}, 'erase': set(range(8))}
        for name, bindings in required.items():
            s = (ROOT / 'shaders' / f'{name}.comp').read_text()
            self.assertIn('#version 430 core', s)
            self.assertIn('local_size_x=256', s)
            self.assertRegex(s, r'if\s*\(id\s*>=\s*particleCount\)\s*return;')
            actual = {int(v) for v in re.findall(r'binding\s*=\s*(\d+)', s)}
            self.assertEqual(actual, bindings)
            self.assertTrue(all(v < 8 for v in actual))
        cpp = (ROOT / 'src' / 'ComputePipeline.cpp').read_text()
        self.assertIn('GL_SHADER_STORAGE_BARRIER_BIT', cpp)
        self.assertIn('GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT', cpp)
        self.assertIn('GL_BUFFER_UPDATE_BARRIER_BIT', cpp)
        self.assertIn('glClientWaitSync(s.fence,0,0)', cpp)
        self.assertNotIn('atomicCompSwap', (ROOT / 'shaders' / 'aggregate.comp').read_text())

    def test_boundary_reference_and_compact_ranges(self):
        points = [(-500, -500, -500), (500, 500, 500), (0, 0, 0),
                  (-400.01, -499, -499), (-400, -499, -499), (501, 0, 0), (math.nan, 0, 0)]
        self.assertEqual([locate(p) for p in points], [0, 999, 555, 0, 1, 1000, 1000])
        self.assertEqual(locate((-0.00001, 0, 0)), 554)
        self.assertEqual(locate((99.99999, 0, 0)), 555)
        self.assertEqual(locate((-400.001, -450, -450)), 0)
        self.assertEqual(locate((-399.901, -450, -450)), 1)
        counts = [0] * 1001
        cells = [locate(p) for p in points]
        for cell in cells:
            counts[cell] += 1
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        cursors = offsets[:-1].copy()
        slots = [None] * len(points)
        for identity, cell in enumerate(cells):
            slots[cursors[cell]] = identity
            cursors[cell] += 1
        self.assertEqual(sorted(slots), list(range(len(points))))
        self.assertEqual(offsets[-1], len(points))
        for cell in range(1001):
            self.assertTrue(all(cells[identity] == cell for identity in slots[offsets[cell]:offsets[cell + 1]]))

    def test_seeded_volume_reference(self):
        buckets = [0] * 8
        sums = [0.] * 3
        for identity in range(100000):
            key = identity * 8 + 42 * 0x9e3779b9
            p = [f32(f32(2 * random_unit(key + axis) - 1) * 500) for axis in range(3)]
            self.assertTrue(all(-500 <= v <= 500 for v in p))
            buckets[sum((1 << axis) for axis, v in enumerate(p) if v >= 0)] += 1
            for axis in range(3):
                sums[axis] += p[axis]
        self.assertTrue(all(11500 < n < 13500 for n in buckets), buckets)
        self.assertTrue(all(abs(s / 100000) < 5 for s in sums), sums)

    def test_identity_compaction_reference(self):
        identities = list(range(9))
        membership = [0, 1, 0, 2, 1, 5, 5, 1, 0]
        survivors = [identity for identity, cell in zip(identities, membership) if cell != 1]
        self.assertEqual(survivors, [0, 2, 3, 5, 6, 8])
        result = survivors + list(range(9, 16))
        self.assertEqual(len(set(result)), len(result))
        self.assertTrue(all(identity >= 9 for identity in result[len(survivors):]))
        shader = (ROOT / 'shaders/erase.comp').read_text()
        self.assertIn('newIdentities[outputSlot]=identities[id]', shader)
        self.assertIn('membership[id]==targetChunk', shader)
        source = (ROOT / 'src/EntityManager.cpp').read_text()
        self.assertIn('nextIdentity_+=count', source)
        self.assertIn('massUpperBound_ remains conservative', source)

    def test_softened_verlet_reference(self):
        def accel(p):
            r2 = sum(x * x for x in p) + 25
            return [-10000 * x / (r2 * math.sqrt(r2)) for x in p]
        def energy(p, v):
            return sum(x * x for x in v) / 2 - 10000 / math.sqrt(sum(x * x for x in p) + 25)
        self.assertEqual(accel([0, 0, 0]), [0., 0., 0.])
        p = [50., 0., 0.]
        v = [0., math.sqrt(10000 * 2500 / 2525 ** 1.5), 0.]
        start = energy(p, v)
        for _ in range(6000):
            a = accel(p)
            v = [f32(v[i] + 0.001 * a[i]) for i in range(3)]
            p = [f32(p[i] + 0.002 * v[i]) for i in range(3)]
            a = accel(p)
            v = [f32(v[i] + 0.001 * a[i]) for i in range(3)]
        self.assertTrue(all(math.isfinite(x) for x in p + v))
        self.assertLess(abs((energy(p, v) - start) / start), 0.001)


if __name__ == '__main__':
    unittest.main(verbosity=2)

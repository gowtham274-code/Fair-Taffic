import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from simulation.network.builder import write_plain_files
from simulation.network.topology import NetTopology
from simulation.scenarios.demand import SCENARIO_RATES, generate_demand, write_route_file

NET_XML = """<net>
<junction id="J" type="traffic_light" x="0" y="0"/><junction id=":J_0" type="internal" x="0" y="0"/>
<junction id="W" type="dead_end" x="-100" y="0"/><junction id="E" type="dead_end" x="100" y="0"/>
<junction id="N" type="dead_end" x="0" y="100"/><junction id="S" type="dead_end" x="0" y="-100"/>
<edge id="W_J" from="W" to="J"><lane id="W_J_0" length="100"/></edge><edge id="J_W" from="J" to="W"><lane id="J_W_0" length="100"/></edge>
<edge id="E_J" from="E" to="J"><lane id="E_J_0" length="100"/></edge><edge id="J_E" from="J" to="E"><lane id="J_E_0" length="100"/></edge>
<edge id="N_J" from="N" to="J"><lane id="N_J_0" length="100"/></edge><edge id="J_N" from="J" to="N"><lane id="J_N_0" length="100"/></edge>
<edge id="S_J" from="S" to="J"><lane id="S_J_0" length="100"/></edge><edge id="J_S" from="J" to="S"><lane id="J_S_0" length="100"/></edge>
<edge id=":J_0" function="internal"><lane id=":J_0_0" length="5"/></edge>
<connection from="W_J" to="J_E"/><connection from="W_J" to="J_N"/><connection from="E_J" to="J_W"/>
<connection from="N_J" to="J_S"/><connection from="S_J" to="J_N"/><connection from="S_J" to="J_E"/>
</net>"""


class TestTopology(unittest.TestCase):
    def test_grid_structure(self):
        t = NetTopology.grid(2, 2, 200)
        self.assertEqual(t.tls_ids, ["J00", "J01", "J10", "J11"])
        self.assertEqual(len(t.entry_edges()), 8)
        self.assertEqual(len(t.exit_edges()), 8)
        for tls in t.tls_ids:
            self.assertEqual(set(t.incoming_slots(tls)), {"N", "E", "S", "W"})
        self.assertEqual(t.validate(expect_tls=4), [])

    def test_slots_geometry(self):
        t = NetTopology.grid(2, 2, 200)
        s = t.incoming_slots("J00")
        self.assertEqual(s["W"], "W0_J00")   # comes from the west boundary
        self.assertEqual(s["S"], "S0_J00")
        self.assertEqual(s["E"], "J10_J00")
        self.assertEqual(s["N"], "J01_J00")

    def test_neighbors_and_sides(self):
        t = NetTopology.grid(2, 2, 200)
        self.assertEqual(t.neighbors("J00"), ["J01", "J10"])
        self.assertEqual({side for _e, side in t.entry_edges()}, {"N", "E", "S", "W"})

    def test_routes_follow_connections_and_no_uturn(self):
        t = NetTopology.grid(2, 2, 200)
        r = t.route("W0_J00", "J11_E1")
        self.assertIsNotNone(r)
        for a, b in zip(r, r[1:]):
            self.assertIn(b, t.connections[a])
            self.assertNotEqual(t.edges[b][1], t.edges[a][0])  # no U-turn
        self.assertIsNone(t.route("W0_J00", "nonexistent"))

    def test_from_net_xml_parses_and_skips_internal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.net.xml"
            p.write_text(NET_XML)
            t = NetTopology.from_net_xml(p)
        self.assertEqual(t.tls_ids, ["J"])
        self.assertNotIn(":J_0", t.edges)
        self.assertEqual(t.incoming_slots("J"), {"W": "W_J", "E": "E_J", "N": "N_J", "S": "S_J"})
        self.assertEqual(t.route("W_J", "J_E"), ["W_J", "J_E"])
        self.assertIsNone(t.route("E_J", "J_E"))  # no connection E_J -> J_E

    def test_plain_files_written_and_wellformed(self):
        with tempfile.TemporaryDirectory() as d:
            f = write_plain_files(d, 2, 2, 200)
            nodes = ET.parse(f["nod"]).getroot().findall("node")
            edges = ET.parse(f["edg"]).getroot().findall("edge")
        self.assertEqual(len(nodes), 12)
        self.assertEqual(sum(1 for n in nodes if n.get("type") == "traffic_light"), 4)
        self.assertEqual(len(edges), 24)  # 4 internal pairs + 8 boundary pairs, both directions


class TestDemand(unittest.TestCase):
    def setUp(self):
        self.t = NetTopology.grid(2, 2, 200)

    def test_deterministic(self):
        a = generate_demand(self.t, "medium", 600, seed=7)
        b = generate_demand(self.t, "medium", 600, seed=7)
        c = generate_demand(self.t, "medium", 600, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_valid_sorted_routes(self):
        vs = generate_demand(self.t, "high", 600, seed=1)
        self.assertGreater(len(vs), 100)
        self.assertEqual([v.depart for v in vs], sorted(v.depart for v in vs))
        for v in vs[:50]:
            self.assertIn(v.route[-1], self.t.exit_edges())
            for a, b in zip(v.route, v.route[1:]):
                self.assertIn(b, self.t.connections[a])

    def test_scenario_ordering_and_uneven(self):
        n = {s: len(generate_demand(self.t, s, 1800, seed=3)) for s in SCENARIO_RATES}
        self.assertLess(n["low"], n["medium"])
        self.assertLess(n["medium"], n["high"])
        vs = generate_demand(self.t, "uneven", 1800, seed=3)
        west = sum(1 for v in vs if v.route[0].startswith("W"))
        east = sum(1 for v in vs if v.route[0].startswith("E"))
        self.assertGreater(west, 2.5 * east)

    def test_scale_and_errors(self):
        self.assertGreater(len(generate_demand(self.t, "low", 600, 1, scale=3)), len(generate_demand(self.t, "low", 600, 1)))
        with self.assertRaises(ValueError):
            generate_demand(self.t, "bogus", 600, 1)

    def test_route_file_is_valid_xml(self):
        vs = generate_demand(self.t, "low", 300, seed=1)
        with tempfile.TemporaryDirectory() as d:
            p = write_route_file(vs, Path(d) / "x" / "r.rou.xml")
            root = ET.parse(p).getroot()
        self.assertEqual(len(root.findall("vehicle")), len(vs))
        self.assertEqual(len(root.findall("vType")), 1)


if __name__ == "__main__":
    unittest.main()

from typing import Dict, List, Optional, Set, Any
from collections import Counter
import networkx as nx


class CollusionRingDetector:
    """
    Bipartite Graph Collusion Ring Detector using NetworkX[cite: 1, 3, 4].
    Maintains officer-garage claim frequencies and computes Jaccard similarity 
    strictly across each officer's top-10 repair garages.
    """
    def __init__(self, jaccard_threshold: float = 0.60):
        self.jaccard_threshold = jaccard_threshold
        self.graph = nx.Graph()
        self.next_ring_counter = 1
        
        # Frequency tracker: officer_node -> Counter({garage_node: claim_count})
        self.officer_garage_counts: Dict[str, Counter] = {}
        
        # Persistent ring mapping: node -> ring_id
        self.officer_to_ring: Dict[str, str] = {}
        self.ring_to_officers: Dict[str, Set[str]] = {}

    @staticmethod
    def _normalize_officer_node(officer_id: str) -> str:
        return officer_id if officer_id.startswith("OFF-") else f"OFF-{officer_id}"

    @staticmethod
    def _normalize_garage_node(garage_id: str) -> str:
        return garage_id if garage_id.startswith("GAR-") else f"GAR-{garage_id}"

    def add_claim(self, officer_id: str, garage_id: str):
        off_node = self._normalize_officer_node(officer_id)
        gar_node = self._normalize_garage_node(garage_id)

        self.graph.add_node(off_node, bipartite=0)
        self.graph.add_node(gar_node, bipartite=1)
        self.graph.add_edge(off_node, gar_node)

        if off_node not in self.officer_garage_counts:
            self.officer_garage_counts[off_node] = Counter()
        self.officer_garage_counts[off_node][gar_node] += 1

    def get_top_10_garages(self, off_node: str) -> Set[str]:
        if off_node not in self.officer_garage_counts:
            return set()
        top_pairs = self.officer_garage_counts[off_node].most_common(10)
        return {gar_node for gar_node, _ in top_pairs}

    def evaluate_rings(self, flagged_officer_ids: List[str]):
        nodes = [
            self._normalize_officer_node(oid)
            for oid in flagged_officer_ids
            if self._normalize_officer_node(oid) in self.graph
        ]

        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                u1, u2 = nodes[i], nodes[j]
                
                # Compare strictly top-10 dominant garages per officer
                g1 = self.get_top_10_garages(u1)
                g2 = self.get_top_10_garages(u2)

                if not g1 or not g2:
                    continue

                intersection_len = len(g1 & g2)
                union_len = len(g1 | g2)
                jaccard = intersection_len / union_len

                if jaccard >= self.jaccard_threshold:
                    self._link_officers(u1, u2)

    def _link_officers(self, off_node1: str, off_node2: str):
        r1 = self.officer_to_ring.get(off_node1)
        r2 = self.officer_to_ring.get(off_node2)

        if not r1 and not r2:
            new_id = f"RING-{self.next_ring_counter:06d}"
            self.next_ring_counter += 1
            self.officer_to_ring[off_node1] = new_id
            self.officer_to_ring[off_node2] = new_id
            self.ring_to_officers[new_id] = {off_node1, off_node2}
        elif r1 and not r2:
            self.officer_to_ring[off_node2] = r1
            self.ring_to_officers[r1].add(off_node2)
        elif not r1 and r2:
            self.officer_to_ring[off_node1] = r2
            self.ring_to_officers[r2].add(off_node1)
        elif r1 != r2:
            primary_ring = min(r1, r2)
            secondary_ring = max(r1, r2)
            for member in self.ring_to_officers[secondary_ring]:
                self.officer_to_ring[member] = primary_ring
                self.ring_to_officers[primary_ring].add(member)
            del self.ring_to_officers[secondary_ring]

    def get_ring_id(self, officer_id: str) -> Optional[str]:
        off_node = self._normalize_officer_node(officer_id)
        return self.officer_to_ring.get(off_node, None)

    def get_real_ring_graph(self, officer_id: str) -> Dict[str, Any]:
        """Returns the real sub-graph around an officer for Person D's evidence PDF."""
        off_node = self._normalize_officer_node(officer_id)
        ring_id = self.get_ring_id(off_node)
        
        if not ring_id or off_node not in self.graph:
            return {
                "officer_id": off_node,
                "network_ring_id": None,
                "nodes": [off_node] if off_node in self.graph else [],
                "edges": [],
                "overlap_score": 0.0,
            }

        ring_members = self.ring_to_officers[ring_id]
        subgraph_nodes = set(ring_members)
        for member in ring_members:
            subgraph_nodes.update(self.get_top_10_garages(member))

        subgraph = self.graph.subgraph(subgraph_nodes)
        edges = [[u, v] for u, v in subgraph.edges()]

        return {
            "officer_id": off_node,
            "network_ring_id": ring_id,
            "nodes": list(subgraph_nodes),
            "edges": edges,
            "overlap_score": self.jaccard_threshold,
        }
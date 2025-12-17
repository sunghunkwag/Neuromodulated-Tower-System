"""
SCIG: Self-Contracting Improvement Graph
A triple-recursive self-improvement loop:
  L0: Improve the artifact (a program in a small DSL)
  L1: Improve the improver (operator selection policy)
  L2: Improve the evaluator (test distribution + contracts/invariants)

This is NOT a gradient-based meta-RL / MAML clone.
It is a contract-guided improvement graph with adversarial test forging and self-tuning patch policies.

Run:
  python scig_demo.py
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# -----------------------------
# 0) Small DSL: Expression Trees
# -----------------------------


class Node:
    def eval(self, x: float) -> float:
        raise NotImplementedError

    def clone(self) -> "Node":
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def walk(self) -> List["Node"]:
        raise NotImplementedError

    def replace_child(self, old: "Node", new: "Node") -> bool:
        """Replace a descendant node in-place; returns True if replaced."""
        return False

    def to_str(self) -> str:
        raise NotImplementedError


@dataclass
class Var(Node):
    def eval(self, x: float) -> float:
        return x

    def clone(self) -> "Node":
        return Var()

    def size(self) -> int:
        return 1

    def walk(self) -> List["Node"]:
        return [self]

    def to_str(self) -> str:
        return "x"


@dataclass
class Const(Node):
    c: float

    def eval(self, x: float) -> float:  # noqa: ARG002
        return self.c

    def clone(self) -> "Node":
        return Const(self.c)

    def size(self) -> int:
        return 1

    def walk(self) -> List["Node"]:
        return [self]

    def to_str(self) -> str:
        # compact formatting
        return f"{self.c:.4g}"


@dataclass
class Unary(Node):
    op: str
    a: Node

    def eval(self, x: float) -> float:
        v = self.a.eval(x)
        if self.op == "neg":
            return -v
        if self.op == "sin":
            return math.sin(v)
        if self.op == "cos":
            return math.cos(v)
        if self.op == "tanh":
            return math.tanh(v)
        if self.op == "abs":
            return abs(v)
        if self.op == "log1p":
            # safe log1p
            return math.log1p(max(v, -0.999999))
        raise ValueError(f"Unknown unary op: {self.op}")

    def clone(self) -> "Node":
        return Unary(self.op, self.a.clone())

    def size(self) -> int:
        return 1 + self.a.size()

    def walk(self) -> List["Node"]:
        return [self] + self.a.walk()

    def replace_child(self, old: Node, new: Node) -> bool:
        if self.a is old:
            self.a = new
            return True
        return self.a.replace_child(old, new)

    def to_str(self) -> str:
        return f"{self.op}({self.a.to_str()})"


@dataclass
class Binary(Node):
    op: str
    a: Node
    b: Node

    def eval(self, x: float) -> float:
        va = self.a.eval(x)
        vb = self.b.eval(x)
        if self.op == "add":
            return va + vb
        if self.op == "sub":
            return va - vb
        if self.op == "mul":
            return va * vb
        if self.op == "div":
            # safe division
            den = vb if abs(vb) > 1e-9 else (1e-9 if vb >= 0 else -1e-9)
            return va / den
        if self.op == "max":
            return max(va, vb)
        if self.op == "min":
            return min(va, vb)
        raise ValueError(f"Unknown binary op: {self.op}")

    def clone(self) -> "Node":
        return Binary(self.op, self.a.clone(), self.b.clone())

    def size(self) -> int:
        return 1 + self.a.size() + self.b.size()

    def walk(self) -> List["Node"]:
        return [self] + self.a.walk() + self.b.walk()

    def replace_child(self, old: Node, new: Node) -> bool:
        if self.a is old:
            self.a = new
            return True
        if self.b is old:
            self.b = new
            return True
        return self.a.replace_child(old, new) or self.b.replace_child(old, new)

    def to_str(self) -> str:
        return f"{self.op}({self.a.to_str()}, {self.b.to_str()})"


# -----------------------------
# 1) Contracts (Invariants) + Violations
# -----------------------------


@dataclass
class Contract:
    """
    A contract is a check that must hold on probe inputs.
    This is deliberately generic: prevents "improvements" that cheat by exploding/NaN-ing, etc.
    You can extend with domain-specific invariants.
    """

    name: str
    check: Callable[[Callable[[float], float], List[float]], bool]


def contract_finite_and_bounded(bound: float = 1e6) -> Contract:
    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            for x in xs:
                y = fn(x)
                if not math.isfinite(y):
                    return False
                if abs(y) > bound:
                    return False
            return True
        except Exception:
            return False

    return Contract(name=f"finite_and_bounded({bound})", check=_check)


def contract_lipschitz_soft(max_slope: float = 1e4) -> Contract:
    """
    A soft Lipschitz bound to block pathological spikes that pass sparse tests.
    """

    def _check(fn: Callable[[float], float], xs: List[float]) -> bool:
        try:
            xs_sorted = sorted(xs)
            ys = [fn(x) for x in xs_sorted]
            for i in range(1, len(xs_sorted)):
                dx = xs_sorted[i] - xs_sorted[i - 1]
                if dx <= 0:
                    continue
                dy = abs(ys[i] - ys[i - 1])
                if dy / dx > max_slope:
                    return False
            return True
        except Exception:
            return False

    return Contract(name=f"lipschitz_soft({max_slope})", check=_check)


# -----------------------------
# 2) Test Forge (Evaluator evolves)
# -----------------------------


@dataclass
class TestForge:
    """
    L2 recursion: the test distribution adapts to expose regressions and exploit disagreements.
    """

    domain: Tuple[float, float] = (-3.0, 3.0)
    base_n: int = 64
    adversarial_n: int = 64
    focus_strength: float = 0.5  # updated by RSI

    def sample_base(self) -> List[float]:
        lo, hi = self.domain
        return [random.uniform(lo, hi) for _ in range(self.base_n)]

    def sample_adversarial(self, fns: List[Callable[[float], float]]) -> List[float]:
        """
        Generate points where candidate functions disagree most.
        """

        lo, hi = self.domain
        pool = [random.uniform(lo, hi) for _ in range(self.adversarial_n * 5)]
        scored: List[Tuple[float, float]] = []
        for x in pool:
            ys = []
            for fn in fns:
                try:
                    ys.append(fn(x))
                except Exception:
                    ys.append(float("nan"))
            # disagreement score: variance with NaN penalty
            if any(not math.isfinite(y) for y in ys):
                score = 1e9
            else:
                score = statistics.pvariance(ys) if len(ys) > 1 else 0.0
            scored.append((score, x))
        scored.sort(reverse=True, key=lambda t: t[0])
        top = [x for _, x in scored[: self.adversarial_n]]

        # mix with a little random to avoid overfitting tests
        mix_n = int(self.adversarial_n * self.focus_strength)
        rand_n = self.adversarial_n - mix_n
        extra = [random.uniform(lo, hi) for _ in range(rand_n)]
        return top[:mix_n] + extra

    def update_focus(self, signal: float) -> None:
        """
        RSI on L2: make tests more adversarial if we detect 'cheating' or instability,
        less adversarial if search is stagnating.
        """

        # clamp
        self.focus_strength = min(0.95, max(0.05, self.focus_strength + 0.1 * signal))


# -----------------------------
# 3) Patch Forge (Improver evolves)
# -----------------------------


UNARY_OPS = ["neg", "sin", "cos", "tanh", "abs", "log1p"]
BINARY_OPS = ["add", "sub", "mul", "div", "max", "min"]


def random_leaf() -> Node:
    if random.random() < 0.5:
        return Var()
    return Const(random.uniform(-2.0, 2.0))


def random_tree(max_depth: int = 4) -> Node:
    if max_depth <= 0:
        return random_leaf()
    r = random.random()
    if r < 0.35:
        return random_leaf()
    if r < 0.60:
        return Unary(random.choice(UNARY_OPS), random_tree(max_depth - 1))
    return Binary(random.choice(BINARY_OPS), random_tree(max_depth - 1), random_tree(max_depth - 1))


def simplify(node: Node) -> Node:
    """
    Tiny algebraic simplifier (keeps it safe and generic).
    """

    if isinstance(node, Unary):
        node.a = simplify(node.a)
        # double neg
        if node.op == "neg" and isinstance(node.a, Unary) and node.a.op == "neg":
            return node.a.a
        return node
    if isinstance(node, Binary):
        node.a = simplify(node.a)
        node.b = simplify(node.b)
        # constant folding
        if isinstance(node.a, Const) and isinstance(node.b, Const):
            try:
                return Const(node.eval(0.0))
            except Exception:
                return node
        # mul by 1 / add 0
        if node.op == "mul":
            if isinstance(node.a, Const) and abs(node.a.c - 1.0) < 1e-9:
                return node.b
            if isinstance(node.b, Const) and abs(node.b.c - 1.0) < 1e-9:
                return node.a
            if isinstance(node.a, Const) and abs(node.a.c) < 1e-9:
                return Const(0.0)
            if isinstance(node.b, Const) and abs(node.b.c) < 1e-9:
                return Const(0.0)
        if node.op == "add":
            if isinstance(node.a, Const) and abs(node.a.c) < 1e-9:
                return node.b
            if isinstance(node.b, Const) and abs(node.b.c) < 1e-9:
                return node.a
        return node
    return node


@dataclass
class OperatorStats:
    """
    L1 recursion: self-tuning operator selection policy (not gradient RL, not MAML).
    Uses credibility-weighted success tracking.
    """

    name: str
    tries: int = 0
    wins: int = 0
    avg_gain: float = 0.0

    def score(self) -> float:
        # optimism under uncertainty + gain
        # (small epsilon to avoid zero division)
        t = self.tries + 1
        w = self.wins + 1
        win_rate = w / t
        return 0.7 * win_rate + 0.3 * (self.avg_gain)


class PatchForge:
    """
    A library of micro-patch operators that can be combined.
    The selection distribution is self-tuned (L1 recursion).
    """

    def __init__(self) -> None:
        self.ops: Dict[str, Callable[[Node], Node]] = {
            "mutate_const": self._mutate_const,
            "replace_subtree": self._replace_subtree,
            "wrap_unary": self._wrap_unary,
            "wrap_binary": self._wrap_binary,
            "swap_children": self._swap_children,
            "simplify": self._simplify,
        }
        self.stats: Dict[str, OperatorStats] = {k: OperatorStats(k) for k in self.ops.keys()}

    def pick_operator(self) -> str:
        # weighted by stats.score, but keep exploration
        names = list(self.ops.keys())
        raw = []
        for n in names:
            raw.append(max(0.01, self.stats[n].score()))
        s = sum(raw)
        probs = [r / s for r in raw]
        return random.choices(names, weights=probs, k=1)[0]

    def apply(self, node: Node) -> Tuple[str, Node]:
        op_name = self.pick_operator()
        new_node = self.ops[op_name](node.clone())
        return op_name, new_node

    def report(self, op_name: str, improved: bool, gain: float) -> None:
        st = self.stats[op_name]
        st.tries += 1
        if improved:
            st.wins += 1
            # running average
            st.avg_gain = (st.avg_gain * (st.wins - 1) + gain) / st.wins

    # ----- operators -----

    def _mutate_const(self, node: Node) -> Node:
        all_nodes = node.walk()
        consts = [n for n in all_nodes if isinstance(n, Const)]
        if not consts:
            return node
        c = random.choice(consts)
        c.c += random.uniform(-1.0, 1.0) * (0.25 + random.random())
        return simplify(node)

    def _replace_subtree(self, node: Node) -> Node:
        all_nodes = node.walk()
        target = random.choice(all_nodes)
        repl = random_tree(max_depth=random.randint(1, 3))
        if target is node:
            return simplify(repl)
        node.replace_child(target, repl)
        return simplify(node)

    def _wrap_unary(self, node: Node) -> Node:
        all_nodes = node.walk()
        target = random.choice(all_nodes)
        wrapped = Unary(random.choice(UNARY_OPS), target.clone())
        if target is node:
            return simplify(wrapped)
        node.replace_child(target, wrapped)
        return simplify(node)

    def _wrap_binary(self, node: Node) -> Node:
        all_nodes = node.walk()
        target = random.choice(all_nodes)
        other = random_tree(max_depth=2)
        if random.random() < 0.5:
            wrapped = Binary(random.choice(BINARY_OPS), target.clone(), other)
        else:
            wrapped = Binary(random.choice(BINARY_OPS), other, target.clone())
        if target is node:
            return simplify(wrapped)
        node.replace_child(target, wrapped)
        return simplify(node)

    def _swap_children(self, node: Node) -> Node:
        bins = [n for n in node.walk() if isinstance(n, Binary)]
        if not bins:
            return node
        b = random.choice(bins)
        b.a, b.b = b.b, b.a
        return simplify(node)

    def _simplify(self, node: Node) -> Node:  # noqa: PLR6301
        return simplify(node)


# -----------------------------
# 4) Version Graph (SCIG core)
# -----------------------------


@dataclass
class Version:
    id: int
    root: Node
    score: float
    complexity: int
    lineage: List[int] = field(default_factory=list)
    note: str = ""


@dataclass
class SCIG:
    """
    The SCIG loop:
      - Keeps an archive (graph) of versions
      - Induces contracts and enforces them
      - Evolves tests (adversarial forging)
      - Evolves operator policy (self-tuning)
    """

    target_fn: Callable[[float], float]
    contracts: List[Contract]
    testforge: TestForge
    patchforge: PatchForge
    max_complexity: int = 60
    accept_margin: float = 1e-6
    rng_seed: int = 7

    archive: List[Version] = field(default_factory=list)
    next_id: int = 0

    def __post_init__(self) -> None:
        random.seed(self.rng_seed)

    def _callable(self, root: Node) -> Callable[[float], float]:
        def fn(x: float) -> float:
            return root.eval(x)

        return fn

    def _loss(self, fn: Callable[[float], float], xs: List[float]) -> float:
        # robust loss (Huber-ish) to avoid outlier spikes skewing everything
        err = 0.0
        for x in xs:
            y = fn(x)
            t = self.target_fn(x)
            d = y - t
            ad = abs(d)
            if ad < 1.0:
                err += 0.5 * d * d
            else:
                err += ad - 0.5
        return err / max(1, len(xs))

    def _passes_contracts(self, fn: Callable[[float], float], probes: List[float]) -> bool:
        return all(c.check(fn, probes) for c in self.contracts)

    def _score(self, root: Node, train_xs: List[float], val_xs: List[float]) -> Tuple[float, float, int]:
        fn = self._callable(root)
        # enforce contracts on union probes
        probes = list(set(train_xs + val_xs))
        if not self._passes_contracts(fn, probes):
            return float("inf"), float("inf"), root.size()

        train = self._loss(fn, train_xs)
        val = self._loss(fn, val_xs)

        # complexity-regularized score
        complexity = root.size()
        reg = 0.001 * complexity
        return train + reg, val + reg, complexity

    def seed(self, n0: int = 8) -> None:
        base_xs = self.testforge.sample_base()
        # start with multiple seeds for diversity
        for _ in range(n0):
            r = random_tree(max_depth=4)
            train_s, val_s, comp = self._score(r, base_xs, base_xs)
            self.archive.append(
                Version(id=self.next_id, root=r, score=val_s, complexity=comp, lineage=[], note="seed")
            )
            self.next_id += 1
        self.archive.sort(key=lambda v: v.score)

    def _select_parent(self) -> Version:
        """
        Select parent from archive:
          - prefer low score but keep novelty via stochasticity
        """

        self.archive.sort(key=lambda v: v.score)
        topk = self.archive[: max(3, min(12, len(self.archive)))]
        # softmax over inverse score
        weights = []
        for v in topk:
            s = v.score
            w = 1.0 / (1e-6 + s)
            weights.append(w)
        return random.choices(topk, weights=weights, k=1)[0]

    def run(self, steps: int = 200, proposals_per_step: int = 8) -> Version:
        if not self.archive:
            self.seed()

        best = min(self.archive, key=lambda v: v.score)

        stagnation = 0
        for _ in range(steps):
            parent = self._select_parent()

            # L2: forge tests using disagreement among top variants
            top_fns = [self._callable(v.root) for v in self.archive[: min(6, len(self.archive))]]
            train_xs = self.testforge.sample_base()
            adv_xs = self.testforge.sample_adversarial(top_fns)
            val_xs = list(set(train_xs + adv_xs))

            parent_train, parent_val, _ = self._score(parent.root, train_xs, val_xs)
            parent_score = parent_val

            improved_any = False
            best_child: Optional[Version] = None

            for _ in range(proposals_per_step):
                op_name, child_root = self.patchforge.apply(parent.root)

                # complexity gate
                if child_root.size() > self.max_complexity:
                    self.patchforge.report(op_name, improved=False, gain=0.0)
                    continue

                child_train, child_val, child_comp = self._score(child_root, train_xs, val_xs)
                if not math.isfinite(child_val):
                    self.patchforge.report(op_name, improved=False, gain=0.0)
                    continue

                gain = max(0.0, parent_score - child_val)

                # Accept if improves beyond margin, and doesn't overfit train-vs-val gap too badly
                # (a cheap anti-cheat heuristic)
                overfit_gap = child_train - child_val
                ok_overfit = overfit_gap < 0.05  # very conservative

                if (child_val + self.accept_margin) < parent_score and ok_overfit:
                    improved_any = True
                    self.patchforge.report(op_name, improved=True, gain=gain)
                    v = Version(
                        id=self.next_id,
                        root=child_root,
                        score=child_val,
                        complexity=child_comp,
                        lineage=parent.lineage + [parent.id],
                        note=f"op={op_name}, gain={gain:.5f}",
                    )
                    self.next_id += 1
                    best_child = v if (best_child is None or v.score < best_child.score) else best_child
                else:
                    self.patchforge.report(op_name, improved=False, gain=gain)

            if improved_any and best_child is not None:
                self.archive.append(best_child)
                self.archive.sort(key=lambda v: v.score)
                if best_child.score < best.score:
                    best = best_child
                stagnation = 0

                # L2 recursion: if we improved, slightly reduce adversarial focus (explore broader)
                self.testforge.update_focus(signal=-0.5)
            else:
                stagnation += 1
                # L2 recursion: if stagnating, increase adversarial focus to break plateaus
                self.testforge.update_focus(signal=+0.5)

            # Keep archive bounded (graph pruning)
            if len(self.archive) > 80:
                self.archive.sort(key=lambda v: v.score)
                self.archive = self.archive[:80]

            # Optional: if stagnation is long, inject a novel seed (escape local minima)
            if stagnation > 25:
                r = random_tree(max_depth=5)
                train_s, val_s, comp = self._score(r, train_xs, val_xs)
                self.archive.append(Version(self.next_id, r, val_s, comp, [], "novel_seed"))
                self.next_id += 1
                stagnation = 0

        return best


# -----------------------------
# 5) Demo target (black-box "world")
# -----------------------------


def hidden_target(x: float) -> float:
    """
    Unknown to the improver; stands for the environment / ground truth.
    """

    return 0.7 * math.sin(1.3 * x) + 0.2 * x * x - 0.1 * x + 0.05 * math.cos(3.0 * x)


# -----------------------------
# 6) Entrypoint
# -----------------------------


def main() -> None:
    contracts = [contract_finite_and_bounded(bound=1e6), contract_lipschitz_soft(max_slope=5e4)]

    scig = SCIG(
        target_fn=hidden_target,
        contracts=contracts,
        testforge=TestForge(domain=(-3.0, 3.0), base_n=64, adversarial_n=64, focus_strength=0.5),
        patchforge=PatchForge(),
        max_complexity=65,
        accept_margin=1e-6,
        rng_seed=11,
    )

    scig.seed(n0=10)
    best = scig.run(steps=250, proposals_per_step=10)

    print("=== SCIG RESULT ===")
    print(f"best_id: {best.id}")
    print(f"best_score: {best.score:.6f}")
    print(f"complexity: {best.complexity}")
    print(f"expr: {best.root.to_str()}")
    print()
    print("=== Operator Stats (self-tuned policy) ===")
    # show sorted by score
    stats = list(scig.patchforge.stats.values())
    stats.sort(key=lambda s: s.score(), reverse=True)
    for st in stats:
        print(
            f"{st.name:15s} tries={st.tries:4d} wins={st.wins:4d} "
            f"avg_gain={st.avg_gain:.6f} score={st.score():.6f}"
        )
    print()
    print(f"TestForge focus_strength (L2 recursion): {scig.testforge.focus_strength:.3f}")


if __name__ == "__main__":
    main()

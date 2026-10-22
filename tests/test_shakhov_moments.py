#!/usr/bin/env python3
"""Independent Gaussian moment checks of the actual bundled kernel weights."""
from __future__ import annotations

import ast
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def weight_expression(path, function_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == function_name)
    assignments = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node)
    name = "weight" if "weight" in assignments else "w"
    weight = next(n for n in assignments[name]
                  if "shape" in {x.id for x in ast.walk(n.value)
                                  if isinstance(x, ast.Name)})
    required = {"shape", "denominator", "denom", "correction_scale", "A"}
    nodes = []
    for key in required:
        if key in assignments:
            nodes.append(assignments[key][0])
    nodes.append(weight)
    nodes.sort(key=lambda n: n.lineno)
    return compile(ast.fix_missing_locations(ast.Module(body=nodes,
                                                       type_ignores=[])),
                   str(path), "exec"), name


for path, function in (
    (ROOT / "solver/JFM_bgk_shakhov_quarter.py", "collide_shakhov_relaxation"),
    (ROOT / "solver/JFM_reviewer_quarter_domain.py", "collision_shakhov_cellwise"),
):
    code, name = weight_expression(path, function)
    for order in (8, 16):
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        axis = np.sqrt(2.0) * nodes
        one_d = weights / np.sqrt(np.pi)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        wx, wy, wz = np.meshgrid(one_d, one_d, one_d, indexing="ij")
        measure = wx * wy * wz
        c2 = x*x + y*y + z*z
        for pr in (1.0, 2.0/3.0, 0.5):
            q = np.array([1e-4, -2e-4, 1.5e-4])
            dot = x*q[0] + y*q[1] + z*q[2]
            env = dict(c2=c2, theta=1.0, pressure=1.0, ploc=1.0,
                       c_dot_q=dot, cdotq=dot, prandtl=pr, Pr=pr)
            exec(code, {}, env)
            weight = env[name]
            assert np.all((weight > 0.0) & (weight < 2.0)), "unclipped regime"
            moment = lambda value: np.sum(measure * weight * value)
            assert np.isclose(moment(np.ones_like(c2)), 1.0, atol=2e-14)
            assert np.isclose(moment(0.5*c2), 1.5, atol=2e-14)
            for component, q_component in zip((x, y, z), q):
                assert np.isclose(moment(component), 0.0, atol=2e-14)
                target_heat_flux = moment(0.5*c2*component)
                # Physical Shakhov target is (1-Pr)*q, independently of
                # the source's algebraic representation of its polynomial.
                assert np.isclose(target_heat_flux, (1.0-pr)*q_component,
                                  rtol=1e-8, atol=2e-14), (
                                      path.name, order, pr, target_heat_flux)
    print(f"[OK] {path.name}: mass, momentum, energy and heat-flux moments")

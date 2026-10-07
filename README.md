# mechcalc-mcp

An MCP server of deterministic engineering checks for people building robots and 3D-printed hardware
with AI agents. A coding agent can model a part in build123d, Blender or FreeCAD and then ask this
server whether the shoulder servo can hold the arm out, whether the leadscrew back-drives, whether the
base tips over at full reach, or what diameter to model so a printed hole fits a 3 mm pin.

Every tool is a closed-form calculation or a table lookup. Nothing calls a model or the network, so the
same input always gives the same number, and each result lists what it did not check.

## Install

```bash
pip install -e .            # from a clone; needs only the `mcp` package
```

Claude Code:

```bash
claude mcp add mechcalc -- mechcalc-mcp
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.mechcalc]
command = "mechcalc-mcp"
```

Any other MCP client: run `mechcalc-mcp` over stdio.

## Tools

| Area | Tools |
|---|---|
| Whole machines | `engine_list`, `engine_analyze` (arm, vehicle, cnc, multirotor, legged, humanoid, linear actuator, self-balancing, vacuum/jamming gripper) |
| Mechanisms | `leadscrew`, `leadscrew_compare`, `lever_actuator`, `arm_workspace`, `tipover`, `drive_sizing`, `gear_mesh`, `planetary_design`, `planetary_suggest` |
| Power and mass | `power_check`, `mass_aggregate`, `battery_mass` |
| Printing | `print_hole`, `print_shaft`, `print_insert_boss`, `print_wall`, `calibrate_plan`, `calibrate_record_hole` |
| Structure | `fea_beam`, `fea_column`, `fea_bolt`, `fea_thermal` |
| Materials | `materials_list`, `material_get` |
| Wearables | `wearable_pressure`, `wearable_capacity` |

Example, a five-servo arm with a 250 g payload: `engine_analyze("arm", {"reach_mm": 380, "payload_g": 250,
"distal_masses": [[55, 280], [55, 330]]})` returns shoulder torque, available torque and margin, tip
deflection, link root stress, safety factor and first bending mode.

## Calibration

Datasheet numbers describe moulded plastic, and a printed part is a stack of extrusions. `calibrate_plan`
says which coupons to print; `calibrate_record_hole` stores what you measured in `~/.mechcalc` (override
with `MECHCALC_DATA_DIR`). From then on `print_hole`, `print_shaft` and `material_get` use your
machine's numbers and say so in their `source` field.

## Limits

These are first-principles screens, good for catching a design that cannot work before anything is
printed or bought. They do not replace a meshed FE solve for complex parts, a dynamics simulation, or a
measurement on the bench. Every threshold is in the code next to the reason for it, so a wrong
assumption is easy to find and fix.

## Contributing

Issues and pull requests are welcome, especially new engines, measured material data, and test cases
that show a check giving the wrong answer. Run `pytest` before opening a PR; every bug fix should come
with the test that would have caught it.

## License

Apache-2.0.

# Matching summary (`20261006_192846_node2_white_square_clean_node3_small_hello_kitty_color_grid_clean_node4_white_grid_green_square_clean`)

- method: `method1_mean_diff`
- target_node: `node_2`
- target_attacks: `node_2/00_white_square`
- tau_same: `0.7`

## Primary transfer

- primary_transfer_node: **`node_3`**
- primary_transfer_attack: **`node_3/00_small_hello_kitty`**
- coverage (novelty_score): `0.056248` (lower = more novel; threshold `0.7`)

## Candidate ranking (by Coverage ascending)

| attack | node | Coverage | novel? |
|---|---|---:|:---:|
| `node_3/00_small_hello_kitty` | `node_3` | 0.056248 | yes |
| `node_4/01_green_square` | `node_4` | 0.504023 | yes |
| `node_3/01_color_grid` | `node_3` | 0.620085 | yes |
| `node_4/00_white_grid` | `node_4` | 0.671677 | yes |


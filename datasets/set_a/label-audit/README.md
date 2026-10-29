# Label audit of the text corpus (appendix A)

## Source

Our own audit. Two input files, copied here:
- `label-audit-wave1.json` — copy of our wave-1 audit file (48 items, 8 Oct 2026). Changes in the copy
  (recorded in its `_note_copy` field): the gpqa:17 old/new gold converted from hashed option ids to letters (A -> C; A is the gold that the
  item-building script's shuffle gives, C is the letter of the corrected option, recovered per item from the live corpus traces), and every GPQA reason replaced by
  `(GPQA: reason withheld, dataset terms)`.
- `audit2-wave2.json` — copy of our wave-2 audit file (20 items, 17 Oct 2026), unchanged (it has no reasons).

## License and redistribution

Our work. GPQA reasons are withheld (GPQA asks that its questions not be revealed); no question text of any set is included.

## What paper 1 says (https://github.com/Alatip/paper-1)

Appendix A, table A.1 (wave 1 / wave 2): wrong gold 21 / 2, grader error 2 / 1, excluded 22 / 14, models really wrong 3 / 3 (2 of them from wave 1).
Section 4.1 of paper 1.

## Files

| file | content |
|---|---|
| `label-audit.csv` | 68 rows (66 items): item_id, dataset, wave, decision, old_gold, new_gold, reason |

In this copy the gpqa:17 gold is stored as letters (`A,C`); the wave-2 corrected golds gsmhard:6235500 and gsmhard:40949084
have `old_gold` (from `gold_benchmark` in `../corpus-gsm-hard/items.csv`).

# LinkerHand L20 to FR3 adapter

`L20_FR3_adapter.stl` is the millimetre runtime mesh generated from the local
`g20法兰.STEP` source. The source STEP was inspected during conversion but is
not retained in the demo asset bundle. Its product name identifies it as
`L20-UR7E转换法兰`.

Measured CAD properties:

- overall envelope: 70 x 70 x 66 mm;
- robot-side pattern: four 6.6 mm through holes on a 50 mm pitch circle;
- hand-side machining: LinkerHand L20 interface;
- one watertight solid, approximately 73,366.5 mm3.

This is the selected part for the mixed-end-effector robot's **right** arm:
that side retains the repository's LinkerHand L20, whose MJCF places the hand
base 60 mm beyond the FR3 hand mount. The 66 mm adapter occupies that wrist
space. The left Sharpa Wave is directly flange-mounted and does not use it.

The other two files in the download directory were rejected:

- `o30法兰2.stp`: 70 x 70 x 63.33 mm, with a different four-hole hand-side
  pattern (3.5 mm holes at x/y = +/-10 mm), intended for the O30 family;
- `sharpa法兰官方.STEP`: 56.4 x 59.7 x 10 mm official Sharpa interface plate,
  suitable for a Wave-hand interface but not a replacement for the L20's
  approximately 60 mm wrist spacer.

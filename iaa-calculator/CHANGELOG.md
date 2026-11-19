# Changelog

Retrospective notes based on supplied source snapshots; original development dates and authorship were not provided.

- **V1** (`IAA_Calculator.py`): CSV categorical comparison, matching unit IDs, JSON agreement report and CSV disagreements.
- **V2**: Label Studio JSON, nesting-depth token scoring, span and relation comparisons, ambiguous token flags.
- **V3**: Stable IDs, transcript masking, UTF-16 offset conversion, relation direction and validation flags.
- **V4**: Normalizes whitespace when validating selected text. Internal docstring still says version 3; filename identifies this snapshot.
- **V5**: Handles escaped whitespace in selection validation; scores shared task IDs rather than requiring identical task sets.
- **V6**: Replaces layer-based kappa with one pooled matched-label-pair score; adds boundary and relationship difference reports.
- **V7**: Expands wrapper masking and pronunciation handling.
- **V8**: Stops masking standalone `[speaker]`, allowing it to be scored as an event marker.

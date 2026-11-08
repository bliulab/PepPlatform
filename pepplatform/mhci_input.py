"""Compatibility import for frozen experiment scripts, including runpy workers."""

if __package__:
    from .hlai_input import HLAIInputError as MHCIInputError
    from .hlai_input import require_hlai_partner_sequence as require_mhci_partner_sequence
else:
    # Filename-loaded workers must select their historical package themselves.
    import runpy
    from pathlib import Path

    _guard = runpy.run_path(str(Path(__file__).with_name("hlai_input.py")))
    MHCIInputError = _guard["HLAIInputError"]
    require_mhci_partner_sequence = _guard["require_hlai_partner_sequence"]

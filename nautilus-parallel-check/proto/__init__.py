"""Isolated nautilus speed-up prototype.

Nothing here is imported by ``src/gwemfish``. The prototype only ever *calls*
gwemfish's public builders and solver; it never patches, wraps or edits them, so
the fisher / hmc / deriv-approx paths -- and the autodiff and lens-equation
machinery they rely on -- execute exactly the code they do today.
"""

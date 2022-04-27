import picmistandard
from typeguard import typechecked


@typechecked
class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
    # note: is translated from outside, does not do any checks itself
    def check(self):
        """
        check validity of self

        if ok pass silently, raise on error
        """
        assert self.n_macroparticles_per_cell is not None, \
            "macroparticles per cell must be given"
        assert self.n_macroparticles is None, \
            "total number of macrosparticles not supported"

        assert self.n_macroparticles_per_cell > 0, \
            "at least one particle per cell required"

        # Note: Call PICMI check interface once available upstream

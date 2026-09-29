"""Optional neural quality backends for voicequal.

Everything in this package needs the ``neural`` extra::

    pip install 'voicequal[neural]'

The core library never imports from here. These backends exist so the
cheap DSP path can be compared against, and escalated to, the neural
predictors the speech-quality field uses as reference points.
"""

from voicequal.neural.dnsmos import DNSMOS, DNSMOSScores, dnsmos_available

__all__ = ["DNSMOS", "DNSMOSScores", "dnsmos_available"]

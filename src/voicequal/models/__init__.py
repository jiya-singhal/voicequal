"""Small learned models that ship inside the wheel and run in pure numpy.

Currently one model: :class:`QualityModel`, a distilled predictor of the
ITU-T P.835 scores (SIG, BAK, OVRL) trained on DNSMOS labels. It runs
in well under a millisecond after feature extraction and needs no
onnxruntime; an ONNX export of the same weights is provided for
browsers and other runtimes.
"""

from voicequal.models.quality import MOSEstimate, QualityModel, predict_mos

__all__ = ["MOSEstimate", "QualityModel", "predict_mos"]

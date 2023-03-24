"""
From: https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edematous/invaded tissue (ED — label 2),
and the necrotic tumor core (NCR — label 1), as described in the latest BraTS summarizing paper. The ground truth data
were created after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same
resolution (1 mm3) and skull-stripped.
"""

from enum import Enum


class TumorLabels(Enum):
    ET = 4
    ED = 2
    NCR = 1

from aics_tf_registration.core.alignment import perform_alignment
from skimage.io import imread
import numpy as np


def test_align_run():
    source = (
        imread(
            "./aics_tf_registration/tests/sample_images/H2B/H2B_20x_100x_source.tiff"
        )
        .squeeze()
        .astype(np.uint16)
    )
    target = (
        imread(
            "./aics_tf_registration/tests/sample_images/H2B/H2B_20x_100x_target.tiff"
        )
        .squeeze()
        .astype(np.uint16)
    )

    _, _, composite = perform_alignment(
        source,
        target,
        smaller_fov_modality="target",
        scale_factor_xy=0.3985,
        scale_factor_z=0.5472,
        source_alignment_channel=0,
        target_alignment_channel=0,
        source_output_channel=0,
        target_output_channel=0,
        prealign_z=False,
        denoise_z=False,
        use_refinement=False,
        save_composite=True,
    )

    assert composite is not None

import warnings

import numpy as np
import dask.array as da

from multiview_stitcher.io import METADATA_TRANSFORM_KEY
from multiview_stitcher import (
    msi_utils,
    param_utils,
    sample_data,
    registration,
    fusion,
    spatial_image_utils,
    ngff_utils,
)
import pytest

from napari.layers import Shapes
from napari_stitcher import viewer_utils, _utils


def _two_channel_data():
    shape = (224, 240)
    right_half = np.arange(shape[1]) >= shape[1] // 2

    return np.stack([
        np.broadcast_to(2 + right_half, shape).astype(np.uint8),
        np.broadcast_to(9 + right_half, shape).astype(np.uint8),
    ])


def _write_two_channel_ome_zarr(zarr_path):
    sim = spatial_image_utils.get_sim_from_array(
        da.from_array(_two_channel_data(), chunks=(1, 112, 120)),
        dims=['c', 'y', 'x'],
        scale={'y': 10.0, 'x': 20.0},
        translation={'y': 100.0, 'x': 200.0},
        c_coords=['DAPI', 'GFP'],
    )

    ngff_utils.write_sim_to_ome_zarr(
        sim,
        zarr_path,
        overwrite=True,
        show_progressbar=False,
    )


# def test_image_layer_to_msim_reads_actual_napari_ome_zarr(
#     tmp_path,
#     make_napari_viewer,
#     monkeypatch,
# ):
#     zarr_path = tmp_path / 'channels.ome.zarr'
#     _write_two_channel_ome_zarr(zarr_path)

#     viewer = make_napari_viewer()
#     layers = viewer.open(str(zarr_path), plugin='napari-ome-zarr')

#     assert len(layers) == 2
#     assert {
#         layer.name.split(': ')[-1]
#         for layer in layers
#     } == {'DAPI', 'GFP'}
#     assert all(
#         layer.source.reader_plugin == 'napari-ome-zarr'
#         for layer in layers
#     )
#     assert all(layer.multiscale for layer in layers)
#     assert all(
#         isinstance(ldata, da.Array)
#         for layer in layers
#         for ldata in layer.data
#     )

#     layer = next(
#         layer for layer in layers if layer.name.split(': ')[-1] == 'GFP')
#     layer.scale = np.asarray((2.0, 3.0))
#     layer.translate = np.asarray((5.0, 7.0))

#     calls = []
#     original_read_msim = _utils.ngff_utils.read_msim_from_ome_zarr

#     def read_msim_from_ome_zarr_spy(zarr_path, *args, **kwargs):
#         calls.append(zarr_path)
#         return original_read_msim(zarr_path, *args, **kwargs)

#     monkeypatch.setattr(
#         _utils.ngff_utils,
#         'read_msim_from_ome_zarr',
#         read_msim_from_ome_zarr_spy,
#     )

#     with warnings.catch_warnings(record=True) as recorded_warnings:
#         warnings.simplefilter('always')
#         msim = viewer_utils.image_layer_to_msim(layer, viewer)

#     assert [str(path) for path in calls] == [str(zarr_path)]
#     assert not any(
#         'Could not load napari-ome-zarr layer' in str(warning.message)
#         for warning in recorded_warnings
#     )

#     sim0 = msi_utils.get_sim_from_msim(msim, scale='scale0')
#     sim1 = msi_utils.get_sim_from_msim(msim, scale='scale1')

#     assert list(sim0.coords['c'].values) == ['GFP']
#     np.testing.assert_array_equal(
#         np.asarray(sim0.sel(c='GFP').isel(t=0).data),
#         _two_channel_data()[1],
#     )
#     assert spatial_image_utils.get_origin_from_sim(sim0) == {
#         'y': 5.0,
#         'x': 7.0,
#     }
#     assert spatial_image_utils.get_spacing_from_sim(sim0) == {
#         'y': 2.0,
#         'x': 3.0,
#     }
#     assert spatial_image_utils.get_spacing_from_sim(sim1) == {
#         'y': 4.0,
#         'x': 6.0,
#     }


@pytest.mark.parametrize(
    "ndim, N_c, N_t", [
        [ndim, N_c, N_t]
        for ndim in [2, 3]
        for N_c in [1, 2]
        for N_t in [1, 2]
    ]
)
def test_create_image_layer_tuples_from_msims(ndim, N_c, N_t, make_napari_viewer):
    """
    Basic test: scroll through time and confirm that no error is thrown.
    """

    viewer = make_napari_viewer()

    tiles_x, tiles_y, tiles_z = 2, 1, 1

    sims = sample_data.generate_tiled_dataset(ndim=ndim, N_t=N_t, N_c=N_c,
            tile_size=5, tiles_x=tiles_x, tiles_y=tiles_y, tiles_z=tiles_z)
    
    msims = [msi_utils.get_msim_from_sim(
        sim,
        scale_factors=[2]) for sim in sims]

    registered_transform_key = 'affine_registered'
    registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key=registered_transform_key,
        reg_channel_index=0
        )
    
    # Fuse msims directly so the result is also multiscale
    mfused = fusion.fuse(images=msims, transform_key=registered_transform_key)
    
    lds = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=registered_transform_key)
    assert len(lds) == N_c * tiles_x * tiles_y * tiles_z
    viewer_utils.add_image_layer_tuples_to_viewer(
        viewer, lds, manage_viewer_transformations=True)

    lds = viewer_utils.create_image_layer_tuples_from_msims(
        [mfused], transform_key=registered_transform_key)
    assert len(lds) == N_c

    viewer_utils.add_image_layer_tuples_to_viewer(
        viewer, lds, manage_viewer_transformations=True)

    # wiggle time
    if N_t > 1:
        current_step = list(viewer.dims.current_step)
        current_step[0] = current_step[0] + 1
        viewer.dims.current_step = tuple(current_step)


@pytest.mark.parametrize("ndim", [2, 3])
def test_create_shape_layer_tuples_from_msims(ndim, monkeypatch):
    spatial_dims = ["z", "y", "x"][-ndim:]
    spatial_shape = (2, 3, 4)[-ndim:]
    scale = dict(zip(spatial_dims, (2.0, 3.0, 4.0)[-ndim:]))
    translation = dict(zip(spatial_dims, (5.0, 6.0, 7.0)[-ndim:]))

    sim = spatial_image_utils.get_sim_from_array(
        da.zeros(spatial_shape),
        dims=spatial_dims,
        scale=scale,
        translation=translation,
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

    # Include off-diagonal terms so the test covers the full affine rather
    # than translation alone.
    affine = np.eye(ndim + 1)
    affine[0, 1] = 0.25
    affine[1, 0] = -0.5
    affine[:-1, -1] = np.arange(ndim) + 10
    transform_key = "affine_test"
    msi_utils.set_affine_transform(
        msim,
        param_utils.affine_to_xaffine(affine),
        transform_key=transform_key,
    )

    layer_tuples = viewer_utils.create_shape_layer_tuples_from_msims(
        [msim, msim],
        transform_key=transform_key,
        colormaps=["red", "green"],
        name_prefix="tile_bounds",
        shape_kwargs={"edge_width": 3, "opacity": 0.5},
    )

    assert len(layer_tuples) == 2
    lines, kwargs, layer_type = layer_tuples[0]
    _, second_kwargs, _ = layer_tuples[1]
    assert layer_type == "shapes"
    assert kwargs["shape_type"] == "line"
    assert kwargs["name"] == "tile_bounds_000"
    assert kwargs["edge_width"] == 3
    assert kwargs["opacity"] == 0.5
    assert second_kwargs["name"] == "tile_bounds_001"
    np.testing.assert_allclose(kwargs["edge_color"], [1, 0, 0, 1])
    np.testing.assert_allclose(second_kwargs["edge_color"], [0, 1, 0, 1])
    assert len(lines) == (4 if ndim == 2 else 12)

    corner_indices = np.array(list(np.ndindex(tuple([2] * ndim))))
    origin = np.array([translation[dim] for dim in spatial_dims])
    spacing = np.array([scale[dim] for dim in spatial_dims])
    corners = corner_indices * (np.array(spatial_shape) - 1) * spacing + origin
    transformed_corners = (
        affine @ np.column_stack([corners, np.ones(len(corners))]).T
    ).T[:, :-1]
    expected_lines = [
        transformed_corners[[i, j]]
        for i in range(len(corner_indices))
        for j in range(i + 1, len(corner_indices))
        if np.sum(np.abs(corner_indices[i] - corner_indices[j])) == 1
    ]
    np.testing.assert_allclose(lines, expected_lines)

    # Confirm that the returned LayerDataTuple kwargs are accepted by napari.
    layer = Shapes(lines, **kwargs)
    assert len(layer.data) == len(expected_lines)

    positional_color_calls = []

    def get_greedy_colors(sims, n_colors, transform_key):
        positional_color_calls.append((sims, n_colors, transform_key))
        return {0: 0, 1: 1}

    monkeypatch.setattr(
        viewer_utils.mv_graph,
        "get_greedy_colors",
        get_greedy_colors,
    )
    positional_tuples = viewer_utils.create_shape_layer_tuples_from_msims(
        [msim, msim],
        transform_key=transform_key,
        n_colors=2,
    )
    assert len(positional_color_calls) == 1
    _, called_n_colors, called_transform_key = positional_color_calls[0]
    assert called_n_colors == 2
    assert called_transform_key == transform_key
    assert positional_tuples[0][1]["edge_color"] == "#E69F00"
    assert positional_tuples[1][1]["edge_color"] == "#56B4E9"

    black_tuples = viewer_utils.create_shape_layer_tuples_from_msims(
        [msim, msim],
        transform_key=transform_key,
        use_positional_colors=False,
    )
    assert all(
        kwargs["edge_color"] == "black" for _, kwargs, _ in black_tuples
    )

    with pytest.raises(ValueError, match="one colormap per msim"):
        viewer_utils.create_shape_layer_tuples_from_msims(
            [msim, msim],
            transform_key=transform_key,
            colormaps=["red"],
        )

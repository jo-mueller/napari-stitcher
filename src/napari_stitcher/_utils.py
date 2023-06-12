import numpy as np
import xarray as xr

from mvregfus import io_utils
from mvregfus.image_array import ImageArray

from dask import delayed, compute
import dask.array as da
from tqdm.dask import TqdmCallback

from napari.utils import progress


def load_tiles(view_dict: dict,
               channels: int,
               times: list,
               max_project: bool = True,
               ) -> dict:
    """
    Return: dict of delayed dask arrays
    """

    # load views
    view_ims = {ch: {t: {vdv['view']: 
                            da.from_delayed(
                                delayed(io_utils.read_tile_from_multitile_czi)
                                   (vdv['filename'],
                                    vdv['view'],
                                    ch,
                                    time_index=t,
                                    max_project=max_project,
                                    origin=vdv['origin'],
                                    spacing=vdv['spacing'],
                                    ),
                                shape=tuple(vdv['shape']),
                                dtype=np.uint16,
                            )
                        for vdv in view_dict.values()}
                    for t in times}
                for ch in channels}

    return view_ims


class TemporarilyDisabledWidgets(object):
    """
    Conext manager to temporarily disable widgets during long computation
    """
    def __init__(self, widgets):
        self.widgets = widgets
        self.enabled_states = {w: True if w.enabled else False for w in widgets}
    def __enter__(self):
        for w in self.widgets:
            w.enabled = False
    def __exit__(self, type, value, traceback):
        for w in self.widgets:
            w.enabled = self.enabled_states[w]


class VisibleActivityDock(object):
    """
    Conext manager to temporarily disable widgets during long computation
    """
    def __init__(self, viewer):
        self.viewer = viewer
    def __enter__(self):
        self.viewer.window._status_bar._toggle_activity_dock(True)
    def __exit__(self, type, value, traceback):
        self.viewer.window._status_bar._toggle_activity_dock(False)


def compute_dask_object(dask_object,
                        viewer,
                        widgets_to_disable=None,
                        message="Registering tiles",
                        scheduler='threading',
                        ):
    """
    Compute dask object. While doing so:
     - show progress bar
     - disable widgets temporarily
    """
    with TemporarilyDisabledWidgets(widgets_to_disable),\
         VisibleActivityDock(viewer),\
         TqdmCallback(tqdm_class=progress, desc=message, bar_format=" "):
        result = compute(dask_object, scheduler=scheduler)[0]

    return result


def add_metadata_to_tiles(viewims, view_dict):

    channels = list(viewims.keys())
    times = list(viewims[channels[0]].keys())

    viewims =   {ch:
                    {t: {vdv['view']:
                            delayed(lambda x, origin, spacing:
                                    ImageArray(x, origin=origin, spacing=spacing))(
                        
                                        viewims[ch][t][vdv['view']],
                                        vdv['origin'],
                                        vdv['spacing'])
                        for vdv in view_dict.values()}
                    for t in times}
                for ch in channels}

    return viewims


# get source file path from open layers
def get_source_path_from_viewer(viewer):
    for l in viewer.layers:
        if l.source.path is not None and l.source.path.endswith('.czi'):
            return l.source.path
    return None


def layer_was_loaded_by_own_reader(layer):
    if 'napari_stitcher_reader_function' in layer.metadata and\
        layer.metadata['napari_stitcher_reader_function'] == 'read_mosaic_czi':
        return True
    else:
        False


def get_str_unique_to_view_from_layer_name(layer_name):
    return layer_name.split(' :: ')[0]


def get_str_unique_to_ch_from_layer_name(layer_name):
    return layer_name.split(' :: ')[1]


def get_view_from_layer(layer):
    return layer.metadata['view']


def filter_layers(layers, view=None, ch=None):
    for l in layers:
        if view is not None and get_str_unique_to_view_from_layer_name(l.name) != view: continue
        if ch is not None and get_str_unique_to_ch_from_layer_name(l.name) != ch: continue
        yield l


def duplicate_channel_xims(xims):

    xims_ch_duplicated = [
        xr.concat([xim] * 2, dim='C')\
        .assign_coords(C=[
            xim.coords['C'].data[0],
            xim.coords['C'].data[0] + '_2']
        ) for xim in xims]
    
    return xims_ch_duplicated


def shift_to_matrix(shift):
    ndim = len(shift)
    M = np.concatenate([shift, [1]], axis=0)
    M = np.concatenate([np.eye(ndim + 1)[:,:ndim], M[:,None]], axis=1)
    return M
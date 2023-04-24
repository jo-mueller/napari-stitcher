import numpy as np
import xarray as xr
from scipy import ndimage
import skimage
from tqdm import tqdm
import networkx as nx

from dask import delayed
import dask.array as da

from mvregfus import mv_utils, multiview
from mvregfus.image_array import ImageArray

from napari_stitcher import _spatial_image_utils, _mv_graph


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def bin_views(view_ims, registration_binning):
        return apply_recursive_dict(
            lambda x: delayed(mv_utils.bin_stack)(x, registration_binning),
            view_ims)


def bin_xim(xim, registration_binning):
    return xr.apply_ufunc(mv_utils.bin_stack, xim, registration_binning,
                   input_core_dims=_spatial_image_utils.get_spatial_dims_from_xim(xim),
                   output_core_dims=_spatial_image_utils.get_spatial_dims_from_xim(xim),
                   )


def register_pair_of_spatial_images(
                   xims: list,
                   registration_binning = None,
                   ) -> dict:
    """
    Register input spatial images. Assume there's no C and T.

    Return: Transform in homogeneous coordinates
    """

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    if registration_binning is not None:
        rxims = [xim.coarsen({dim: registration_binning[dim] for dim in spatial_dims},
                             boundary="trim").mean().astype(xim.dtype) for xim in xims]
    else:
        rxims = xims
        
    ims = []
    for rxim in rxims:
        origin = _spatial_image_utils.get_origin_from_xim(rxim)
        spacing = _spatial_image_utils.get_spacing_from_xim(rxim)
        ims.append(
            # delayed(ImageArray)(rxim.data,
            (ImageArray)(rxim.squeeze().data,

                       origin=[origin[dim] for dim in spatial_dims],
                       spacing=[spacing[dim] for dim in spatial_dims])
        )

    p = (multiview.register_linear_elastix)(
            ims[0], ims[1],
            None,
            '',
            f'',
            None,
            )
    
    p = (mv_utils.params_to_matrix)(p)

    return p


def get_registration_pair_graph(
        g,
        # method=,
        registration_binning = None,
):
    """
        - get shortest paths between all views and the reference view (deambiguate with largest overlap)
        - return directed graph of xims and registration edges with (delayed) transforms
    """
    g_reg = g.to_directed() # returns a deep copy

    pairs = _mv_graph.get_registration_pairs_from_overlap_graph(g_reg,
                    method='shortest_paths_considering_overlap')
    
    for pair in pairs:
        # g_pairs.edges[(pair[0], pair[1])]['marked_for_registration'] = True
        # g_pairs.add_edge(pair[0], pair[1])
        g_reg.edges[(pair[0], pair[1])]['transform'] = \
            delayed(register_pair_of_spatial_images)(
                [g.nodes[pair[0]]['xim'], g.nodes[pair[1]]['xim']],
                registration_binning=registration_binning,
                    )

    return g_reg


# def register_graph(
#         g,
#         registration_binning = None,
#         ) -> dict:
    
#     """
#     Attach (delayed) transforms to all edges with 'to_register: True'
#     attribute in the directed input graph
#     """

#     # gd = g.to_directed() # create deep copy
    
#     # register all pairs of views along the shortest paths
#     for e in g.edges:
#         if not 'marked_for_registration' in g.edges[e].keys()\
#             or not g.edges[e]['marked_for_registration']: continue
        
#         g.edges[e]['transform'] = \
#             delayed(register_pair_of_spatial_images)(
#                 [g.nodes[e[0]]['xim'], g.nodes[e[1]]['xim']],
#                 registration_binning=registration_binning,
#                     )
    
#     return g
#     # return _mv_graph.compute_graph(gd)

# def register_

#     ds_xims = xr.merge([xim.rename({dim: "%s_%s" %(dim, ixim)
#                                     for dim in spatial_dims}).to_dataset(name='im%s' %ixim)
#                         for ixim, xim in enumerate(xims[:2])])
    
    # xr.apply_ufunc(lambda x: x,
    #                ds_xims.sel(C=ds_xims.coords['C'][0]),
    #                dask='allowed',
    #             #    input_core_dims=[['%s_%s' %(dim, iim)  for iim in range(2) for dim in spatial_dims]],
    #                input_core_dims=[['T']],
    #                output_core_dims=('M1', 'M2'),
    #                )

    # delayed(multiview.register_linear_elastix)(
    #     ImageArray()
    # )

    # # perform pairwise registrations
    # pair_ps = {t: {(view1, view2):
    #                     delayed(multiview.register_linear_elastix)
    #                              (view_reg_ims[reg_channel][t][view1],
    #                               view_reg_ims[reg_channel][t][view2],
    #                             #  (
    #                             # delayed(ImageArray)(view_reg_ims[reg_channel][t][view1],
    #                             #         spacing=view_dict[view1]['spacing'],
    #                             #         origin=view_dict[view1]['origin']),
    #                             #   delayed(ImageArray)(view_reg_ims[reg_channel][t][view2],
    #                             #              spacing=view_dict[view2]['spacing'],
    #                             #              origin=view_dict[view2]['origin']),
    #                               -1, #degree
    #                               None,
    #                               '',
    #                               f'{view1}_{view2}',
    #                               None
    #                              )
    #             for view1, view2 in pairs}
    #         for t in times}

    # # get final transform parameters
    # ps = {t: delayed(multiview.get_params_from_pairs)(
    #                             views[ref_view_index],
    #                             pairs,
    #                             [pair_ps[t][(v1,v2)] for v1, v2 in pairs],
    #                             None, # time_alignment_params
    #                             True, # consider_reg_quality
    #                             [view_reg_ims[reg_channel][t][view] for view in views],
    #                             {view: iview for iview, view in enumerate(views)}
    #                             )
    #         for t in times}

    # ps = {t: delayed(lambda x: {view: x[iview] for iview, view in enumerate(views)})
    #             (ps[t])
    #         for t in times}

    # return ps


def correct_random_drift(ims, reg_ch=0, zoom_factor=10, particle_reinstantiation_stepsize=30, sigma_t=3):
    """
    ## Stage shift correction (currently 2d, but easily extendable)

    Goal: Correct for random stage shifts in timelapse movies in the absence of reference points that are fixed relative to the stage.

    Method: Assume that in the absence of random shifts particle trajectories are smooth in time. Find raw (uncorrected) trajectories, smooth them over time and determine the random stage shifts as the mean deviation of the actual trajectories from the smoothed ones.

    Steps:
    1) Calculate PIV fields (or optical flow) for the timelapse movie
    2) Track the initial coordinates as virtual particles throughout the timelapse
    3) Smooth trajectories and determine the mean deviations (N_t, d_x, d_y) from the actual trajectories
    4) Use the obtained deviations to correct the timelapse and verify result quality visually

    Comments:
    - assumes input data to be in the format (t, c, y, x)
    """

    import dask.delayed as d
    import dask
    from dask.diagnostics import ProgressBar

    # for some reason float images give a different output
    ims = ims.astype(np.uint16)

    of_channel = reg_ch
    regims = np.array([ndimage.zoom(ims[it, of_channel], 1./zoom_factor, order=1) for it in range(len(ims[:]))])

    fs = [d(skimage.registration.optical_flow_tvl1)(regims[t],
                                                    regims[t+1],
                                                attachment=10000,
                                                )
                for t in tqdm(range(len(ims) - 1)[:])]

    print('Computing optical flow...')
    with ProgressBar():
        fs=np.array(dask.compute(fs)[0])#*zoom_factor


    print('Tracking virtual particles...')

    # coordinates to be tracked
    x,y = np.mgrid[0:regims[0].shape[0], 0:regims[0].shape[1]]

    # define starting point(s) for coordinates to be tracked
    # for long movies, it's good to have several starting tps
    # as the coordinates can move out of the FOV
    
    starting_tps = range(0, len(ims), particle_reinstantiation_stepsize)

    posss = []
    for starting_tp in starting_tps:
        poss = [np.array([x, y])]
        for t in tqdm(range(starting_tp, len(ims) - 1)):
            displacement = np.array([ndimage.map_coordinates(
                                fs[t][dim],
                                np.array([poss[-1][0].flatten(), poss[-1][1].flatten()]),
                                order=1, mode='constant', cval=np.nan)
                            .reshape(x.shape) for dim in range(2)])
            poss.append(displacement + poss[-1])
        posss.append(np.array(poss))    


    print('Smoothing trajectories...')
    posss_smooth = [ndimage.gaussian_filter(posss[istp], [sigma_t, 0, 0, 0], mode='nearest')
                        for istp, starting_tp in enumerate(starting_tps)]

    devs = np.array([np.nanmean([posss_smooth[istp][t-starting_tp] - posss[istp][t-starting_tp]
                        for istp, starting_tp in enumerate(starting_tps) if starting_tp <= t], axis=(0, -1, -2))
                    for t in range(len(ims))])

    devs = devs * zoom_factor

    print('Correct drifts...')
    imst = np.array([[ndimage.affine_transform(ims[t,ch],
                                            matrix=[[1,0],[0,1]],
                                            offset=-devs[t], order=1)
                    for ch in range(ims.shape[1])]
                    for t in tqdm(range(len(ims)))])

    return imst, devs


import dask_image
def get_stabilization_parameters(tl, sigma=2):
    """
    Assume first dimension is time
    """

    ndim = tl[0].ndim

    ps = da.stack([da.from_delayed(delayed(skimage.registration.phase_cross_correlation)(
            tl[t-1],
            tl[t],
            upsample_factor=3,
            normalization=None)[0], shape=(ndim, ), dtype=float)
            for t in range(1, tl.shape[0])])
    
    ps = da.concatenate([da.zeros((1, ndim)), ps], axis=0)

    ps_cum = da.cumsum(ps, axis=0)
    ps_cum_filtered = dask_image.ndfilters.gaussian_filter(ps_cum, [sigma, 0], mode='nearest')

    # deltas = ps_cum - ps_cum_filtered
    deltas = ps_cum_filtered - ps_cum

    # imst = da.stack([da.from_delayed(delayed(ndimage.affine_transform)(tl[t],
    #                                         matrix=np.eye(2),
    #                                         offset=-ps[t-1] if t else np.zeros(2), order=1), shape=tl[0].shape, dtype=tl[0].dtype)

    return deltas



if __name__ == "__main__":

    viewer = napari.Viewer()
    viewer.open(filename)
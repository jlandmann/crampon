
import numpy as np
import logging
import pandas as pd
import xarray as xr
from crampon.cfg import SEC_IN_DAY, SEC_IN_HOUR
from oggm.core.flowline import *
import itertools
from scipy import optimize

# Module logger
log = logging.getLogger(__name__)


def merge_to_one_glacier_crampon(main, tribs, filename='climate_daily',
                         input_filesuffix=''):
    """
    Merge multiple tributary glacier flowlines to a main glacier
    This function will merge multiple tributary glaciers to a main glacier
    and write modified `model_flowlines` to the main GlacierDirectory.
    The provided tributaries must have an intersecting downstream line.
    To be sure about this, use `intersect_downstream_lines` first.
    This function is mainly responsible to reporject the flowlines, set
    flowline attributes and to copy additional files, like the necessary
    climate files.

    Parameters
    ----------
    main : `py:class:crampon.GlacierDirectory`
        The new GDir of the glacier of interest
    tribs : list or dictionary containing crampon.GlacierDirectories
        True tributary glaciers to the main glacier
    filename: str
        Baseline climate file
    input_filesuffix: str
        Filesuffix to the climate file
    """

    # read flowlines of the Main glacier
    fls = main.read_pickle('model_flowlines')
    mfl = fls.pop(-1)  # remove main line from list and treat seperately

    for trib in tribs:

        # read tributary flowlines and append to list
        tfls = trib.read_pickle('model_flowlines')

        # copy climate file and local_mustar to new gdir
        # if we have a merge-merge situation we need to copy multiple files
        rgiids = set([fl.rgi_id for fl in tfls])

        for uid in rgiids:
            if len(rgiids) == 1:
                # we do not have a merge-merge situation
                in_id = ''
                out_id = trib.rgi_id
            else:
                in_id = '_' + uid
                out_id = uid

            climfile_in = filename + in_id + input_filesuffix + '.nc'
            climfile_out = filename + '_' + out_id + input_filesuffix + '.nc'
            shutil.copyfile(os.path.join(trib.dir, climfile_in),
                            os.path.join(main.dir, climfile_out))

            # todo: remove when every gl. in CRAMPON has an OGGM cali as well
            try:
                _m = os.path.basename(trib.get_filepath('local_mustar')).split(
                    '.')
                muin = _m[0] + in_id + '.' + _m[1]
                muout = _m[0] + '_' + out_id + '.' + _m[1]

                shutil.copyfile(os.path.join(trib.dir, muin),
                                os.path.join(main.dir, muout))
            except:
                pass

            _ipot = os.path.basename(trib.get_filepath('ipot')).split('.')
            ipotin = _ipot[0] + in_id + '.' + _ipot[1]
            ipotout = _ipot[0] + '_' + out_id + '.' + _ipot[1]

            shutil.copyfile(os.path.join(trib.dir, ipotin),
                            os.path.join(main.dir, ipotout))

            _ipotpf = os.path.basename(
                trib.get_filepath('ipot_per_flowline')).split('.')
            ipotpfin = _ipotpf[0] + in_id + '.' + _ipotpf[1]
            ipotpfout = _ipotpf[0] + '_' + out_id + '.' + _ipotpf[1]

            shutil.copyfile(os.path.join(trib.dir, ipotpfin),
                            os.path.join(main.dir, ipotpfout))

        # sort flowlines descending
        tfls.sort(key=lambda x: x.order, reverse=True)

        # loop over tributaries and reproject to main glacier
        for nr, tfl in enumerate(tfls):

            # 1. Step: Change projection to the main glaciers grid
            _line = salem.transform_geometry(tfl.line,
                                             crs=trib.grid, to_crs=main.grid)

            # 2. set new line
            tfl.set_line(_line)

            # 3. set map attributes
            dx = [shpg.Point(tfl.line.coords[i]).distance(
                shpg.Point(tfl.line.coords[i+1]))
                for i, pt in enumerate(tfl.line.coords[:-1])]  # get distance
            # and check if equally spaced
            if not np.allclose(dx, np.mean(dx), atol=1e-2):
                raise RuntimeError('Flowline is not evenly spaced.')

            tfl.dx = np.mean(dx).round(2)
            tfl.map_dx = mfl.map_dx
            tfl.dx_meter = tfl.map_dx * tfl.dx

            # 3. remove attributes, they will be set again later
            tfl.inflow_points = []
            tfl.inflows = []

            # 4. set flows to, mainly to update flows_to_point coordinates
            if tfl.flows_to is not None:
                tfl.set_flows_to(tfl.flows_to)

        # append tributary flowlines to list
        fls += tfls

    # add main flowline to the end
    fls = fls + [mfl]

    # Finally write the flowlines
    main.write_pickle(fls, 'model_flowlines')
    # todo: stop to fake inversion flowlines!?
    main.write_pickle(fls, 'inversion_flowlines')  # OGGM doesn't seem to do it
    return fls

# overwrite OGGM function
merge_to_one_glacier = merge_to_one_glacier_crampon


def clean_merged_flowlines_crampon(gdir, buffer=None):
    """Order and cut merged flowlines to size.
    After matching flowlines were found and merged to one glacier directory
    this function makes them nice:
    There should only be one flowline per bed, so overlapping lines have to be
    cut, attributed to a another flowline and orderd.
    Parameters
    ----------
    gdir : oggm.GlacierDirectory
        The GDir of the glacier of interest
    buffer: float
        Buffer around the flowlines to find overlaps
    """

    # No buffer does not work
    if buffer is None:
        buffer = cfg.PARAMS['kbuffer']

    # Number of pixels to arbitrarily remove at junctions
    lid = int(cfg.PARAMS['flowline_junction_pix'])

    fls = gdir.read_pickle('model_flowlines')

    # seperate the main main flowline
    mainfl = fls.pop(-1)

    # split fls in main and tribs
    mfls = [fl for fl in fls if fl.flows_to is None]
    tfls = [fl for fl in fls if fl not in mfls]

    # --- first treat the main flowlines ---
    # sort by order and length as a second choice
    mfls.sort(key=lambda x: (x.order, len(x.inflows), x.length_m),
              reverse=False)

    merged = []

    # for fl1 in mfls:
    while len(mfls) > 0:
        fl1 = mfls.pop(0)

        ol_index = []  # list of index from first overlap

        # loop over other main lines and main main line
        for fl2 in mfls + [mainfl]:

            # calculate overlap, maybe use larger buffer here only to find it
            _overlap = fl1.line.intersection(fl2.line.buffer(buffer*2))

            # calculate indice of first overlap if overlap length > 0
            oix = 9999
            if _overlap.length > 0 and fl1 != fl2 and fl2.flows_to != fl1:
                if isinstance(_overlap, shpg.MultiLineString):
                    if _overlap[0].coords[0] == fl1.line.coords[0]:
                        # if the head of overlap is same as the first line,
                        # best guess is, that the heads are close topgether!
                        _ov1 = _overlap[1].coords[1]
                    else:
                        _ov1 = _overlap[0].coords[1]
                else:
                    _ov1 = _overlap.coords[1]
                for _i, _p in enumerate(fl1.line.coords):
                    if _p == _ov1:
                        oix = _i
                # low indices are more likely due to an wrong overlap
                if oix < 10:
                    oix = 9999
            ol_index.append(oix)

        ol_index = np.array(ol_index)
        if np.all(ol_index == 9999):
            log.warning('Glacier %s could not be merged, removed!' %
                        fl1.rgi_id)
            # remove possible tributary flowlines
            tfls = [fl for fl in tfls if fl.rgi_id != fl1.rgi_id]
            # skip rest of this while loop
            continue

        # make this based on first overlap, but consider order and or length
        minx = ol_index[ol_index <= ol_index.min()+10][-1]
        i = np.where(ol_index == minx)[0][-1]
        _olline = (mfls + [mainfl])[i]

        # 1. cut line to size
        _line = fl1.line

        bufferuse = buffer
        while bufferuse > 0:
            _overlap = _line.intersection(_olline.line.buffer(bufferuse))
            _linediff = _line.difference(_overlap)  # cut to new line

            # if the tributary flowline is longer than the main line,
            # _line will contain multiple LineStrings: only keep the first
            if isinstance(_linediff, shpg.MultiLineString):
                _linediff = _linediff[0]

            if len(_linediff.coords) < 10:
                bufferuse -= 1
            else:
                break

        if bufferuse <= 0:
            log.warning('Glacier %s would be to short after merge, removed!' %
                        fl1.rgi_id)
            # remove possible tributary flowlines
            tfls = [fl for fl in tfls if fl.rgi_id != fl1.rgi_id]
            # skip rest of this while loop
            continue

        # remove cfg.PARAMS['flowline_junction_pix'] from the _line
        # gives a bigger gap at the junction and makes sure the last
        # point is not corrupted in terms of spacing
        _line = shpg.LineString(_linediff.coords[:-lid])

        # 2. set new line
        fl1.set_line(_line)

        # 3. set flow to attributes. This also adds inflow values to other
        fl1.set_flows_to(_olline)

        # change the array size of tributary flowline attributs
        for atr, value in fl1.__dict__.items():
            if atr in ['_ptrap', '_prec']:
                # those are indices, remove those above nx
                fl1.__setattr__(atr, value[value < fl1.nx])
            elif isinstance(value, np.ndarray) and (len(value) > fl1.nx):
                # those are actual parameters on the grid
                fl1.__setattr__(atr, value[:fl1.nx])

        merged.append(fl1)
    allfls = merged + tfls

    # now check all lines for possible cut offs
    for fl in allfls:
        try:
            fl.flows_to_indice
        except AssertionError:
            mfl = fl.flows_to
            # remove it from original
            mfl.inflow_points.remove(fl.flows_to_point)
            mfl.inflows.remove(fl)

            prdis = mfl.line.project(fl.tail)
            mfl_keep = mfl

            while mfl.flows_to is not None:
                prdis2 = mfl.flows_to.line.project(fl.tail)
                if prdis2 < prdis:
                    mfl_keep = mfl
                    prdis = prdis2
                mfl = mfl.flows_to

            # we should be good to add this line here
            fl.set_flows_to(mfl_keep.flows_to)

    allfls = allfls + [mainfl]

    for fl in allfls:
        fl.inflows = []
        fl.inflow_points = []
        if hasattr(fl, '_lazy_flows_to_indice'):
            delattr(fl, '_lazy_flows_to_indice')
        if hasattr(fl, '_lazy_inflow_indices'):
            delattr(fl, '_lazy_inflow_indices')

    for fl in allfls:
        if fl.flows_to is not None:
            fl.set_flows_to(fl.flows_to)

    for fl in allfls:
        fl.order = line_order(fl)

    # order flowlines in descending way
    allfls.sort(key=lambda x: x.order, reverse=False)

    # assert last flowline is main flowline
    assert allfls[-1] == mainfl

    # Finally write the flowlines
    gdir.write_pickle(allfls, 'model_flowlines')

# overwrite OGGM function
clean_merged_flowlines = clean_merged_flowlines_crampon
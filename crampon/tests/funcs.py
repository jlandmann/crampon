""" Some generic functions used for testing, some relying on OGGM."""

import os
import crampon
from crampon import cfg
from crampon.workflow import execute_entity_task
from oggm.utils import get_demo_file, mkdir
from oggm.tests.funcs import get_test_dir


def init_ogg(reset=False, border=40, logging_level='INFO'):
    """
    OGGM initializes HEF, we take OGG (Oberer Grindelwaldgletscher).

    todo: soon switch to some other glacier which has more data.

    Parameters
    ----------
    reset : bool
        Whether to reset the test GlacierDirectory.
    border : int
        Domain border in pixels.
    logging_level : str
        Logging level accepted by cfg.initialize(). Default: 'INFO' (verbose).

    Returns
    -------
    None.
    """

    from crampon.core.preprocessing import gis, climate, centerlines
    import geopandas as gpd

    # test directory
    testdir = os.path.join(get_test_dir(), 'tmp_border{}'.format(border))
    if not os.path.exists(testdir):
        os.makedirs(testdir)
        reset = True

    # Init
    # funnily, with generic names like "dem" we get correct data (dangerous!)
    cfg.initialize(os.path.join('..', '..', 'sandbox', 'CH_params.cfg'),
                   logging_level=logging_level)
    cfg.PATHS['dem_file'] = get_demo_file('dem.tif')
    cfg.PARAMS['baseline_climate'] = ''
    cfg.PATHS['working_dir'] = testdir
    cfg.PARAMS['border'] = border

    ogg_file = get_demo_file('outlines.shp')
    ol = gpd.read_file(ogg_file)
    ol = ol.to_crs(epsg=4326)
    entity = ol.iloc[0]

    gdir = crampon.GlacierDirectory(entity, reset=reset)
    if not gdir.has_file('inversion_params'):
        reset = True
        gdir = crampon.GlacierDirectory(entity, reset=reset)

    if not reset:
        return gdir

    gis.define_glacier_region(gdir, entity=entity)
    execute_entity_task(gis.glacier_masks, [gdir])
    execute_entity_task(centerlines.compute_centerlines, [gdir])
    centerlines.initialize_flowlines(gdir)
    centerlines.compute_downstream_line(gdir)
    centerlines.compute_downstream_bedshape(gdir)
    centerlines.catchment_area(gdir)
    centerlines.catchment_intersections(gdir)
    centerlines.catchment_width_geom(gdir)
    centerlines.catchment_width_correction(gdir)
    climate.process_custom_climate_data(gdir)

    return gdir

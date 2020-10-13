"""
pymmaster.vector_tools provides tools to manipulate tilings (naming, extents) and vectors (rasterize, buffer, transform, ...)
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import gdal, ogr, osr, gdalconst

def SRTMGL1_naming_to_latlon(tile_name):
    if tile_name[0] == 'S' or tile_name[0] == 's':
        lat = -int(tile_name[1:3])
    elif tile_name[0] == 'N' or tile_name[0] == 'n':
        lat = int(tile_name[1:3])
    else:
        sys.exit('Could not read latitude according to SRTMGL1 naming convention.')

    if tile_name[3] == 'W' or tile_name[3] == 'w':
        lon = -int(tile_name[4:7])
    elif tile_name[3] == 'E' or tile_name[3] == 'e':
        lon = int(tile_name[4:7])
    else:
        sys.exit('Could not read longitude according to SRTMGL1 naming convention.')

    return lat, lon

def latlon_to_SRTMGL1_naming(lat,lon):

    if lat<0:
        str_lat = 'S'
    else:
        str_lat = 'N'

    if lon<0:
        str_lon = 'W'
    else:
        str_lon = 'E'

    tile_name = str_lat+str(int(abs(np.floor(lat)))).zfill(2)+str_lon+str(int(abs(np.floor(lon)))).zfill(3)

    return tile_name


def latlon_to_UTM(lat,lon):
    # utm module excludes regions south of 80°S and north of 84°N, unpractical for global vector manipulation
    # utm_all = utm.from_latlon(lat,lon)
    # utm_nb=utm_all[2]

    # utm zone from longitude without exclusions
    if -180 <= lon < 180:
        utm_nb = int(
            np.floor((lon + 180) / 6)) + 1  # lon=-180 refers to UTM zone 1 towards East (West corner convention)
    else:
        sys.exit('Longitude value is out of range.')

    if 0 <= lat < 90:  # lat=0 refers to North (South corner convention)
        epsg = '326' + str(utm_nb).zfill(2)
        utm_zone = str(utm_nb).zfill(2) + 'N'
    elif -90 <= lat < 0:
        epsg = '327' + str(utm_nb).zfill(2)
        utm_zone = str(utm_nb).zfill(2) + 'S'
    else:
        sys.exit('Latitude value is out of range.')

    return epsg, utm_zone


def epsg_from_utm(utm_zone):
    str_utm_nb = utm_zone[0:2]
    str_utm_ns = utm_zone[2]

    if str_utm_ns == 'N':
        epsg = '326' + str_utm_nb
    elif str_utm_ns == 'S':
        epsg = '327' + str_utm_nb
    else:
        sys.exit('UTM format not recognized.')

    return int(epsg)


def utm_from_epsg(epsg):
    str_epsg = str(epsg)
    str_epsg_ns = str_epsg[0:3]
    str_epsg_nb = str_epsg[3:5]

    if str_epsg_ns == '326':
        utm = str_epsg_nb + 'N'
    elif str_epsg_ns == '327':
        utm = str_epsg_nb + 'S'
    else:
        sys.exit('EPSG UTM format not recognized.')

    return utm


def poly_utm_latlontile(tile_name, utm_zone):
    lat, lon = SRTMGL1_naming_to_latlon(tile_name)
    extent = lon, lat, lon + 1, lat + 1
    poly = poly_from_extent(extent)

    epsg_out = epsg_from_utm(utm_zone)  # tile can be projected in whatever utm zone
    trans = coord_trans(False, 4326, False, epsg_out)

    poly.Transform(trans)

    return poly


#create a regularly spaced extent in a utm zone, with 3 overlapping pixel on the edges for reprojections
def niceextent_utm_latlontile(tile_name, utm_zone, gsd):
    poly = poly_utm_latlontile(tile_name,utm_zone)
    xmin, ymin, xmax, ymax = extent_from_poly(poly)

    xmin = xmin - xmin % gsd - 3*gsd
    ymin = ymin - ymin % gsd - 3*gsd
    xmax = xmax - xmax % gsd + 3*gsd
    ymax = ymax - ymax % gsd + 3*gsd

    return xmin, ymin, xmax, ymax


def create_mem_shp(geom, srs, layer_name='NA', layer_type=ogr.wkbPolygon, field_id='ID', field_val='1',field_type=ogr.OFTInteger):
    ds = gdal.GetDriverByName('MEMORY').Create('test.shp', 0, 0, 0, gdal.OF_VECTOR)
    layer = ds.CreateLayer(layer_name, srs, layer_type)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn(field_id, field_type))
    defn = layer.GetLayerDefn()

    # if there are multiple geometries, put the "for" loop here
    # create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField(field_id, field_val)

    # make geometry
    geom = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)
    layer = feat = geom = None

    return ds

def latlontile_nodatamask(geoimg,tile_name):

    #create latlon tile polygon in utm projection
    lat, lon = SRTMGL1_naming_to_latlon(tile_name)
    extent = lon, lat, lon + 1, lat + 1
    poly = poly_from_extent(extent)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    #put in a memory vector
    ds_shp = create_mem_shp(poly,srs)

    return geoimg_mask_on_feat_shp_ds(ds_shp, geoimg)

def poly_from_coords(list_coord):
    # creating granule polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in list_coord:
        ring.AddPoint(float(coord[0]), float(coord[1]))

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly

def get_poly_centroid(poly):
    centroid = poly.Centroid()

    center_lon, center_lat, _ = centroid.GetPoint()

    return center_lon, center_lat

def extent_rast(raster_in):
    ds = gdal.Open(raster_in, gdalconst.GA_ReadOnly)
    x0_ref, dx_ref, dxdy_ref, y0_ref, dydx_ref, dy_ref = ds.GetGeoTransform()
    proj_wkt = ds.GetProjection()
    col_tot = ds.RasterXSize
    lin_tot = ds.RasterYSize
    x1_ref = x0_ref + col_tot * dx_ref
    y1_ref = y0_ref + lin_tot * dy_ref
    ds = None

    # extent format: Xmin, Ymin, Xmax, Ymax
    xmin = min(x0_ref, x1_ref)
    ymin = min(y0_ref, y1_ref)
    xmax = max(x0_ref, x1_ref)
    ymax = max(y0_ref, y1_ref)

    extent = [xmin, ymin, xmax, ymax]

    return extent, proj_wkt


def poly_from_extent(extent):
    # create a polygon from extent, coordinates order as in gdal
    xmin, ymin, xmax, ymax = extent

    ring = ogr.Geometry(ogr.wkbLinearRing)  # creating polygon ring
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmin, ymin)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)  # creating polygon

    return poly


def extent_from_poly(poly):
    env = poly.GetEnvelope()
    extent = env[0], env[2], env[1], env[3]

    return extent


def coord_trans(is_src_wkt, proj_src, is_tgt_wkt, proj_tgt):
    # choice between WKT or EPSG
    source_proj = osr.SpatialReference()
    if is_src_wkt:
        source_proj.ImportFromWkt(proj_src)
    else:
        source_proj.ImportFromEPSG(proj_src)

    target_proj = osr.SpatialReference()
    if is_tgt_wkt:
        target_proj.ImportFromWkt(proj_tgt)
    else:
        target_proj.ImportFromEPSG(proj_tgt)

    transform = osr.CoordinateTransformation(source_proj, target_proj)

    return transform


def list_shp_field_inters_extent(fn_shp, field_name, extent, proj_ext):
    poly = poly_from_extent(extent)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(fn_shp, 0)
    layer = ds.GetLayer()

    proj_shp = layer.GetSpatialRef().ExportToWkt()

    trans = coord_trans(True, proj_ext, True, proj_shp)

    poly.Transform(trans)

    list_field_inters = []
    for feat in layer:
        feat_geom = feat.GetGeometryRef()
        # inters = feat_geom.Intersection(poly)

        if feat_geom.Intersect(poly):
            list_field_inters.append(feat.GetField(field_name))

    return list_field_inters

def inters_list_poly_with_poly(list_poly,poly):

    list_inters=[]
    for poly_2 in list_poly:
        inters = poly_2.Intersection(poly)

        if not inters.IsEmpty():
            list_inters.append(poly_2)

    return list_inters

def get_buffered_area_ratio(geom,proj_in,buff):

    centroid_lon, centroid_lat, _ = geom.Centroid().GetPoint()
    epsg, utm = latlon_to_UTM(centroid_lat,centroid_lon)

    trans = coord_trans(True,proj_in,False,int(epsg))

    geom.Transform(trans)

    area_init = geom.GetArea()
    geom_buff = geom.Buffer(buff)
    area_buff = geom_buff.GetArea()

    return (area_buff-area_init)/area_init*100., area_init


def create_mem_raster_on_geoimg(geoimg):
    masktarget = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, gdal.GDT_Byte)
    masktarget.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    masktarget.SetProjection(geoimg.proj_wkt)
    masktarget.GetRasterBand(1).Fill(0)

    return masktarget


def geoimg_mask_on_feat_shp_ds(shp_ds, geoimg, layer_name='NA', feat_id='ID', feat_val='1', **kwargs):
    ds_out = create_mem_raster_on_geoimg(geoimg)
    rasterize_feat_shp_ds(shp_ds, ds_out, layer_name=layer_name, feat_id=feat_id, feat_val=feat_val, **kwargs)
    mask = ds_out.GetRasterBand(1).ReadAsArray()
    mask = mask.astype(float)

    return mask == 1


def rasterize_feat_shp_ds(shp_ds, raster_ds, layer_name='NA', feat_id='ID', feat_val='1', all_touched=False,
                          exclude=False):
    if not exclude:
        str_eq = "='"
    else:
        str_eq = "!='"

    sql_stat = 'SELECT * FROM ' + layer_name + ' WHERE ' + feat_id + str_eq + feat_val + "'"

    opts = gdal.RasterizeOptions(burnValues=[1], bands=[1], SQLStatement=sql_stat, allTouched=all_touched)
    gdal.Rasterize(raster_ds, shp_ds, options=opts)

#tools specific to ASTER L1A data


def extract_odl_astL1A(fn):
    f = open(fn, 'r')
    body = f.read()

    def get_odl_parenth_value(text_odl, obj_name):
        posobj = str.find(text_odl, obj_name)
        posval = str.find(text_odl[posobj + 1:len(text_odl)], 'VALUE')
        posparenthesis = str.find(text_odl[posobj + 1 + posval:len(text_odl)], '(')
        posendval = str.find(text_odl[posobj + 1 + posval + posparenthesis:len(text_odl)], ')')

        val = text_odl[posobj + posval + posparenthesis + 2:posobj + posval + posparenthesis + posendval + 1]

        return val

    def get_odl_quot_value(text_odl, obj_name):
        posobj = str.find(text_odl, obj_name)
        posval = str.find(text_odl[posobj + 1:len(text_odl)], 'VALUE')
        posquote = str.find(text_odl[posobj + 1 + posval:len(text_odl)], '"')
        posendval = str.find(text_odl[posobj + posval + posquote + 2:len(text_odl)], '"')

        val = text_odl[posobj + posval + posquote + 2:posobj + posval + +posquote + posendval + 2]

        return val

    # get latitude
    lat_val = get_odl_parenth_value(body, 'GRingPointLatitude')
    lat_tuple = [float(lat_val.split(',')[0]), float(lat_val.split(',')[1]), float(lat_val.split(',')[2]),
                 float(lat_val.split(',')[3])]

    # get longitude
    lon_val = get_odl_parenth_value(body, 'GRingPointLongitude')
    lon_tuple = [float(lon_val.split(',')[0]), float(lon_val.split(',')[1]), float(lon_val.split(',')[2]),
                 float(lon_val.split(',')[3])]

    # get calendar date + time of day
    caldat_val = get_odl_quot_value(body, 'CalendarDate')
    timeday_val = get_odl_quot_value(body, 'TimeofDay')
    caldat = datetime(year=int(caldat_val.split('-')[0]), month=int(caldat_val.split('-')[1]),
                      day=int(caldat_val.split('-')[2]),
                      hour=int(timeday_val.split(':')[0]), minute=int(timeday_val.split(':')[1]),
                      second=int(timeday_val.split(':')[2][0:2]),
                      microsecond=int(timeday_val.split(':')[2][3:6]) * 1000)

    # get cloud cover
    cloudcov_val = get_odl_quot_value(body, 'SceneCloudCoverage')
    cloudcov_perc = int(cloudcov_val)

    # get flag if bands acquired or not: band 1,2,3N,3B,4,5,6,7,8,9,10,11,12,13,14
    list_band = []
    band_attr = get_odl_quot_value(body, 'Band3N_Available')
    band_avail = band_attr[0:3] == 'Yes'
    list_band.append(band_avail)
    band_attr = get_odl_quot_value(body, 'Band3B_Available')
    band_avail = band_attr[0:3] == 'Yes'
    list_band.append(band_avail)

    range_band = list(range(1, 15))
    range_band.remove(3)
    for i in range_band:
        band_attr = get_odl_quot_value(body, 'Band' + str(i) + '_Available')
        band_avail = band_attr[0:3] == 'Yes'
        list_band.append(band_avail)

    band_tags = pd.DataFrame(data=list_band,
                             index=['band_3N', 'band_3B', 'band_1', 'band_2', 'band_4', 'band_5', 'band_6', 'band_7',
                                    'band_8', 'band_9', 'band_10', 'band_11', 'band_12', 'band_13', 'band_14'])

    # get scene orientation angle
    orient_attr = get_odl_quot_value(body, 'ASTERSceneOrientationAngle')
    # orient_angl = float(orient_attr)
    # some .met files are in fact somehow incomplete for angles... let's forget it!
    orient_angl = float(15.)

    return lat_tuple, lon_tuple, caldat, cloudcov_perc, band_tags, orient_angl


def l1astrip_polygon(l1a_subdir):
    # number of l1a granules
    strip_l1a = [os.path.join(l1a_subdir, l1a) for l1a in os.listdir(l1a_subdir) if l1a.endswith('.met')]

    list_poly = []
    for l1a in strip_l1a:
        lat_tup, lon_tup, _, _, _, _ = extract_odl_astL1A(l1a)

        max_lon = np.max(lon_tup)
        min_lon = np.min(lon_tup)

        if min_lon < -160 and max_lon > 160:
            # if this is happening, ladies and gentlemen, bad news, we definitely have an image on the dateline

            # let's do two full polygons from each side of the dateline...
            lon_rightside = np.array(lon_tup, dtype=float)
            lon_rightside[lon_rightside < -160] += 360

            lon_leftside = np.array(lon_tup, dtype=float)
            lon_leftside[lon_leftside > 160] -= 360

            rightside_coord = list(zip(list(lon_rightside) + [lon_rightside[0]], lat_tup + [lat_tup[0]]))
            rightside_poly = poly_from_coords(rightside_coord)

            leftside_coord = list(zip(list(lon_leftside) + [lon_leftside[0]], lat_tup + [lat_tup[0]]))
            leftside_poly = poly_from_coords(leftside_coord)

            # create a world polygon and get intersection
            world_coord = [(-180, -90), (-180, 90), (180, 90), (180, -90), (-180, -90)]
            world_poly = poly_from_coords(world_coord)

            leftside_inters = world_poly.Intersection(leftside_poly)
            rightside_inters = world_poly.Intersection(rightside_poly)

            # add both to list
            list_poly += [leftside_inters, rightside_inters]
        else:
            list_coord = list(zip(lon_tup + [lon_tup[0]], lat_tup + [lat_tup[0]]))
            poly = poly_from_coords(list_coord)
            list_poly.append(poly)

    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)

    for i in range(len(list_poly)):
        # stacking polygons in multipolygon
        multipoly.AddGeometry(list_poly[i])

    cascadedpoly = multipoly.UnionCascaded()

    return cascadedpoly

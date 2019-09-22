from __future__ import print_function
import gdal, osr, ogr
import xarray as xr
import os, sys
import numpy as np

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

def latlon_to_UTM(lat,lon):

    #utm module excludes regions south of 80°S and north of 84°N, unpractical for global vector manipulation
    # utm_all = utm.from_latlon(lat,lon)
    # utm_nb=utm_all[2]

    #utm zone from longitude without exclusions
    if -180<=lon<180:
        utm_nb=int(np.floor((lon+180)/6))+1 #lon=-180 refers to UTM zone 1 towards East (West corner convention)
    else:
        sys.exit('Longitude value is out of range.')

    if 0<=lat<90: #lat=0 refers to North (South corner convention)
        epsg='326'+str(utm_nb).zfill(2)
        utm_zone=str(utm_nb).zfill(2)+'N'
    elif -90<=lat<0:
        epsg='327'+str(utm_nb).zfill(2)
        utm_zone=str(utm_nb).zfill(2)+'S'
    else:
        sys.exit('Latitude value is out of range.')

    return epsg, utm_zone

def epsg_from_utm(utm_zone):

    str_utm_nb = utm_zone[0:2]
    str_utm_ns = utm_zone[2]

    if str_utm_ns == 'N':
        epsg = '326'+str_utm_nb
    elif str_utm_ns == 'S':
        epsg = '327'+str_utm_nb
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

def poly_utm_latlontile(tile_name,utm_zone):

    lat, lon = SRTMGL1_naming_to_latlon(tile_name)
    extent = lon, lat, lon + 1, lat + 1
    poly = poly_from_extent(extent)

    epsg_out = epsg_from_utm(utm_zone) # tile can be projected in whatever utm zone
    trans = coord_trans(False, 4326, False, epsg_out)

    poly.Transform(trans)

    return poly

def niceextent_utm_latlontile(tile_name,utm_zone,gsd):

    poly = poly_utm_latlontile(tile_name,utm_zone)
    xmin, ymin, xmax, ymax = extent_from_poly(poly)

    xmin = xmin - xmin % gsd
    ymin = ymin - ymin % gsd
    xmax = xmax - xmax % gsd
    ymax = ymax - ymax % gsd

    return xmin, ymin, xmax, ymax

def create_mem_shp(geom,srs,layer_name='NA',layer_type=ogr.wkbPolygon,field_id='ID',field_val='1'):

    ds = gdal.GetDriverByName('MEMORY').Create('test.shp',0,0,0,gdal.OF_VECTOR)
    layer = ds.CreateLayer(layer_name, srs, layer_type)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn(field_id, ogr.OFTInteger))
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

def latlontile_nodatamask(geoimg,tile_name,utm_zone):

    #create latlon tile polygon in utm projection
    poly = poly_utm_latlontile(tile_name,utm_zone)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_from_utm(utm_zone))
    #put in a memory vector
    ds_shp = create_mem_shp(poly,srs)

    return geoimg_mask_on_feat_shp_ds(ds_shp,geoimg)


def poly_from_extent(extent):

    #create a polygon from extent, coordinates order as in gdal
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

    linearring=poly.GetGeometryRef(0)

    x1, y1, _ = linearring.GetPoint(0)

    x2, y2, _ = linearring.GetPoint(2)

    xmin=min(x1,x2)
    ymin=min(y1,y2)
    xmax=max(x1,x2)
    ymax=max(y1,y2)

    extent = xmin, ymin, xmax, ymax

    return extent

def coord_trans(is_src_wkt,proj_src,is_tgt_wkt,proj_tgt):

    #choice between WKT or EPSG
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

def list_shp_field_inters_extent(fn_shp,field_name,extent,proj_ext):

    poly = poly_from_extent(extent)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(fn_shp, 0)
    layer = ds.GetLayer()

    proj_shp = layer.GetSpatialRef().ExportToWkt()

    trans = coord_trans(True,proj_ext,True,proj_shp)

    poly.Transform(trans)

    list_field_inters=[]
    for feat in layer:
        feat_geom = feat.GetGeometryRef()
        inters = feat_geom.Intersection(poly)

        if not inters.IsEmpty():
            list_field_inters.append(feat.GetField(field_name))

    return list_field_inters

def create_mem_raster_on_geoimg(geoimg):

    masktarget = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, gdal.GDT_Byte)
    masktarget.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    masktarget.SetProjection(geoimg.proj_wkt)
    masktarget.GetRasterBand(1).Fill(0)

    return masktarget

def geoimg_mask_on_feat_shp_ds(shp_ds,geoimg,layer_name='NA',feat_id='ID',feat_val='1',**kwargs):

    ds_out = create_mem_raster_on_geoimg(geoimg)
    rasterize_feat_shp_ds(shp_ds,ds_out,layer_name=layer_name,feat_id=feat_id,feat_val=feat_val,**kwargs)
    mask = ds_out.GetRasterBand(1).ReadAsArray()
    mask = mask.astype(float)

    return mask == 1

def rasterize_feat_shp_ds(shp_ds,raster_ds,layer_name='NA',feat_id='ID',feat_val='1',all_touched=False,exclude=False):

    if not exclude:
        str_eq="='"
    else:
        str_eq="!='"

    sql_stat='SELECT * FROM '+layer_name+' WHERE '+feat_id+str_eq+feat_val+"'"

    opts=gdal.RasterizeOptions(burnValues=[1],bands=[1],SQLStatement=sql_stat,allTouched=all_touched)
    gdal.Rasterize(raster_ds,shp_ds,options=opts)


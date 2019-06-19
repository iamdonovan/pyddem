#!/usr/bin/env python
import os
import argparse
import gdal
import lxml.etree as etree
import lxml.builder as builder


def _argparser():
    parser = argparse.ArgumentParser(description="Given a GDAL dataset, create a MicMac xml worldfile.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename', action='store', type=str,
                        help='List of DEM files to read and stack.')
    parser.add_argument('-m', '--mask', action='store', type=str, default='./MEC-Malt/Masq_STD-MALT_DeZoom1.tif',
                        help='Path to mask file [./MEC-Malt/Masq_STD-MALT_DeZoom1.tif]')
    parser.add_argument('-g', '--geom', action='store', type=str, default='eGeomMNTEuclid',
                        help='MicMac Geometry name [eGeomMNTEuclid]')
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()
    
    ds = gdal.Open(args.filename)
    ext = os.path.splitext(args.filename)[-1]
    ulx, dx, _, uly, _, dy = ds.GetGeoTransform()
    
    E = builder.ElementMaker()    
    FileOriMnt = E.FileOriMnt
    NameFileMnt = E.NameFileMnt
    NameFileMasque = E.NameFileMasque
    NombrePixels = E.NombrePixels
    OriginePlani = E.OriginePlani
    ResolutionPlani = E.ResolutionPlani
    OrigineAlti = E.OrigineAlti
    ResolutionAlti = E.ResolutionAlti
    Geometrie = E.Geometrie

    outxml = FileOriMnt(
                NameFileMnt(args.filename),
                NameFileMasque(args.mask),
                NombrePixels(' '.join([str(ds.RasterXSize), str(ds.RasterYSize)])),
                OriginePlani(' '.join([str(ulx), str(uly)])),
                ResolutionPlani(' '.join([str(dx), str(dy)])),
                OrigineAlti('0'),
                ResolutionAlti('1'),
                Geometrie(args.geom)
             )

    tree = etree.ElementTree(outxml)
    tree.write(args.filename.replace(ext, '.xml'), pretty_print=True, 
               xml_declaration=False, encoding="utf-8")


if __name__ == "__main__":
    main()
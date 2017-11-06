#!/bin/bash
# post-process everything output from MicMac into nice files to use:
#	- masked DEM
#	- hillshade
#	- georeferenced orthophoto
#	- footprints of DEMs (can use to clip edges, for example)
# use: bash PostProcessMicMac.sh utm_zone,
#	where utm_zone has the form ""6 +north" for the projection used in processing.

resize_rasters () {
    image1=$1
    image2=$2
    # first, get the raster sizes and check that they're the same
    img1size=($(gdalinfo $image1 | grep 'Size is' | grep -o '[0-9]*'))
    img2size=($(gdalinfo $image2 | grep 'Size is' | grep -o '[0-9]*'))
    # if the rasters are the same size, we continue.
    if [[ "${img1size[0]}" -eq "${img2size[0]}" && "${img1size[1]}" -eq "${img2size[1]}" ]]; then
        echo "$image1 and $image2 are the same size. Exiting..."
        return 1
    fi
    # get the upper left and lower right corners of image1
    ul=$(gdalinfo $image1 | grep 'Upper Left' | grep -Eo '[+-]?[0-9]*\.[0-9]*\, [0-9]*\.[0-9]*' )
    lr=$(gdalinfo $image1 | grep 'Lower Right' | grep -Eo '[+-]?[0-9]*\.[0-9]*\, [0-9]*\.[0-9]*' )
    # split into two arrays    
    ul_arr=($(echo $ul | tr -d ,))
    lr_arr=($(echo $lr | tr -d ,))
    echo "gdalwarp -te ${ul_arr[0]} ${lr_arr[1]} ${lr_arr[0]} ${ul_arr[1]} -ts ${img1size[@]} $image2 ${image2%.tif}_resize.tif"
    echo "Re-sizing $image2 to agree with $image1 size."
    gdalwarp -te ${ul_arr[0]} ${lr_arr[1]} ${lr_arr[0]} ${ul_arr[1]} -ts ${img1size[@]} $image2 ${image2%.tif}_resize.tif
    mv -v ${image2%.tif}_resize.tif $image2
}

# first, get the masked dems and put them into a folder creatively called MaskedDEMs
# also get the orthophotos and put them into a folder creatively called OrthoImgs
basedir=$(pwd) #get the directory that we're starting in.
#mkdir -p MaskedDEMs OrthoImgs Footprints CorrelationImgs
mkdir -p PROCESSED_FINAL
outdir=$basedir/PROCESSED_FINAL

echo "using projection $1"
echo "getting masked DEMs and orthoimages."

for dir in $(ls -d AST_L1A*); do
	echo $dir

	#tmpstr=${dir:11:8}
	#datestr=${tmpstr:4:4}${tmpstr:0:4}
	datestr=$dir
	mkdir -p $outdir/$datestr
	
	cd $dir

	if [ -d "MEC-Malt" ]; then
		cd MEC-Malt
		# note: this could actually be hard-coded, since it's probably always 9.
        finalimgs=($(ls Z_Num*_DeZoom1_STD-MALT.tif))
        finalmsks=($(ls AutoMask_STD-MALT_Num*.tif))
        finalcors=($(ls Correl_STD-MALT_Num*.tif))
		# find the last image name. ancient systems like Moulin don't like the -1 index.
        lastimg=${finalimgs[-1]}
        lastmsk=${finalmsks[-1]}
        lastcor=${finalcors[-1]}
		# here's the kludge that should work on all bash systems, even
		# the ancient ones like RHEL6.
		#imgind=$((${#finalimgs[@]}-1))
		#mskind=$((${#finalmsks[@]}-1))

		#lastimg=${finalimgs[imgind]}
		#lastmsk=${finalmsks[mskind]}

		# strip the extension
        laststr="${lastimg%.*}"
        maskstr="${lastmsk%.*}"
        corrstr="${lastcor%.*}"

		# now, assign the CRS we got to the mask, dem, and apply.
		echo "Georeferencing correlation mask"
		gdal_translate -a_nodata 0 -a_srs "+proj=utm +zone=$1 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" $lastcor $dir\_CORR.tif
		echo "Creating temporary georeferenced DEM"
		gdal_translate -a_srs "+proj=utm +zone=$1 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" $lastimg tmp_geo.tif
		echo "Creating temporary georeferenced Mask"
		gdal_translate -a_srs "+proj=utm +zone=$1 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" -a_nodata 0 $lastmsk tmp_msk.tif
		cd ../
		if [ -d "Ortho-MEC-Malt" ]; then 
			cd Ortho-MEC-Malt
			echo "Creating double size correlation mask for ortho"
			gdal_translate -tr 15 15 -a_srs "+proj=utm +zone=$1 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" -a_nodata 0 ../MEC-Malt/$lastmsk tmp_mskDouble.tif
			echo "Creating temporary georeferenced ortho"
			gdal_translate -tr 15 15 -a_srs "+proj=utm +zone=$1 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" Orthophotomosaic.tif tmp_V123.tif
            resize_rasters tmp_mskDouble.tif tmp_V123.tif
			cd ../
		fi
		cd MEC-Malt
		# apply the mask
		echo "Applying mask to georeferenced DEM"
# TEMPORARY
		#/mn/moulin/project/Software/gdal-1.11.2/swig/python/scripts/gdal_calc.py -A tmp_msk.tif -B tmp_geo.tif --outfile=$dir\_Z.tif --calc="A*B"
		gdal_calc.py -A tmp_msk.tif -B tmp_geo.tif --outfile=$dir\_Z.tif --calc="B*(A>0)" --NoDataValue=-9999
		cp -v $dir\_Z.tif $outdir/$datestr #might be good to code orig. wd here.
		gdaldem hillshade $dir\_Z.tif $outdir/$datestr/$dir\_HS.tif
		gdal_calc.py -A $dir\_CORR.tif --outfile=$outdir/$datestr/$dir\_CORR.tif --calc="((A.astype(float)-127)/128)*100" --NoDataValue=-9999
		#cp -v $dir\_CORR.tif $outdir/$datestr #might be good to code orig. wd here.
		rm -v tmp_msk.tif tmp_geo.tif
		rm -v $dir\_Z.tif $dir\_CORR.tif
		cd ../
		if [ -d "Ortho-MEC-Malt" ]; then 			
			cd Ortho-MEC-Malt			
			gdal_calc.py -B tmp_mskDouble.tif -A tmp_V123.tif --outfile=$dir\_V123.tif --calc="((A!=255)*(A+1)+(A==255)*A)*(B>0)" --NoDataValue=0 --allBands=A
			#Expression complicated to solve real 0 values not being NoData and 255 no being +1-ed to 0
			rm -v tmp_V123.tif tmp_mskDouble.tif
			cp -v $dir\_V123.tif $outdir/$datestr
			rm -v $dir\_V123.tif
			cd ../
		fi
	else
		echo "No directory MEC-Malt found in $dir. Exiting."
	fi

#	if [ -d "Ortho-MEC-Malt" ]; then 
#		cd Ortho-MEC-Malt
#		echo $dir		
#		gdal_translate -a_srs "+proj=utm +zone=$1 +ellps=WGS84 +datum=WGS84 +units=m +no_defs" -a_nodata 0 Orthophotomosaic.tif $dir\_V123.tif
#		mv -v $dir\_V123.tif $outdir/$datestr #might be good to code orig. wd here.
#		cd ../
#	fi

	cd $basedir #back to original directory.
done 

# # next, create hillshade images.
# cd MaskedDEMs
# echo "getting hillshade images."
# for dem in $(ls *Z.tif); do
	# demname=${dem%.*}
	# echo "Creating hillshade for ${demname::-2}"
	# gdaldem hillshade $dem "${demname::-2}"_HS.tif
# done

# mkdir -p $basedir/Hillshades && mv *HS.tif $basedir/Hillshades

# # now, sort the DEMs into folders
# for dem in $(ls *Z.tif); do
	# #dates are in form mmddyyyy, want them in yyyymmdd.
	# tmpstr=${dem:11:8}
	# datestr=${tmpstr:4:4}${tmpstr:0:4}

	# mkdir -p $datestr
	# mv -v $dem $datestr
# done

#cd $basedir/OrthoImgs
# now it's trickier. we want image footprints. hopefully this works.
#mkdir -p tmp_masks
#for img in $(ls *.tif); do 
#	imgname=${img%.*}
#	gdal_calc.py -A $img --calc="1" --outfile=tmp_masks/$imgname\_mask.tif
#	cd tmp_masks
#	gdal_polygonize.py $imgname\_mask.tif -f "ESRI Shapefile" $imgname.shp
#	cd ..
#done

## now, we merge everything into a final file called ~/Footprints/Footprints.shp
#cd tmp_masks

#shapefileArr=($(ls *.shp)) # this way, it's actually an array!
## update the data table with: "granule", "dem_name", "folder", "date", and remove "DN"
#shpname=${shapefileArr[0]%.*}
#ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname DROP COLUMN DN" # will have a field DN after running gdal_polygonize up above.
#ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN granule  string(80)"
#ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN dem_name string(80)"
#ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN folder   string(80)"
#ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN date     date"
## now, prepare the inputs
#gname=$shpname
#tmpdt=${gname:11:8}
#yr=${tmpdt:4:4}
#mm=${tmpdt:0:2}
#dd=${tmpdt:2:2}
#dtfolder=$yr$mm$dd
## now, update the table with the right information.
#ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET granule = '$gname'"
#ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET dem_name = '${gname}_Z.tif'"
#ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET folder = '$basedir/MaskedDEMs/$yr$mm$dd'"
#ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET date = '$yr-$mm-$dd'"
## finally, copy the first file to Footprints.shp
#ogr2ogr -f 'ESRI Shapefile' $basedir/Footprints/Footprints.shp ${shapefileArr[0]}

## now, iterate over the rest of the array and merge them into ../../Footprints/Footprints.shp,
## adding the correct fields and data as we go.
#for shp in ${shapefileArr[@]:1}; do
#	shpname=${shp%.*}
#	ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname DROP COLUMN DN" # will have a field DN after running gdal_polygonize up above.
#	ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN granule  string(80)"
#	ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN dem_name string(80)"
#	ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN folder   string(80)"
#	ogrinfo $shpname.dbf -sql "ALTER TABLE $shpname ADD COLUMN date     date"
#	# now, prepare the inputs
#	gname=$shpname
#	tmpdt=${gname:11:8}
#	yr=${tmpdt:4:4}
#	mm=${tmpdt:0:2}
#	dd=${tmpdt:2:2}
#	dtfolder=$yr$mm$dd
#	# now, update the table with the right information.
#	ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET granule = '$gname'"
#	ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET dem_name = '${gname}_Z.tif'"
#	ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET folder = '$basedir/MaskedDEMs/$yr$mm$dd'"
#	ogrinfo $shpname.dbf -dialect SQLite -sql "UPDATE '$shpname' SET date = '$yr-$mm-$dd'"
#	# finally, merge the file to Footprints.shp
#	ogr2ogr -f 'ESRI Shapefile' -update -append $basedir/Footprints/Footprints.shp $shp
#done


## move up one level and remove the temporary directory tmp_masks
#cd ..
#rm -rf tmp_masks
# change back to basedir. Crack a beer, you're done.
echo "Processing is complete. Go enjoy a beverage, you've earned it."
cd $basedir
